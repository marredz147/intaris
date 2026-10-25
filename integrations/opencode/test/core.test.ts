import { afterEach, describe, expect, test } from "bun:test"
import { createCore, formatErrorDetail, loadApiKey } from "../src/core"
import { IntarisPlugin } from "../intaris"
import { chmodSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs"
import { join } from "node:path"

const originalFetch = globalThis.fetch
const originalEnv = { ...process.env }
afterEach(() => {
  globalThis.fetch = originalFetch
  process.env = { ...originalEnv }
})

function fixture(decision: unknown = { decision: "approve", call_id: "audit-1" }) {
  const calls: Array<{ path: string; body: any }> = []
  let audit: any = { user_decision: null }
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    const path = new URL(url).pathname
    const body = init.body ? JSON.parse(String(init.body)) : null
    calls.push({ path, body })
    const data = path === "/api/v1/intention" ? { session_id: "oc-s1" }
      : path === "/api/v1/evaluate" ? decision
      : path.startsWith("/api/v1/audit/") ? audit : {}
    return new Response(JSON.stringify(data), { status: 200 })
  }) as typeof fetch
  return { calls, resolve: (value: any) => { audit = value } }
}

async function core() {
  process.env.INTARIS_FAIL_OPEN = "false"
  delete process.env.INTARIS_API_KEY_FILE
  const hooks = await createCore({ worktree: "/project" })
  const before = (args: any = {}, sessionID = "s1", tool = "bash") => hooks["tool.execute.before"](
    { tool, sessionID, callID: "call-1" }, { args },
  )
  return { hooks, before }
}

describe("guard decisions", () => {
  test("approved call evaluated once and forwarded unchanged, including unavailable-tool text", async () => {
    const mock = fixture()
    const { before } = await core()
    await before({ command: "pwd", error: "unavailable tool" })
    expect(mock.calls.filter(c => c.path === "/api/v1/evaluate")).toHaveLength(1)
    expect(mock.calls.find(c => c.path === "/api/v1/evaluate")?.body.args.error).toBe("unavailable tool")
  })
  for (const result of [{ decision: "deny" }, { decision: "unexpected" }, {}, true]) {
    test(`blocks invalid or denied result ${JSON.stringify(result)}`, async () => {
      fixture(result)
      const { before } = await core()
      await expect(before()).rejects.toThrow("[intaris]")
    })
  }
  test("missing session id blocks before network call", async () => {
    const mock = fixture()
    const { before } = await core()
    await expect(before({}, "")).rejects.toThrow("Missing session identity")
    expect(mock.calls).toHaveLength(0)
  })
  test("evaluator outage and malformed JSON block by default", async () => {
    for (const response of [new Response("down", { status: 503 }), new Response("not-json")]) {
      fixture()
      const fetchBase = globalThis.fetch
      globalThis.fetch = (async (url: string, init: RequestInit) => new URL(url).pathname === "/api/v1/evaluate"
        ? response.clone() : fetchBase(url, init)) as typeof fetch
      const { before } = await core()
      await expect(before()).rejects.toThrow("Evaluation failed")
    }
  })
  test("structured evaluator errors are readable; raw bodies stay private", async () => {
    const cases: Array<[Response, string]> = [
      [Response.json({ detail: { code: "review_unavailable", message: "Automated review could not complete.", retryable: true } }, { status: 503 }),
        "Evaluation failed for bash: HTTP 503: review_unavailable: Automated review could not complete. (INTARIS_FAIL_OPEN=false)"],
      [new Response("private server response", { status: 503 }), "Evaluation failed for bash: HTTP 503 (INTARIS_FAIL_OPEN=false)"],
      [Response.json({ detail: "Session not found" }, { status: 404 }), "Evaluation rejected for bash: HTTP 404: Session not found"],
    ]
    for (const [response, message] of cases) {
      fixture()
      const fetchBase = globalThis.fetch
      globalThis.fetch = (async (url: string, init: RequestInit) => new URL(url).pathname === "/api/v1/evaluate"
        ? response.clone() : fetchBase(url, init)) as typeof fetch
      const { before } = await core()
      const error = await before().catch((err: Error) => err)
      expect(String(error)).not.toContain("[object Object]")
      expect((error as Error).message).toBe(`[intaris] ${message}`)
    }
  })
  test("error detail formatting keeps only Intaris-authored text", () => {
    expect(formatErrorDetail([{ msg: "field required", input: "secret" }, null, { loc: ["x"] }])).toBe("field required")
    expect(formatErrorDetail({ token: "secret" })).toBeNull()
    expect(formatErrorDetail({ message: "only message" })).toBe("only message")
    expect(formatErrorDetail(" ")).toBeNull()
    expect(formatErrorDetail(42)).toBeNull()
  })
  test("custom and MCP names remain intact for evaluation", async () => {
    const mock = fixture()
    const { before } = await core()
    await before({ path: "/tmp/a" }, "s1", "server_tool")
    await before({ input: "x" }, "s1", "custom-execute")
    expect(mock.calls.filter(c => c.path === "/api/v1/evaluate").map(c => c.body.tool)).toEqual(["server_tool", "custom-execute"])
  })
  test("failed reasoning submission blocks stale intention evaluation", async () => {
    const mock = fixture()
    const fetchBase = globalThis.fetch
    globalThis.fetch = (async (url: string, init: RequestInit) => new URL(url).pathname === "/api/v1/reasoning"
      ? new Response("private server response", { status: 503 }) : fetchBase(url, init)) as typeof fetch
    const { hooks, before } = await core()
    await hooks.event({ event: { type: "message.updated", properties: { info: { id: "m1", role: "user", sessionID: "s1" } } } })
    await hooks.event({ event: { type: "message.part.updated", properties: { part: { type: "text", text: "intent", messageID: "m1", sessionID: "s1" } } } })
    await expect(before()).rejects.toThrow("User intention submission failed")
    expect(mock.calls.filter(c => c.path === "/api/v1/evaluate")).toHaveLength(0)
  })
  test("escalation denied and cancellation before late approval", async () => {
    const mock = fixture({ decision: "escalate", call_id: "audit-1" })
    const { hooks, before } = await core()
    const pending = before()
    await new Promise(resolve => setTimeout(resolve, 20))
    mock.resolve({ user_decision: "approve" })
    hooks.dispose()
    await expect(pending).rejects.toThrow("Approval cancelled")
  })
  test("escalation denial blocks tool", async () => {
    const mock = fixture({ decision: "escalate", call_id: "audit-1" })
    const { hooks, before } = await core()
    const pending = before()
    mock.resolve({ user_decision: "deny" })
    await expect(pending).rejects.toThrow("DENIED by user")
    hooks.dispose()
  })
  for (const call_id of [undefined, null, 42, {}, "", " "]) {
    test(`invalid escalation call ID ${JSON.stringify(call_id)} blocks`, async () => {
      fixture({ decision: "escalate", call_id })
      const { before, hooks } = await core()
      await expect(before()).rejects.toThrow("Missing escalation call ID")
      hooks.dispose()
    })
  }
  test("per-call cancellation rejects escalation without unloading core", async () => {
    const mock = fixture({ decision: "escalate", call_id: "audit-1" })
    const { hooks } = await core()
    const controller = new AbortController()
    const pending = hooks["tool.execute.before"]({ tool: "bash", sessionID: "s1", callID: "c1", signal: controller.signal }, { args: {} })
    await new Promise(resolve => setTimeout(resolve, 20))
    controller.abort()
    mock.resolve({ user_decision: "approve" })
    await expect(pending).rejects.toThrow("cancelled")
    mock.resolve({ user_decision: "deny" })
    await expect(hooks["tool.execute.before"]({ tool: "bash", sessionID: "s1", callID: "c2" }, { args: {} })).rejects.toThrow("DENIED")
    hooks.dispose()
  })
  test("suspended session aborts before late reactivation", async () => {
    const mock = fixture({ decision: "deny", session_status: "suspended" })
    const { hooks } = await core()
    const controller = new AbortController()
    const pending = hooks["tool.execute.before"]({ tool: "bash", sessionID: "s1", callID: "c", signal: controller.signal }, { args: {} })
    await new Promise(resolve => setTimeout(resolve, 20))
    controller.abort()
    await expect(pending).rejects.toThrow("cancelled")
    expect(mock.calls.filter(c => c.path === "/api/v1/evaluate")).toHaveLength(1)
    hooks.dispose()
  })
  test("invalid reevaluation and malformed escalation ID after reactivation block", async () => {
    for (const reResult of [{ decision: "unexpected" }, { decision: "escalate", call_id: 123 }]) {
      fixture()
      let evaluations = 0
      const base = globalThis.fetch
      globalThis.fetch = (async (url: string, init: RequestInit) => {
        const path = new URL(url).pathname
        if (path === "/api/v1/evaluate") {
          evaluations++
          return Response.json(evaluations === 1 ? { decision: "deny", session_status: "suspended" } : reResult)
        }
        if (path === "/api/v1/session/oc-s1") return Response.json({ status: "active" })
        return base(url, init)
      }) as typeof fetch
      const { hooks, before } = await core()
      await expect(before()).rejects.toThrow(reResult.decision === "unexpected" ? "Invalid re-evaluation decision" : "Missing escalation call ID")
      hooks.dispose()
    }
  })
  test("threshold-one recording waits for persisted user event, and fails closed", async () => {
    for (const status of [200, 503]) {
      const mock = fixture()
      process.env.INTARIS_SESSION_RECORDING = "true"
      process.env.INTARIS_RECORDING_FLUSH_SIZE = "1"
      let release!: (response: Response) => void
      const base = globalThis.fetch
      globalThis.fetch = (async (url: string, init: RequestInit) => new URL(url).pathname.endsWith("/events")
        ? new Promise<Response>(resolve => { release = resolve }) : base(url, init)) as typeof fetch
      const { hooks, before } = await core()
      await hooks.event({ event: { type: "session.created", properties: { info: { id: "s1" } } } })
      await new Promise(resolve => setTimeout(resolve, 10))
      const reasoning = hooks["chat.message"]({ sessionID: "s1", messageID: "m" }, { parts: [{ type: "text", text: "synthetic intent" }] })
      await new Promise(resolve => setTimeout(resolve, 10))
      expect(mock.calls.filter(c => c.path === "/api/v1/reasoning")).toHaveLength(0)
      const tool = before()
      await new Promise(resolve => setTimeout(resolve, 10))
      expect(mock.calls.filter(c => c.path === "/api/v1/evaluate")).toHaveLength(0)
      release(new Response(JSON.stringify({}), { status }))
      await reasoning
      await new Promise(resolve => setTimeout(resolve, 10))
      release(Response.json({}))
      if (status === 503) await expect(tool).rejects.toThrow()
      else await tool
      hooks.dispose()
    }
  })
  test("parent deletion cancels waiting child", async () => {
    fixture({ decision: "escalate", call_id: "audit-1" })
    const { hooks } = await core()
    await hooks.event({ event: { type: "session.created", properties: { info: { id: "parent" } } } })
    await hooks.event({ event: { type: "session.created", properties: { info: { id: "child", parentID: "parent" } } } })
    const pending = hooks["tool.execute.before"]({ tool: "bash", sessionID: "child", callID: "c" }, { args: {} })
    await new Promise(resolve => setTimeout(resolve, 20))
    await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "parent" } } } })
    await expect(pending).rejects.toThrow("Approval cancelled")
    hooks.dispose()
  })
  test("V1 wrapper forwards disposal and stops recording flush timer", async () => {
    const mock = fixture({ decision: "escalate", call_id: "audit-1" })
    process.env.INTARIS_SESSION_RECORDING = "true"
    process.env.INTARIS_RECORDING_FLUSH_MS = "20"
    const hooks = await IntarisPlugin({ worktree: "/project" } as any)
    const pending = hooks["tool.execute.before"]!({ tool: "bash", sessionID: "s1", callID: "c" }, { args: {} })
    await new Promise(resolve => setTimeout(resolve, 30))
    mock.resolve({ user_decision: "approve" })
    await hooks.dispose?.()
    await expect(pending).rejects.toThrow("Approval cancelled")
    const count = mock.calls.filter(c => c.path.endsWith("/events")).length
    await new Promise(resolve => setTimeout(resolve, 50))
    expect(mock.calls.filter(c => c.path.endsWith("/events"))).toHaveLength(count)
  })
})

test("resumed session metadata overrides global root and includes child identity", async () => {
  const mock = fixture()
  const hooks = await createCore({ worktree: "/A", resolveSessionMetadata: async () => ({ workingDirectory: "/B", parentID: "parent", agent: "explorer", title: "Inspect B" }) })
  await hooks["tool.execute.before"]({ tool: "bash", sessionID: "resumed", callID: "c" }, { args: {} })
  const body = mock.calls.find(c => c.path === "/api/v1/intention")?.body
  expect(body.parent_session_id).toBe("oc-parent")
  expect(body.details.working_directory).toBe("/B")
  expect(body.details.agent_type).toBe("explorer")
  expect(body.policy.allow_paths).toContain("/B/*")
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "resumed" } } } })
  expect(mock.calls.find(c => c.path.endsWith("/agent-summary"))?.body.summary).toContain("Working directory: /B")
  hooks.dispose()
})

test("rename during pending creation preserves parent, child and delivered user context", async () => {
  const mock = fixture()
  const base = globalThis.fetch
  let release!: () => void
  let hold = true
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    if (hold && new URL(url).pathname === "/api/v1/intention" && JSON.parse(String(init.body)).session_id === "oc-parent") {
      hold = false
      await new Promise<void>(resolve => { release = resolve })
    }
    return base(url, init)
  }) as typeof fetch
  const hooks = await createCore({ worktree: "/project" })
  await hooks.event({ event: { type: "session.created", properties: { info: { id: "parent", title: "Old" } } } })
  await hooks.event({ event: { type: "session.created", properties: { info: { id: "child", title: "Child", parentID: "parent" } } } })
  await hooks.event({ event: { type: "message.updated", properties: { info: { id: "user-1", role: "user", sessionID: "parent" } } } })
  await hooks.event({ event: { type: "message.part.updated", properties: { part: { messageID: "user-1", sessionID: "parent", type: "text", text: "approved intent" } } } })
  await hooks.event({ event: { type: "session.updated", properties: { info: { id: "parent", title: "Renamed" } } } })
  expect(mock.calls.filter(c => c.path.endsWith("/status") && c.body?.status === "completed")).toHaveLength(0)
  release()
  await hooks["tool.execute.before"]({ tool: "read", sessionID: "parent", callID: "c" }, { args: { path: "/project/a" } })
  expect(mock.calls.find(c => c.path === "/api/v1/reasoning")?.body.content).toBe("User message: approved intent")
  expect(mock.calls.some(c => c.path === "/api/v1/session/oc-parent" && c.body?.intention === "Renamed")).toBe(true)
  expect(mock.calls.find(c => c.path === "/api/v1/intention" && c.body.session_id === "oc-child")?.body.parent_session_id).toBe("oc-parent")
  hooks.dispose()
})

test("created child tool waits for metadata before Intaris creation", async () => {
  const mock = fixture()
  let release!: (dir: string) => void
  const hooks = await createCore({ worktree: "/A", sessionDirectory: () => new Promise(resolve => { release = resolve }) })
  const created = hooks.event({ event: { type: "session.created", properties: { info: { id: "child", parentID: "parent", title: "Child" } } } })
  const pending = hooks["tool.execute.before"]({ tool: "bash", sessionID: "child", callID: "c" }, { args: {} })
  expect(mock.calls).toHaveLength(0)
  release("/B")
  await created
  await pending
  const body = mock.calls.find(c => c.path === "/api/v1/intention")?.body
  expect(body.parent_session_id).toBe("oc-parent")
  expect(body.details.working_directory).toBe("/B")
  expect(body.intention).toContain("Child")
  hooks.dispose()
})

test("file credential mode validates owner, file type, permissions, and contents without fallback", () => {
  const root = join(import.meta.dir, "../../../.tmp/core-tests")
  mkdirSync(root, { recursive: true, mode: 0o700 })
  const dir = mkdtempSync(join(root, "intaris-test-"))
  try {
    const path = join(dir, "key")
    writeFileSync(path, "dummy\n", { mode: 0o600 })
    expect(loadApiKey({ INTARIS_API_KEY_FILE: path, INTARIS_API_KEY: "fallback" })).toBe("dummy")
    chmodSync(path, 0o644)
    expect(() => loadApiKey({ INTARIS_API_KEY_FILE: path, INTARIS_API_KEY: "fallback" })).toThrow()
    chmodSync(path, 0o600)
    writeFileSync(path, "one\ntwo")
    expect(() => loadApiKey({ INTARIS_API_KEY_FILE: path })).toThrow()
    const link = join(dir, "link")
    symlinkSync(path, link)
    expect(() => loadApiKey({ INTARIS_API_KEY_FILE: link })).toThrow()
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
