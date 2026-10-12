import { afterEach, expect, test } from "bun:test"
import { Cause, Effect, Exit, Fiber, Option, Queue, Stream } from "effect"
import { Tool } from "@opencode/schema/tool"
import entry from "./intaris"

const originalFetch = globalThis.fetch
const originalEnv = { ...process.env }
afterEach(() => { globalThis.fetch = originalFetch; process.env = { ...originalEnv } })

function fixture(decision: unknown = { decision: "approve", call_id: "audit", path: "llm", latency_ms: 1 }, evalStatus = 200) {
  const requests: Array<{ path: string; body: any }> = []
  let audit = { user_decision: "approve" }
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    const path = new URL(url).pathname
    const body = init.body ? JSON.parse(String(init.body)) : null
    requests.push({ path, body })
    const response = path === "/api/v1/intention" ? { session_id: "oc-root" }
      : path === "/api/v1/evaluate" ? decision
      : path.startsWith("/api/v1/audit/") ? audit : {}
    return Response.json(response, { status: path === "/api/v1/evaluate" ? evalStatus : 200 })
  }) as typeof fetch
  const hooks: Record<string, (event: any) => any> = {}
  let registered = [{ id: "read" }, { id: "shell" }]
  let sessionPath = "/repo"
  let childAgent = "explorer"
  const ctx = {
    location: { directory: "/repo" },
    tool: {
      list: () => Effect.succeed(registered),
      hook: (name: string, callback: (event: any) => any) => Effect.sync(() => {
        hooks[name] = callback
        return { dispose: Effect.void }
      }),
    },
    session: {
      get: ({ sessionID }: any) => Effect.succeed({
        id: sessionID, location: { directory: sessionPath },
        title: "Session", ...(sessionID === "child" && { parentID: "root", agent: childAgent }),
      }),
      context: () => Effect.succeed([{ id: "user-1", type: "user", text: "Run the requested work" }]),
    },
    event: { subscribe: () => Stream.never },
  }
  const call = (tool = "execute", sessionID = "root", id = "call-1", input: any = { code: "await tools.shell({command:'pwd'})" }) => ({
    tool, sessionID, messageID: "assistant-1", id, input, agent: "build",
  })
  return { requests, hooks, ctx, call, setDecision: (next: unknown) => { decision = next }, setAudit: (next: any) => { audit = next }, setTools: (next: any) => { registered = next }, setSession: (path: string, agent: string) => { sessionPath = path; childAgent = agent } }
}

test("V1 default exports one server and retains direct tool evaluation", async () => {
  expect(Object.keys(entry).sort()).toEqual(["effect", "id", "server"])
  const fx = fixture()
  const hooks = await entry.server({ client: { app: { log: async () => ({}) }, tui: { showToast: async () => ({}) } }, worktree: "/repo", directory: "/repo" } as any)
  await hooks["tool.execute.before"]({ tool: "read", sessionID: "root", callID: "direct" }, { args: { path: "/repo/file" } })
  expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
  expect(fx.requests.find(r => r.path === "/api/v1/evaluate")?.body.tool).toBe("read")
  fx.setDecision({ decision: "deny", call_id: "deny-1", path: "llm", latency_ms: 1 })
  await expect(hooks["tool.execute.before"]({ tool: "read", sessionID: "root", callID: "denied" }, { args: { path: "/repo/other" } })).rejects.toThrow("DENIED")
  for (const loadOptions of [{}, { sessionInfo: "not a function", onDispose: "not a function" }]) {
    const loaded = await (entry.server as any)({ client: { app: { log: async () => ({}) }, tui: { showToast: async () => ({}) } }, worktree: "/repo", directory: "/repo" }, loadOptions)
    fx.setDecision({ decision: "approve", call_id: "another", path: "llm", latency_ms: 1 })
    await loaded["tool.execute.before"]({ tool: "read", sessionID: "root", callID: "next" }, { args: {} })
  }
})

test("V2 evaluates one outer input; nested sequential and parallel calls skip; direct and child remain isolated", async () => {
  const fx = fixture()
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    const before = fx.hooks["execute.before"]
    const after = fx.hooks["execute.after"]
    yield* before(fx.call())
    const nested = fx.call("shell", "root", "call-1", { command: "pwd" })
    yield* Effect.forkChild(Effect.gen(function* () { yield* before(nested); yield* after({ ...nested, status: "completed", result: { output: "pwd" } }) }))
    yield* Effect.forkChild(Effect.gen(function* () { yield* before({ ...nested, tool: "read" }); yield* after({ ...nested, tool: "read", status: "completed", result: { output: "ok" } }) }))
    yield* Effect.promise(() => new Promise(resolve => setTimeout(resolve, 20)))
    yield* after({ ...fx.call(), status: "completed", result: { output: { error: "script failed" } } })
    yield* before(fx.call("read", "root", "direct", { path: "/repo/a" }))
    yield* before(fx.call("read", "child", "call-1", { path: "/repo/b" }))
    const calls = fx.requests.filter(r => r.path === "/api/v1/evaluate")
    expect(calls).toHaveLength(3)
    expect(calls[0].body).toMatchObject({ tool: "execute", args: { code: "await tools.shell({command:'pwd'})" } })
    expect(calls[1].body.tool).toBe("read")
    expect(calls[2].body.session_id).toBe("oc-child")
    expect(fx.requests.find(r => r.path === "/api/v1/intention" && r.body.session_id === "oc-child")?.body.parent_session_id).toBe("oc-root")
  })))
})

test("V2 outer deny, HTTP outage and malformed approvals never permit nested execution", async () => {
  for (const [decision, status] of [
    [{ decision: "deny", call_id: "audit", path: "llm", latency_ms: 1 }, 200],
    [{ decision: "approve" }, 200],
    [{ detail: "backend unavailable" }, 503],
  ]) {
    const fx = fixture(decision, status as number)
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      yield* entry.effect(fx.ctx as any)
      const exit = yield* Effect.exit(fx.hooks["execute.before"](fx.call()))
      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isFailure(exit)) expect(Option.getOrThrow(Cause.findErrorOption(exit.cause))).toBeInstanceOf(Tool.Error)
      expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
    })))
  }
})

test("V2 approval records one outer result, including completed script errors", async () => {
  process.env.INTARIS_SESSION_RECORDING = "true"
  process.env.INTARIS_RECORDING_FLUSH_SIZE = "1"
  const fx = fixture()
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    yield* fx.hooks["execute.before"](fx.call())
    yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("shell", "root", "call-1", { command: "pwd" })))
    yield* Effect.promise(() => new Promise(resolve => setTimeout(resolve, 10)))
    yield* fx.hooks["execute.after"]({ ...fx.call(), status: "completed", result: { output: { error: "script failure" } } })
  })))
  await new Promise(resolve => setTimeout(resolve, 10))
  const recorded = fx.requests.filter(r => r.path.endsWith("/events")).flatMap(r => r.body)
  expect(recorded.filter((r: any) => r.type === "tool_call")).toHaveLength(1)
  expect(recorded.filter((r: any) => r.type === "tool_result")).toHaveLength(1)
  expect(recorded.find((r: any) => r.type === "tool_result")?.data.is_error).toBe(true)
  expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
})

test("registered execute catalog still admits the synthetic outer; nested execute, direct shell and child stay isolated", async () => {
  const fx = fixture()
  fx.setTools([{ id: "execute" }, { id: "shell" }])
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    yield* fx.hooks["execute.before"](fx.call())
    const nestedExecute = fx.call("execute", "root", "call-1", { code: "nested" })
    const nested = yield* Effect.forkChild(Effect.gen(function* () {
      yield* fx.hooks["execute.before"](nestedExecute)
      yield* fx.hooks["execute.after"]({ ...nestedExecute, status: "completed", result: { output: "nested" } })
    }))
    yield* Fiber.join(nested)
    yield* fx.hooks["execute.after"]({ ...fx.call(), status: "completed", result: { output: "outer" } })
    yield* fx.hooks["execute.before"](fx.call("shell", "root", "shell-1", { command: "pwd" }))
    yield* fx.hooks["execute.before"](fx.call("shell", "child", "call-1", { command: "pwd" }))
  })))
  const calls = fx.requests.filter(r => r.path === "/api/v1/evaluate")
  expect(calls).toHaveLength(3)
  expect(calls[1].body).toMatchObject({ tool: "bash", args: { command: "pwd", _intaris_opencode_tool: "shell" } })
  expect(calls[2].body.session_id).toBe("oc-child")
})

test("suspended outer execute ends without re-evaluation even when fail-open is configured", async () => {
  process.env.INTARIS_FAIL_OPEN = "true"
  const fx = fixture({ decision: "deny", session_status: "suspended", call_id: "audit", path: "fast", latency_ms: 1 })
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    const exit = yield* Effect.exit(fx.hooks["execute.before"](fx.call()))
    expect(Exit.isFailure(exit)).toBe(true)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
    expect(fx.requests.filter(r => r.path.startsWith("/api/v1/session/") && r.path.endsWith("/status"))).toHaveLength(0)
  })))
})

test("failing or cancelled owner cleans up; reused tuple needs fresh evaluation", async () => {
  const fx = fixture()
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    const outer = fx.call()
    const first = yield* Effect.forkChild(Effect.gen(function* () {
      yield* fx.hooks["execute.before"](outer)
      return yield* Effect.fail(new Error("later step failed"))
    }))
    expect(Exit.isFailure(yield* Fiber.await(first))).toBe(true)
    const second = yield* Effect.forkChild(Effect.gen(function* () {
      yield* fx.hooks["execute.before"](outer)
      return yield* Effect.never
    }))
    yield* Effect.promise(() => new Promise(resolve => setTimeout(resolve, 10)))
    yield* Fiber.interrupt(second)
    yield* fx.hooks["execute.before"](outer)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(3)
    yield* fx.hooks["execute.after"]({ ...outer, status: "completed", result: { output: "ok" } })
  })))
})

test("duplicate outer tuple is rejected while approval is active", async () => {
  const fx = fixture()
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    yield* fx.hooks["execute.before"](fx.call())
    const duplicate = yield* Effect.exit(fx.hooks["execute.before"](fx.call()))
    expect(Exit.isFailure(duplicate)).toBe(true)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
    yield* fx.hooks["execute.after"]({ ...fx.call(), status: "completed", result: { output: "ok" } })
  })))
})

test("escalation approval opens exactly the original outer family", async () => {
  const fx = fixture({ decision: "escalate", call_id: "audit", path: "alignment", latency_ms: 1 })
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    yield* fx.hooks["execute.before"](fx.call())
    const nested = fx.call("shell", "root", "call-1", { command: "pwd" })
    const child = yield* Effect.forkChild(fx.hooks["execute.before"](nested))
    yield* Fiber.join(child)
    yield* fx.hooks["execute.after"]({ ...fx.call(), status: "completed", result: { output: "done" } })
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
    expect(fx.requests.filter(r => r.path === "/api/v1/audit/audit")).toHaveLength(1)
  })))
}, 5000)

test("pending approval blocks nested calls without a second evaluation", async () => {
  const fx = fixture()
  const base = globalThis.fetch
  let release!: () => void
  let started!: () => void
  const evaluating = new Promise<void>(resolve => { started = resolve })
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    if (new URL(url).pathname === "/api/v1/evaluate") {
      started()
      await new Promise<void>(resolve => { release = resolve })
    }
    return base(url, init)
  }) as typeof fetch
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    const outer = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call()))
    yield* Effect.promise(() => evaluating)
    const nested = yield* Effect.exit(fx.hooks["execute.before"](fx.call("shell", "root", "call-1", { command: "pwd" })))
    expect(Exit.isFailure(nested)).toBe(true)
    release()
    yield* Fiber.join(outer)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
  })))
})

test("outer evaluator outage cannot use legacy FAIL_OPEN", async () => {
  process.env.INTARIS_FAIL_OPEN = "true"
  const fx = fixture({ detail: "unavailable" }, 503)
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    expect(Exit.isFailure(yield* Effect.exit(fx.hooks["execute.before"](fx.call())))).toBe(true)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
  })))
})

test("cancelled escalation cannot admit a late approval or a stale nested call", async () => {
  const fx = fixture({ decision: "escalate", call_id: "audit", path: "llm", latency_ms: 1 })
  fx.setAudit({ user_decision: null })
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    const first = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call()))
    yield* Effect.promise(() => new Promise(resolve => setTimeout(resolve, 15)))
    yield* Fiber.interrupt(first)
    fx.setAudit({ user_decision: "approve" })
    fx.setDecision({ decision: "approve", call_id: "fresh", path: "llm", latency_ms: 1 })
    yield* fx.hooks["execute.before"](fx.call())
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(2)
    expect(fx.requests.filter(r => r.path === "/api/v1/audit/audit")).toHaveLength(0)
    yield* fx.hooks["execute.after"]({ ...fx.call(), status: "completed", result: { output: "ok" } })
  })))
})

test("child idle then resumed with same user ID refreshes metadata and user context", async () => {
  const fx = fixture()
  const queue = await Effect.runPromise(Queue.unbounded<any>())
  ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    yield* fx.hooks["execute.before"](fx.call("read", "child", "first", { path: "/repo/a" }))
    yield* Queue.offer(queue, { type: "session.execution.succeeded", data: { sessionID: "child" } })
    yield* Effect.promise(async () => {
      for (let n = 0; n < 100 && !fx.requests.some(r => r.path === "/api/v1/session/oc-child/agent-summary"); n++) {
        await new Promise(resolve => setTimeout(resolve, 2))
      }
    })
    expect(fx.requests.some(r => r.path === "/api/v1/session/oc-child/agent-summary")).toBe(true)
    fx.setSession("/resumed", "reviewer")
    yield* fx.hooks["execute.before"](fx.call("read", "child", "second", { path: "/resumed/a" }))
    const created = fx.requests.filter(r => r.path === "/api/v1/intention" && r.body.session_id === "oc-child")
    expect(created).toHaveLength(2)
    expect(created[1].body.details).toMatchObject({ working_directory: "/resumed", agent_type: "reviewer" })
    expect(created[1].body.parent_session_id).toBe("oc-root")
    expect(fx.requests.filter(r => r.path === "/api/v1/reasoning")).toHaveLength(2)
  })))
})

test("cancelled preparation cannot resurrect a tuple or mutate later session metadata", async () => {
  for (const hold of ["context", "metadata"] as const) {
    const fx = fixture()
    let started!: () => void
    let release!: () => void
    const waiting = new Promise<void>(resolve => { started = resolve })
    let first = true
    if (hold === "context") {
      const original = fx.ctx.session.context
      ;(fx.ctx.session as any).context = () => {
        if (!first) return original()
        first = false
        return Effect.promise(() => new Promise<any[]>(resolve => {
          release = () => resolve([{ id: "user-1", type: "user", text: "stale" }])
          started()
        }))
      }
    } else {
      const original = fx.ctx.session.get
      ;(fx.ctx.session as any).get = ({ sessionID }: any) => {
        if (!first) return Effect.succeed({ id: sessionID, location: { directory: "/fresh" }, parentID: "root", agent: "reviewer", title: "new" })
        first = false
        return Effect.promise(() => new Promise<any>(resolve => {
          release = () => resolve({ id: sessionID, location: { directory: "/stale" }, parentID: "wrong", agent: "stale", title: "old" })
          started()
        }))
      }
      expect(typeof original).toBe("function")
    }
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      yield* entry.effect(fx.ctx as any)
      const outer = fx.call("execute", "child", "reused")
      const cancelled = yield* Effect.forkChild(fx.hooks["execute.before"](outer))
      yield* Effect.promise(() => waiting)
      yield* Fiber.interrupt(cancelled)
      yield* fx.hooks["execute.before"](outer)
      release()
      yield* Effect.promise(() => new Promise(resolve => setTimeout(resolve, 10)))
      const calls = fx.requests.filter(r => r.path === "/api/v1/evaluate")
      expect(calls).toHaveLength(1)
      const created = fx.requests.find(r => r.path === "/api/v1/intention" && r.body.session_id === "oc-child")?.body
      expect(created.details.working_directory).toBe(hold === "metadata" ? "/fresh" : "/repo")
      expect(created.parent_session_id).toBe("oc-root")
      yield* fx.hooks["execute.after"]({ ...outer, status: "completed", result: { output: "ok" } })
    })))
  }
})
