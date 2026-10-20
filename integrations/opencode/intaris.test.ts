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
  let reasoning: unknown = { ok: true, call_id: "reasoning-1" }
  let eventStatus = 200
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    const path = new URL(url).pathname
    const body = init.body ? JSON.parse(String(init.body)) : null
    requests.push({ path, body })
    const response = path === "/api/v1/intention" ? { session_id: "oc-root" }
      : path === "/api/v1/evaluate" ? decision
      : path === "/api/v1/reasoning" ? reasoning
      : path.startsWith("/api/v1/audit/") ? audit : {}
    return Response.json(response, { status: path === "/api/v1/evaluate" ? evalStatus : path.endsWith("/events") ? eventStatus : 200 })
  }) as typeof fetch
  const hooks: Record<string, (event: any) => any> = {}
  let registered = [{ id: "read" }, { id: "shell" }]
  let sessionPath = "/repo"
  let childAgent = "explorer"
  let messages: any[] = [{ id: "user-1", type: "user", text: "Run the requested work" }]
  const sessionHooks: Record<string, (event: any) => any> = {}
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
      context: () => Effect.succeed(messages),
      hook: (name: string, callback: (event: any) => any) => Effect.sync(() => {
        sessionHooks[name] = callback
        return { dispose: Effect.void }
      }),
    },
    event: { subscribe: () => Stream.never },
  }
  const call = (tool = "execute", sessionID = "root", id = "call-1", input: any = { code: "await tools.shell({command:'pwd'})" }) => ({
    tool, sessionID, messageID: "assistant-1", id, input, agent: "build",
  })
  return { requests, hooks, sessionHooks, ctx, call, setContext: (next: any[]) => { messages = next }, setReasoning: (next: unknown) => { reasoning = next }, setEventStatus: (next: number) => { eventStatus = next }, setDecision: (next: unknown) => { decision = next }, setAudit: (next: any) => { audit = next }, setTools: (next: any) => { registered = next }, setSession: (path: string, agent: string) => { sessionPath = path; childAgent = agent } }
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

test("failed and non-shutdown interrupted executions idle the session; shutdown does not", async () => {
  const fx = fixture()
  const queue = await Effect.runPromise(Queue.unbounded<any>())
  ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    for (const id of ["fa", "in", "sh"]) {
      yield* Queue.offer(queue, { type: "session.created", durable: { aggregateID: id }, data: { title: "T" } })
    }
    yield* Effect.promise(async () => {
      for (let n = 0; n < 100; n++) {
        if (["fa", "in", "sh"].every(id => fx.requests.some((r: any) => r.path === "/api/v1/intention" && r.body?.session_id === `oc-${id}`))) return
        await new Promise(resolve => setTimeout(resolve, 2))
      }
    })
    yield* Queue.offer(queue, { type: "session.execution.failed", data: { sessionID: "fa", error: { type: "error", message: "boom" } } })
    yield* Queue.offer(queue, { type: "session.execution.interrupted", data: { sessionID: "in", reason: "user" } })
    yield* Queue.offer(queue, { type: "session.execution.interrupted", data: { sessionID: "sh", reason: "shutdown" } })
    yield* Effect.promise(async () => {
      for (let n = 0; n < 100; n++) {
        const idle = (id: string) => fx.requests.some((r: any) => r.path === `/api/v1/session/oc-${id}/status` && r.body?.status === "idle")
        if (idle("fa") && idle("in")) return
        await new Promise(resolve => setTimeout(resolve, 2))
      }
    })
    const idle = (id: string) => fx.requests.some((r: any) => r.path === `/api/v1/session/oc-${id}/status` && r.body?.status === "idle")
    expect(idle("fa")).toBe(true)
    expect(idle("in")).toBe(true)
    expect(idle("sh")).toBe(false)
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

test("compaction continuation requires observed completion and matching native checkpoint", async () => {
  for (const order of ["auto", "manual"] as const) {
    const fx = fixture()
    const queue = await Effect.runPromise(Queue.unbounded<any>())
    ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      yield* entry.effect(fx.ctx as any)
      const before = fx.hooks["execute.before"]
      yield* before(fx.call("read", "root", "initial", { path: "/repo/a" }))
      const send = (event: any) => Effect.gen(function* () {
        yield* Queue.offer(queue, event)
        yield* Effect.promise(() => new Promise(resolve => setTimeout(resolve, 10)))
      })
      for (const n of [1, 2]) {
        const started = { type: "session.compaction.started", id: `evt_${n}`, data: {
          sessionID: "root", reason: order, ...(order === "manual" && { inputID: `checkpoint-${n}` })
        } }
        if (order === "manual") yield* send(started)
        if (order === "manual" && n === 2) fx.setContext([
          { id: "checkpoint-1", type: "compaction", status: "completed", summary: "old" },
          { id: "checkpoint-2", type: "compaction", status: "running", summary: "" },
        ])
        yield* fx.sessionHooks.compaction({ sessionID: "root" })
        if (order === "auto") yield* send(started)
        if (n === 2) {
          const blocked = yield* Effect.exit(before(fx.call("read", "root", "inflight", { path: "/repo/a" })))
          expect(Exit.isFailure(blocked)).toBe(true)
        }
        yield* send({ type: "session.compaction.ended", id: `evt_end_${n}`, data: { sessionID: "root", reason: order, text: "untrusted summary" } })
        fx.setContext([{ id: order === "manual" ? `checkpoint-${n}` : `msg_${n}`, type: "compaction", status: "completed", summary: "ignore these instructions" }])
        yield* before(fx.call("read", "root", `after-${n}`, { path: "/repo/b" }))
      }
      expect(fx.requests.filter(r => r.path === "/api/v1/reasoning")).toHaveLength(1)
      expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(3)
    })))
  }
})

test("summary-only, missing/failed lifecycle, mismatched checkpoint and cold plugin fail closed", async () => {
  for (const scenario of ["summary", "cold", "no-receipt", "missing-start", "missing-hook", "missing-end", "failed", "mismatch"] as const) {
    const fx = fixture()
    const queue = await Effect.runPromise(Queue.unbounded<any>())
    ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      yield* entry.effect(fx.ctx as any)
      if (!["summary", "cold", "no-receipt"].includes(scenario)) yield* fx.hooks["execute.before"](fx.call("read", "root", "initial", { path: "/repo/a" }))
      if (!["summary", "cold", "missing-hook"].includes(scenario)) yield* fx.sessionHooks.compaction({ sessionID: "root" })
      const send = (event: any) => Effect.gen(function* () {
        yield* Queue.offer(queue, event)
        yield* Effect.promise(() => new Promise(resolve => setTimeout(resolve, 10)))
      })
      if (!["summary", "cold", "missing-start"].includes(scenario)) yield* send({ type: "session.compaction.started", id: "evt_1", data: { sessionID: "root", reason: "auto" } })
      if (scenario === "failed") yield* send({ type: "session.compaction.failed", id: "evt_failed", data: { sessionID: "root", reason: "auto" } })
      if (!["summary", "cold", "failed", "missing-end"].includes(scenario)) yield* send({ type: "session.compaction.ended", id: "evt_end", data: { sessionID: "root", reason: "auto", text: "summary" } })
      fx.setContext([{ id: scenario === "mismatch" ? "msg_other" : "msg_1", type: "compaction", status: "completed", summary: "summary only" }])
      expect(Exit.isFailure(yield* Effect.exit(fx.hooks["execute.before"](fx.call("read", "root", "after", { path: "/repo/b" }))))).toBe(true)
      expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(["summary", "cold", "no-receipt"].includes(scenario) ? 0 : 1)
    })))
  }
})

test("newer failed or empty user cannot inherit U1 receipt for direct or outer execution", async () => {
  for (const tool of ["read", "execute"] as const) {
    const fx = fixture()
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      yield* entry.effect(fx.ctx as any)
      yield* fx.hooks["execute.before"](fx.call("read", "root", "first", { path: "/repo/a" }))
      fx.setContext([{ id: "user-2", type: "user", text: "New request" }])
      const base = globalThis.fetch
      globalThis.fetch = (async (url: string, init: RequestInit) => new URL(url).pathname === "/api/v1/reasoning"
        ? Response.json({ detail: "unavailable" }, { status: 503 }) : base(url, init)) as typeof fetch
      expect(Exit.isFailure(yield* Effect.exit(fx.hooks["execute.before"](fx.call(tool, "root", "failed"))))).toBe(true)
      fx.setContext([{ id: "user-3", type: "user", text: "" }])
      expect(Exit.isFailure(yield* Effect.exit(fx.hooks["execute.before"](fx.call(tool, "root", "empty"))))).toBe(true)
      expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
      globalThis.fetch = base
      fx.setContext([{ id: "user-4", type: "user", text: "Fresh instruction" }])
      yield* fx.hooks["execute.before"](fx.call(tool, "root", "fresh"))
      expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(2)
    })))
  }
})

test("delivered user blocks compacted receipt; synthetic and manual compaction do not masquerade as users", async () => {
  const fx = fixture()
  const queue = await Effect.runPromise(Queue.unbounded<any>())
  ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    yield* fx.hooks["execute.before"](fx.call("read", "root", "initial", { path: "/repo/a" }))
    const send = (type: string, inboxID: string, item?: any) => Effect.gen(function* () {
      yield* Queue.offer(queue, { type, data: { sessionID: "root", inboxID, ...(item && { item }) } })
      yield* Effect.promise(() => new Promise(resolve => setTimeout(resolve, 10)))
    })
    yield* fx.sessionHooks.compaction({ sessionID: "root" })
    yield* Queue.offer(queue, { type: "session.compaction.started", id: "evt_1", data: { sessionID: "root", reason: "auto" } })
    yield* Effect.promise(() => new Promise(resolve => setTimeout(resolve, 10)))
    yield* Queue.offer(queue, { type: "session.compaction.ended", id: "evt_end", data: { sessionID: "root", reason: "auto" } })
    yield* Effect.promise(() => new Promise(resolve => setTimeout(resolve, 10)))
    fx.setContext([{ id: "msg_1", type: "compaction", status: "completed", summary: "not a user" }])
    yield* send("session.inbox.enqueued", "user-2", { type: "user", payload: { text: "new request" } })
    expect(Exit.isSuccess(yield* Effect.exit(fx.hooks["execute.before"](fx.call("read", "root", "continued", { path: "/repo/b" }))))).toBe(true)
    yield* send("session.inbox.cancelled", "user-2")
    yield* send("session.inbox.enqueued", "synthetic-1", { type: "synthetic", payload: { text: "generated" } })
    yield* send("session.inbox.delivered", "synthetic-1")
    yield* send("session.inbox.enqueued", "manual-1", { type: "compaction", payload: {} })
    yield* send("session.inbox.delivered", "manual-1")
    expect(Exit.isSuccess(yield* Effect.exit(fx.hooks["execute.before"](fx.call("read", "root", "continued-again", { path: "/repo/b" }))))).toBe(true)
    yield* send("session.inbox.enqueued", "user-2", { type: "user", payload: { text: "new request" } })
    yield* send("session.inbox.delivered", "user-2")
    expect(Exit.isFailure(yield* Effect.exit(fx.hooks["execute.before"](fx.call("read", "root", "blocked", { path: "/repo/b" }))))).toBe(true)
    yield* send("session.inbox.cancelled", "user-2")
    expect(Exit.isFailure(yield* Effect.exit(fx.hooks["execute.before"](fx.call("read", "root", "still-blocked", { path: "/repo/b" }))))).toBe(true)
    yield* send("session.inbox.enqueued", "user-3", { type: "user", payload: { text: "Fresh instruction" } })
    yield* send("session.inbox.delivered", "user-3")
    fx.setContext([{ id: "user-3", type: "user", text: "Fresh instruction" }])
    yield* fx.hooks["execute.before"](fx.call("read", "root", "fresh", { path: "/repo/c" }))
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(4)
  })))
})

test("V2 recording fence fails closed on upload failure and retries the same selected native user", async () => {
  process.env.INTARIS_SESSION_RECORDING = "true"
  process.env.INTARIS_RECORDING_FLUSH_SIZE = "1"
  const fx = fixture()
  fx.setEventStatus(503)
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    expect(Exit.isFailure(yield* Effect.exit(fx.hooks["execute.before"](fx.call("read", "root", "failed-u1", { path: "/repo/a" }))))).toBe(true)
    expect(fx.requests.filter(r => r.path === "/api/v1/reasoning")).toHaveLength(0)
    fx.setEventStatus(200)
    yield* fx.hooks["execute.before"](fx.call("read", "root", "u1", { path: "/repo/a" }))
    fx.setContext([{ id: "user-2", type: "user", text: "Latest explicit instruction" }])
    yield* fx.hooks["execute.before"](fx.call("read", "root", "u2", { path: "/repo/b" }))
    const reasoning = fx.requests.filter(r => r.path === "/api/v1/reasoning")
    expect(reasoning.map(r => r.body.content)).toEqual([
      "User message: Run the requested work", "User message: Latest explicit instruction",
    ])
    expect(reasoning.every(r => !r.body.from_events)).toBe(true)
    expect(fx.requests.filter(r => r.path.endsWith("/events")).length).toBeGreaterThan(1)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(2)
  })))
})

test("V2 no-op or invalid reasoning acknowledgement cannot evaluate or seed compaction proof", async () => {
  process.env.INTARIS_SESSION_RECORDING = "true"
  process.env.INTARIS_RECORDING_FLUSH_SIZE = "1"
  for (const response of [{ ok: true, call_id: "" }, { ok: false, call_id: "older" }, {}]) {
    const fx = fixture()
    fx.setReasoning(response)
    const queue = await Effect.runPromise(Queue.unbounded<any>())
    ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      yield* entry.effect(fx.ctx as any)
      expect(Exit.isFailure(yield* Effect.exit(fx.hooks["execute.before"](fx.call("read", "root", "failed", { path: "/repo/a" }))))).toBe(true)
      expect(fx.requests.find(r => r.path === "/api/v1/reasoning")?.body).toMatchObject({ content: "User message: Run the requested work" })
      yield* fx.sessionHooks.compaction({ sessionID: "root" })
      yield* Queue.offer(queue, { type: "session.compaction.started", id: "evt_1", data: { sessionID: "root", reason: "auto" } })
      yield* Queue.offer(queue, { type: "session.compaction.ended", id: "evt_end", data: { sessionID: "root", reason: "auto" } })
      yield* Effect.promise(() => new Promise(resolve => setTimeout(resolve, 15)))
      fx.setContext([{ id: "msg_1", type: "compaction", status: "completed", summary: "older message" }])
      expect(Exit.isFailure(yield* Effect.exit(fx.hooks["execute.before"](fx.call("read", "root", "after", { path: "/repo/b" }))))).toBe(true)
      expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(0)
    })))
  }
})

test("completed proof is revoked by move, revert, unknown delivery or event-processing error", async () => {
  for (const change of ["session.moved", "session.revert.staged", "session.revert.cleared", "session.revert.committed", "session.inbox.delivered", "session.renamed"] as const) {
    const fx = fixture()
    const queue = await Effect.runPromise(Queue.unbounded<any>())
    ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      yield* entry.effect(fx.ctx as any)
      yield* fx.hooks["execute.before"](fx.call("read", "root", "initial", { path: "/repo/a" }))
      yield* fx.sessionHooks.compaction({ sessionID: "root" })
      const send = (event: any) => Effect.gen(function* () {
        yield* Queue.offer(queue, event)
        yield* Effect.promise(() => new Promise(resolve => setTimeout(resolve, 10)))
      })
      yield* send({ type: "session.compaction.started", id: "evt_1", data: { sessionID: "root", reason: "auto" } })
      yield* send({ type: "session.compaction.ended", id: "evt_end", data: { sessionID: "root", reason: "auto" } })
      fx.setContext([{ id: "msg_1", type: "compaction", status: "completed", summary: "old" }])
      yield* fx.hooks["execute.before"](fx.call("read", "root", "continued", { path: "/repo/b" }))
      yield* send({ type: change, data: change === "session.renamed" ? null : { sessionID: "root", inboxID: "unknown" } })
      expect(Exit.isFailure(yield* Effect.exit(fx.hooks["execute.before"](fx.call("read", "root", "blocked", { path: "/repo/c" }))))).toBe(true)
      expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(2)
    })))
  }
})

test("V2 records the selected user before reasoning so its own recording cannot stale the intention", async () => {
  process.env.INTARIS_SESSION_RECORDING = "true"
  process.env.INTARIS_RECORDING_FLUSH_SIZE = "1"
  const fx = fixture()
  const base = globalThis.fetch
  let releaseEvent!: () => void
  let eventStarted!: () => void
  let eventSettled!: () => void
  const started = new Promise<void>(resolve => { eventStarted = resolve })
  const settled = new Promise<void>(resolve => { eventSettled = resolve })
  let latestUserSeq = 0
  let intention = "old intention"
  let observedSeq = -1
  let submittedText = ""
  let evaluations = 0
  let deferredFirstUserBatch = false
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    const path = new URL(url).pathname
    if (path.endsWith("/events")) {
      const events = JSON.parse(String(init.body)) as Array<{ type: string; data: { role?: string } }>
      if (events.some(event => event.type === "message" && event.data.role === "user")) {
        if (!deferredFirstUserBatch) {
          deferredFirstUserBatch = true
          eventStarted()
          await new Promise<void>(resolve => { releaseEvent = resolve })
        }
        latestUserSeq++
        eventSettled()
      }
    }
    if (path === "/api/v1/reasoning") {
      observedSeq = latestUserSeq // IntentionBarrier captures the durable user-message revision.
      submittedText = JSON.parse(String(init.body)).content
    }
    if (path === "/api/v1/evaluate") {
      evaluations++
      await settled
      if (observedSeq === latestUserSeq) intention = submittedText
      return Response.json({ decision: intention === submittedText ? "approve" : "deny", call_id: "eval", path: "llm", latency_ms: 1 })
    }
    return base(url, init)
  }) as typeof fetch
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    const call = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "ordered", { path: "/repo/a" })))
    yield* Effect.promise(() => started)
    const reasoningBeforeRecordingSettles = fx.requests.filter(r => r.path === "/api/v1/reasoning").length
    releaseEvent()
    const exit = yield* Fiber.await(call)
    expect(reasoningBeforeRecordingSettles).toBe(0)
    expect(Exit.isSuccess(exit)).toBe(true)
    expect(observedSeq).toBe(latestUserSeq)
    expect(observedSeq).toBeGreaterThan(0)
    expect(intention).toBe("User message: Run the requested work")
    expect(evaluations).toBe(1)
  })))
})

test("concurrent V2 tools share a user reasoning submission; a duplicate late failure cannot poison its receipt", async () => {
  const fx = fixture()
  const base = globalThis.fetch
  let releaseFirst!: () => void
  let releaseDuplicate!: () => void
  let firstStarted!: () => void
  const started = new Promise<void>(resolve => { firstStarted = resolve })
  let secondContextSeen!: () => void
  const secondContext = new Promise<void>(resolve => { secondContextSeen = resolve })
  const context = fx.ctx.session.context
  let contextCalls = 0
  fx.ctx.session.context = () => {
    if (++contextCalls === 2) secondContextSeen()
    return context()
  }
  let submissions = 0
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    if (new URL(url).pathname === "/api/v1/reasoning") {
      submissions++
      if (submissions === 1) {
        firstStarted()
        await new Promise<void>(resolve => { releaseFirst = resolve })
        return Response.json({ ok: true, call_id: "receipt-u1" })
      }
      await new Promise<void>(resolve => { releaseDuplicate = resolve })
      return Response.json({ detail: "duplicate request failed" }, { status: 503 })
    }
    return base(url, init)
  }) as typeof fetch
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    const first = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "parallel-1", { path: "/repo/a" })))
    yield* Effect.promise(() => started)
    const second = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "parallel-2", { path: "/repo/b" })))
    yield* Effect.promise(() => secondContext)
    yield* Effect.promise(() => new Promise<void>(resolve => queueMicrotask(resolve)))
    releaseFirst()
    const firstExit = yield* Fiber.await(first)
    releaseDuplicate?.()
    const secondExit = yield* Fiber.await(second)
    const thirdExit = yield* Effect.exit(fx.hooks["execute.before"](fx.call("read", "root", "later", { path: "/repo/c" })))
    expect(Exit.isSuccess(firstExit)).toBe(true)
    expect(Exit.isSuccess(secondExit)).toBe(true)
    expect(Exit.isSuccess(thirdExit)).toBe(true)
    expect(submissions).toBe(1)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(3)
  })))
})

test("V2 user fence waits for an already-inflight earlier recording upload", async () => {
  process.env.INTARIS_SESSION_RECORDING = "true"
  process.env.INTARIS_RECORDING_FLUSH_SIZE = "1"
  const fx = fixture()
  const queue = await Effect.runPromise(Queue.unbounded<any>())
  ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
  const base = globalThis.fetch
  let releasePrior!: () => void
  let priorStarted!: () => void
  const started = new Promise<void>(resolve => { priorStarted = resolve })
  let sessionStarted!: () => void
  const sessionReady = new Promise<void>(resolve => { sessionStarted = resolve })
  const uploads: Array<Array<{ type: string; data: { role?: string } }>> = []
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    if (new URL(url).pathname === "/api/v1/intention") sessionStarted()
    if (new URL(url).pathname.endsWith("/events")) {
      const events = JSON.parse(String(init.body)) as Array<{ type: string; data: { role?: string } }>
      uploads.push(events)
      if (uploads.length === 1) {
        priorStarted()
        await new Promise<void>(resolve => { releasePrior = resolve })
      }
    }
    return base(url, init)
  }) as typeof fetch
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    yield* Queue.offer(queue, { type: "session.created", durable: { aggregateID: "root" }, data: { title: "Session" } })
    yield* Effect.promise(() => sessionReady)
    yield* Queue.offer(queue, { type: "session.text.ended", data: { sessionID: "root", assistantMessageID: "assistant-previous", text: "previous" } })
    yield* Effect.promise(() => started)
    const call = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "after-inflight", { path: "/repo/a" })))
    yield* Effect.promise(() => new Promise<void>(resolve => queueMicrotask(resolve)))
    const beforePriorSettles = { uploads: uploads.length, reasoning: fx.requests.filter(r => r.path === "/api/v1/reasoning").length }
    releasePrior()
    expect(Exit.isSuccess(yield* Fiber.await(call))).toBe(true)
    expect(beforePriorSettles).toEqual({ uploads: 1, reasoning: 0 })
    expect(uploads.slice(1).flat().filter(event => event.type === "message" && event.data.role === "user").length).toBeGreaterThan(0)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
  })))
})

test("newer U2 supersedes an in-flight U1 reasoning response without inheriting its receipt", async () => {
  const fx = fixture()
  const context = fx.ctx.session.context
  let u2ContextSeen!: () => void
  const u2Context = new Promise<void>(resolve => { u2ContextSeen = resolve })
  let contextCalls = 0
  fx.ctx.session.context = () => {
    if (++contextCalls === 2) u2ContextSeen()
    return context()
  }
  const base = globalThis.fetch
  let releaseU1!: () => void
  let u1Started!: () => void
  const started = new Promise<void>(resolve => { u1Started = resolve })
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    if (new URL(url).pathname === "/api/v1/reasoning") {
      if (JSON.parse(String(init.body)).content === "User message: Run the requested work") {
        u1Started()
        await new Promise<void>(resolve => { releaseU1 = resolve })
      }
      return Response.json({ ok: true, call_id: "receipt" })
    }
    return base(url, init)
  }) as typeof fetch
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    const u1 = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "u1", { path: "/repo/a" })))
    yield* Effect.promise(() => started)
    fx.setContext([{ id: "user-2", type: "user", text: "Newer instruction" }])
    const u2 = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "u2", { path: "/repo/b" })))
    yield* Effect.promise(() => u2Context)
    yield* Effect.promise(() => new Promise<void>(resolve => queueMicrotask(resolve)))
    releaseU1()
    expect(Exit.isFailure(yield* Fiber.await(u1))).toBe(true)
    expect(Exit.isSuccess(yield* Fiber.await(u2))).toBe(true)
    expect(Exit.isSuccess(yield* Effect.exit(fx.hooks["execute.before"](fx.call("read", "root", "later", { path: "/repo/c" }))))).toBe(true)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(2)
  })))
})

test("cancelling one V2 tool does not cancel the shared user submission needed by another", async () => {
  const fx = fixture()
  const base = globalThis.fetch
  let release!: () => void
  let started!: () => void
  const reasoningStarted = new Promise<void>(resolve => { started = resolve })
  let submissions = 0
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    if (new URL(url).pathname === "/api/v1/reasoning") {
      submissions++
      started()
      await new Promise<void>(resolve => { release = resolve })
      return Response.json({ ok: true, call_id: "receipt" })
    }
    return base(url, init)
  }) as typeof fetch
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    const cancelled = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "cancel", { path: "/repo/a" })))
    yield* Effect.promise(() => reasoningStarted)
    const surviving = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "survive", { path: "/repo/b" })))
    yield* Effect.promise(() => new Promise<void>(resolve => queueMicrotask(resolve)))
    yield* Fiber.interrupt(cancelled)
    release()
    expect(Exit.isSuccess(yield* Fiber.await(surviving))).toBe(true)
    expect(submissions).toBe(1)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
  })))
})

test("a failed shared user submission releases ownership for an ordinary same-ID retry", async () => {
  const fx = fixture()
  fx.setReasoning({ ok: true, call_id: "" })
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    expect(Exit.isFailure(yield* Effect.exit(fx.hooks["execute.before"](fx.call("read", "root", "failed", { path: "/repo/a" }))))).toBe(true)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(0)
    fx.setReasoning({ ok: true, call_id: "receipt-u1" })
    expect(Exit.isSuccess(yield* Effect.exit(fx.hooks["execute.before"](fx.call("read", "root", "retry", { path: "/repo/b" }))))).toBe(true)
    expect(fx.requests.filter(r => r.path === "/api/v1/reasoning")).toHaveLength(2)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
  })))
})

test("removed session cannot receive a late old admission receipt after same-ID recreation", async () => {
  const fx = fixture()
  const queue = await Effect.runPromise(Queue.unbounded<any>())
  ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
  const base = globalThis.fetch
  let releaseOld!: () => void
  let oldStarted!: () => void
  let deleted!: () => void
  const started = new Promise<void>(resolve => { oldStarted = resolve })
  const removed = new Promise<void>(resolve => { deleted = resolve })
  let submissions = 0
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    const path = new URL(url).pathname
    if (path === "/api/v1/reasoning") {
      if (++submissions === 1) {
        oldStarted()
        await new Promise<void>(resolve => { releaseOld = resolve })
      }
      return Response.json({ ok: true, call_id: `receipt-${submissions}` })
    }
    if (path === "/api/v1/session/oc-root/agent-summary") deleted()
    return base(url, init)
  }) as typeof fetch
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    const old = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "old", { path: "/repo/a" })))
    yield* Effect.promise(() => started)
    yield* Queue.offer(queue, { type: "session.deleted", data: { sessionID: "root" } })
    yield* Effect.promise(() => removed)
    yield* Effect.promise(() => new Promise<void>(resolve => queueMicrotask(resolve)))
    const fresh = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "fresh", { path: "/repo/b" })))
    yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
    expect(submissions).toBe(1)
    releaseOld()
    expect(Exit.isFailure(yield* Fiber.await(old))).toBe(true)
    expect(Exit.isSuccess(yield* Fiber.await(fresh))).toBe(true)
    expect(submissions).toBe(2)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
  })))
})

test("replacement admission waits for retired session's already-issued user recording", async () => {
  process.env.INTARIS_SESSION_RECORDING = "true"
  process.env.INTARIS_RECORDING_FLUSH_SIZE = "1"
  const fx = fixture()
  const queue = await Effect.runPromise(Queue.unbounded<any>())
  ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
  const base = globalThis.fetch
  let releaseOld!: () => void
  let oldStarted!: () => void
  let removed!: () => void
  const started = new Promise<void>(resolve => { oldStarted = resolve })
  const deleted = new Promise<void>(resolve => { removed = resolve })
  let userSeq = 0
  let expectedSeq = -1
  let reasoningText = ""
  let refreshed = "old intention"
  const reasoning: string[] = []
  let newUserUploads = 0
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    const path = new URL(url).pathname
    if (path.endsWith("/events")) {
      const events = JSON.parse(String(init.body)) as Array<{ type: string; data: { role?: string; messageID?: string } }>
      for (const event of events) {
        if (event.type !== "message" || event.data.role !== "user") continue
        if (event.data.messageID === "user-2") newUserUploads++
        if (event.data.messageID === "user-1" && !releaseOld) {
          oldStarted()
          await new Promise<void>(resolve => { releaseOld = resolve })
        }
        userSeq++
      }
    }
    if (path === "/api/v1/reasoning") {
      reasoningText = JSON.parse(String(init.body)).content
      reasoning.push(reasoningText)
      expectedSeq = userSeq
    }
    if (path === "/api/v1/evaluate") {
      if (expectedSeq === userSeq) refreshed = reasoningText
      return Response.json({ decision: refreshed === reasoningText ? "approve" : "deny", call_id: "eval", path: "llm", latency_ms: 1 })
    }
    if (path === "/api/v1/session/oc-root/agent-summary") removed()
    return base(url, init)
  }) as typeof fetch
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    const old = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "old", { path: "/repo/a" })))
    yield* Effect.promise(() => started)
    yield* Queue.offer(queue, { type: "session.deleted", data: { sessionID: "root" } })
    yield* Effect.promise(() => deleted)
    yield* Effect.promise(() => new Promise<void>(resolve => queueMicrotask(resolve)))
    fx.setContext([{ id: "user-2", type: "user", text: "New instruction" }])
    const fresh = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "fresh", { path: "/repo/b" })))
    yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
    const beforeOldSettles = { reasoning: reasoning.length, uploads: newUserUploads }
    releaseOld()
    expect(Exit.isFailure(yield* Fiber.await(old))).toBe(true)
    expect(Exit.isSuccess(yield* Fiber.await(fresh))).toBe(true)
    expect(beforeOldSettles).toEqual({ reasoning: 0, uploads: 0 })
    expect(reasoning).toEqual(["User message: New instruction"])
    expect(refreshed).toBe("User message: New instruction")
    expect(expectedSeq).toBe(userSeq)
  })))
})

test("child completion uploads its final buffered recording before retiring the state", async () => {
  process.env.INTARIS_SESSION_RECORDING = "true"
  process.env.INTARIS_RECORDING_FLUSH_SIZE = "50"
  const fx = fixture()
  const queue = await Effect.runPromise(Queue.unbounded<any>())
  ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
  const base = globalThis.fetch
  const uploaded: any[] = []
  let finished!: () => void
  const completion = new Promise<void>(resolve => { finished = resolve })
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    const path = new URL(url).pathname
    if (path === "/api/v1/session/oc-child/events") {
      uploaded.push(...JSON.parse(String(init.body)))
    }
    if (path === "/api/v1/session/oc-child/agent-summary") finished()
    return base(url, init)
  }) as typeof fetch
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    const call = fx.call("read", "child", "last", { path: "/repo/a" })
    yield* fx.hooks["execute.before"](call)
    yield* fx.hooks["execute.after"]({ ...call, status: "completed", result: { output: "done" } })
    yield* Queue.offer(queue, { type: "session.execution.succeeded", data: { sessionID: "child" } })
    yield* Effect.promise(() => completion)
    yield* Effect.promise(() => new Promise<void>(resolve => queueMicrotask(resolve)))
    expect(uploaded.some(event => event.type === "tool_call")).toBe(true)
    expect(uploaded.some(event => event.type === "tool_result")).toBe(true)
  })))
})

test("retired idle-resume cannot record its old user into a replacement's active intention", async () => {
  process.env.INTARIS_SESSION_RECORDING = "true"
  process.env.INTARIS_RECORDING_FLUSH_SIZE = "1"
  const fx = fixture()
  const queue = await Effect.runPromise(Queue.unbounded<any>())
  ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
  const base = globalThis.fetch
  let idleSeen!: () => void
  let resumeSeen!: () => void
  let releaseResume!: () => void
  let deleted!: () => void
  let newerReasoningSeen!: () => void
  let releaseNewer!: () => void
  const idle = new Promise<void>(resolve => { idleSeen = resolve })
  const resuming = new Promise<void>(resolve => { resumeSeen = resolve })
  const removed = new Promise<void>(resolve => { deleted = resolve })
  const newerReasoning = new Promise<void>(resolve => { newerReasoningSeen = resolve })
  let userSeq = 0
  let expectedSeq = -1
  let latestIntention = "old intention"
  let proposed = ""
  let lateOldRecords = 0
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    const path = new URL(url).pathname
    const body = init.body ? JSON.parse(String(init.body)) : null
    if (path.endsWith("/events")) {
      for (const event of body as Array<{ type: string; data: { role?: string; messageID?: string } }>) {
        if (event.type === "message" && event.data.role === "user") {
          if (event.data.messageID === "user-old") lateOldRecords++
          userSeq++
        }
      }
    }
    if (path === "/api/v1/session/oc-root/status" && body.status === "idle") idleSeen()
    if (path === "/api/v1/session/oc-root/status" && body.status === "active") {
      resumeSeen()
      await new Promise<void>(resolve => { releaseResume = resolve })
    }
    if (path === "/api/v1/session/oc-root/agent-summary") deleted()
    if (path === "/api/v1/reasoning") {
      expectedSeq = userSeq
      proposed = body.content
      if (body.content === "User message: New instruction") {
        newerReasoningSeen()
        await new Promise<void>(resolve => { releaseNewer = resolve })
      }
    }
    if (path === "/api/v1/evaluate") {
      if (expectedSeq === userSeq) latestIntention = proposed
      return Response.json({ decision: latestIntention === proposed ? "approve" : "deny", call_id: "eval", path: "llm", latency_ms: 1 })
    }
    return base(url, init)
  }) as typeof fetch
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    yield* fx.hooks["execute.before"](fx.call("read", "root", "baseline", { path: "/repo/a" }))
    yield* Queue.offer(queue, { type: "session.execution.succeeded", data: { sessionID: "root" } })
    yield* Effect.promise(() => idle)
    fx.setContext([{ id: "user-old", type: "user", text: "Superseded instruction" }])
    const old = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "old", { path: "/repo/b" })))
    yield* Effect.promise(() => resuming)
    yield* Queue.offer(queue, { type: "session.deleted", data: { sessionID: "root" } })
    yield* Effect.promise(() => removed)
    yield* Effect.promise(() => new Promise<void>(resolve => queueMicrotask(resolve)))
    fx.setContext([{ id: "user-new", type: "user", text: "New instruction" }])
    const fresh = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "new", { path: "/repo/c" })))
    yield* Effect.promise(() => newerReasoning)
    releaseResume()
    expect(Exit.isFailure(yield* Fiber.await(old))).toBe(true)
    yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
    const oldRecordsAfterNewReasoning = lateOldRecords
    releaseNewer()
    expect(Exit.isSuccess(yield* Fiber.await(fresh))).toBe(true)
    expect(oldRecordsAfterNewReasoning).toBe(0)
    expect(latestIntention).toBe("User message: New instruction")
    expect(expectedSeq).toBe(userSeq)
  })))
})

test("V1 size-triggered recording flush settles before event-derived reasoning", async () => {
  process.env.INTARIS_SESSION_RECORDING = "true"
  process.env.INTARIS_RECORDING_FLUSH_SIZE = "1"
  const fx = fixture()
  const base = globalThis.fetch
  let uploadStarted!: () => void
  let releaseUpload!: () => void
  const started = new Promise<void>(resolve => { uploadStarted = resolve })
  let recordedUser = false
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    const path = new URL(url).pathname
    if (path.endsWith("/events")) {
      const events = JSON.parse(String(init.body)) as Array<{ type: string; data: { role?: string } }>
      if (events.some(event => event.type === "message" && event.data.role === "user")) {
        uploadStarted()
        await new Promise<void>(resolve => { releaseUpload = resolve })
        recordedUser = true
      }
    }
    if (path === "/api/v1/reasoning") {
      const reply = await base(url, init)
      return recordedUser ? reply : Response.json({ ok: true, call_id: "" })
    }
    return base(url, init)
  }) as typeof fetch
  const hooks = await entry.server({ client: { app: { log: async () => ({}) }, tui: { showToast: async () => ({}) } }, worktree: "/repo", directory: "/repo" } as any)
  await hooks.event({ event: { type: "session.created", properties: { info: { id: "root" } } } })
  const message = hooks["chat.message"]({ sessionID: "root", messageID: "user-1" }, { parts: [{ type: "text", text: "Fresh request" }] })
  await started
  const beforeUploadSettles = fx.requests.filter(r => r.path === "/api/v1/reasoning").length
  releaseUpload()
  await message
  const reasoning = fx.requests.filter(r => r.path === "/api/v1/reasoning")
  expect(beforeUploadSettles).toBe(0)
  expect(reasoning).toHaveLength(1)
  expect(reasoning[0].body).toMatchObject({ session_id: "oc-root", content: "", from_events: true })
  expect(recordedUser).toBe(true)
})

test("checkpoint-only continuation clears root idle so subsequent completion emits idle again", async () => {
  const fx = fixture()
  const queue = await Effect.runPromise(Queue.unbounded<any>())
  ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
  const base = globalThis.fetch
  let firstIdle!: () => void
  const idle = new Promise<void>(resolve => { firstIdle = resolve })
  const statuses: string[] = []
  let serverActive = true
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    if (new URL(url).pathname === "/api/v1/session/oc-root/status") {
      const status = JSON.parse(String(init.body)).status
      statuses.push(status)
      serverActive = status === "active"
      if (status === "idle" && statuses.filter(value => value === "idle").length === 1) firstIdle()
    }
    if (new URL(url).pathname === "/api/v1/evaluate") serverActive = true
    return base(url, init)
  }) as typeof fetch
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    yield* fx.hooks["execute.before"](fx.call("read", "root", "initial", { path: "/repo/a" }))
    yield* Queue.offer(queue, { type: "session.execution.succeeded", data: { sessionID: "root" } })
    yield* Effect.promise(() => idle)
    yield* Queue.offer(queue, { type: "session.inbox.enqueued", data: { sessionID: "root", inboxID: "manual-1", item: { type: "compaction", payload: {} } } })
    yield* Queue.offer(queue, { type: "session.inbox.delivered", data: { sessionID: "root", inboxID: "manual-1" } })
    yield* Queue.offer(queue, { type: "session.compaction.started", id: "evt_manual", data: { sessionID: "root", reason: "manual", inputID: "checkpoint-1" } })
    yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
    yield* fx.sessionHooks.compaction({ sessionID: "root" })
    yield* Queue.offer(queue, { type: "session.compaction.ended", id: "evt_end", data: { sessionID: "root", reason: "manual", text: "summary" } })
    yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
    fx.setContext([{ id: "checkpoint-1", type: "compaction", status: "completed", summary: "not an instruction" }])
    yield* fx.hooks["execute.before"](fx.call("read", "root", "continued", { path: "/repo/b" }))
    expect(serverActive).toBe(true)
    yield* Queue.offer(queue, { type: "session.execution.succeeded", data: { sessionID: "root" } })
    yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(2)
    expect(statuses.filter(status => status === "idle")).toHaveLength(2)
  })))
})

test("move during deferred compaction context cannot re-install revoked user proof", async () => {
  const fx = fixture()
  const queue = await Effect.runPromise(Queue.unbounded<any>())
  ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    yield* fx.hooks["execute.before"](fx.call("read", "root", "initial", { path: "/repo/a" }))
    const context = fx.ctx.session.context
    let lookupStarted!: () => void
    let releaseLookup!: () => void
    const started = new Promise<void>(resolve => { lookupStarted = resolve })
    let first = true
    fx.ctx.session.context = () => {
      if (!first) return context()
      first = false
      return Effect.promise(() => new Promise<any[]>(resolve => {
        releaseLookup = () => resolve([{ id: "user-1", type: "user", text: "Run the requested work" }])
        lookupStarted()
      }))
    }
    const hook = yield* Effect.forkChild(fx.sessionHooks.compaction({ sessionID: "root" }))
    yield* Effect.promise(() => started)
    yield* Queue.offer(queue, { type: "session.moved", data: { sessionID: "root", location: { directory: "/repo" }, projectID: "project" } })
    yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
    releaseLookup()
    yield* Fiber.join(hook)
    yield* Queue.offer(queue, { type: "session.compaction.started", id: "evt_1", data: { sessionID: "root", reason: "auto" } })
    yield* Queue.offer(queue, { type: "session.compaction.ended", id: "evt_end", data: { sessionID: "root", reason: "auto", text: "summary" } })
    yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
    fx.setContext([{ id: "msg_1", type: "compaction", status: "completed", summary: "not an instruction" }])
    expect(Exit.isFailure(yield* Effect.exit(fx.hooks["execute.before"](fx.call("read", "root", "blocked", { path: "/repo/b" }))))).toBe(true)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
    fx.setContext([{ id: "user-2", type: "user", text: "Fresh instruction" }])
    yield* fx.hooks["execute.before"](fx.call("read", "root", "fresh", { path: "/repo/c" }))
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(2)
  })))
})

test("stale compaction lookup success or failure cannot erase a newly completed proof", async () => {
  for (const outcome of ["success", "failure"] as const) {
    const fx = fixture()
    const queue = await Effect.runPromise(Queue.unbounded<any>())
    ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      yield* entry.effect(fx.ctx as any)
      yield* fx.hooks["execute.before"](fx.call("read", "root", "initial", { path: "/repo/a" }))
      const context = fx.ctx.session.context
      let lookupStarted!: () => void
      let finishLookup!: () => void
      const started = new Promise<void>(resolve => { lookupStarted = resolve })
      let first = true
      fx.ctx.session.context = () => {
        if (!first) return context()
        first = false
        return Effect.promise(() => new Promise<any[]>((resolve, reject) => {
          finishLookup = () => outcome === "success"
            ? resolve([{ id: "user-1", type: "user", text: "Run the requested work" }])
            : reject(new Error("stale context failed"))
          lookupStarted()
        }))
      }
      const stale = yield* Effect.forkChild(fx.sessionHooks.compaction({ sessionID: "root" }))
      yield* Effect.promise(() => started)
      yield* Queue.offer(queue, { type: "session.moved", data: { sessionID: "root", location: { directory: "/repo" }, projectID: "project" } })
      yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
      fx.setContext([{ id: "user-2", type: "user", text: "Fresh instruction" }])
      yield* fx.hooks["execute.before"](fx.call("read", "root", "fresh-user", { path: "/repo/fresh" }))
      yield* fx.sessionHooks.compaction({ sessionID: "root" })
      yield* Queue.offer(queue, { type: "session.compaction.started", id: "evt_fresh", data: { sessionID: "root", reason: "auto" } })
      yield* Queue.offer(queue, { type: "session.compaction.ended", id: "evt_end", data: { sessionID: "root", reason: "auto", text: "summary" } })
      yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
      finishLookup()
      yield* Fiber.join(stale)
      fx.setContext([{ id: "msg_fresh", type: "compaction", status: "completed", summary: "not an instruction" }])
      yield* fx.hooks["execute.before"](fx.call("read", "root", "continued", { path: "/repo/b" }))
      expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(3)
    })))
  }
})

test("old lookup from a removed session ID stays stale after a replacement proves its user", async () => {
  const fx = fixture()
  const queue = await Effect.runPromise(Queue.unbounded<any>())
  ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
  const base = globalThis.fetch
  let removed!: () => void
  const deleted = new Promise<void>(resolve => { removed = resolve })
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    if (new URL(url).pathname === "/api/v1/session/oc-root/agent-summary") removed()
    return base(url, init)
  }) as typeof fetch
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    yield* fx.hooks["execute.before"](fx.call("read", "root", "initial", { path: "/repo/a" }))
    const context = fx.ctx.session.context
    let lookupStarted!: () => void
    let finishLookup!: () => void
    const started = new Promise<void>(resolve => { lookupStarted = resolve })
    let first = true
    fx.ctx.session.context = () => {
      if (!first) return context()
      first = false
      return Effect.promise(() => new Promise<any[]>(resolve => {
        finishLookup = () => resolve([{ id: "user-1", type: "user", text: "old" }])
        lookupStarted()
      }))
    }
    const stale = yield* Effect.forkChild(fx.sessionHooks.compaction({ sessionID: "root" }))
    yield* Effect.promise(() => started)
    yield* Queue.offer(queue, { type: "session.deleted", data: { sessionID: "root" } })
    yield* Effect.promise(() => deleted)
    yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
    fx.setContext([{ id: "user-2", type: "user", text: "Fresh instruction" }])
    yield* fx.hooks["execute.before"](fx.call("read", "root", "new", { path: "/repo/b" }))
    yield* fx.sessionHooks.compaction({ sessionID: "root" })
    yield* Queue.offer(queue, { type: "session.compaction.started", id: "evt_new", data: { sessionID: "root", reason: "auto" } })
    yield* Queue.offer(queue, { type: "session.compaction.ended", id: "evt_end", data: { sessionID: "root", reason: "auto", text: "summary" } })
    yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
    finishLookup()
    yield* Fiber.join(stale)
    fx.setContext([{ id: "msg_new", type: "compaction", status: "completed", summary: "not an instruction" }])
    yield* fx.hooks["execute.before"](fx.call("read", "root", "continued", { path: "/repo/c" }))
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(3)
  })))
})

test("finishing one current context lookup does not revoke another independent tool lookup", async () => {
  const fx = fixture()
  let started!: () => void
  const bothStarted = new Promise<void>(resolve => { started = resolve })
  const releases: Array<() => void> = []
  fx.ctx.session.context = () => Effect.promise(() => new Promise<any[]>(resolve => {
    releases.push(() => resolve([{ id: "user-1", type: "user", text: "Run the requested work" }]))
    if (releases.length === 2) started()
  }))
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    const first = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "one", { path: "/repo/a" })))
    const second = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "two", { path: "/repo/b" })))
    yield* Effect.promise(() => bothStarted)
    releases[0]()
    expect(Exit.isSuccess(yield* Fiber.await(first))).toBe(true)
    releases[1]()
    expect(Exit.isSuccess(yield* Fiber.await(second))).toBe(true)
    expect(fx.requests.filter(r => r.path === "/api/v1/reasoning")).toHaveLength(1)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(2)
  })))
})

test("tool lookup begun before a move cannot evaluate its stale typed user context", async () => {
  const fx = fixture()
  const queue = await Effect.runPromise(Queue.unbounded<any>())
  ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
  let started!: () => void
  let release!: () => void
  const waiting = new Promise<void>(resolve => { started = resolve })
  const context = fx.ctx.session.context
  let first = true
  fx.ctx.session.context = () => {
    if (!first) return context()
    first = false
    return Effect.promise(() => new Promise<any[]>(resolve => {
      release = () => resolve([{ id: "user-1", type: "user", text: "stale" }])
      started()
    }))
  }
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    const stale = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "stale", { path: "/repo/a" })))
    yield* Effect.promise(() => waiting)
    yield* Queue.offer(queue, { type: "session.moved", data: { sessionID: "root", location: { directory: "/repo" }, projectID: "project" } })
    yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
    release()
    expect(Exit.isFailure(yield* Fiber.await(stale))).toBe(true)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(0)
    yield* fx.hooks["execute.before"](fx.call("read", "root", "fresh", { path: "/repo/b" }))
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
  })))
})

for (const change of ["delivered user", "committed revert"] as const) {
  test(`${change} revokes an in-flight old reasoning admission`, async () => {
    const fx = fixture()
    const queue = await Effect.runPromise(Queue.unbounded<any>())
    ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
    const base = globalThis.fetch
    let reasoningStarted!: () => void
    let releaseOld!: () => void
    const started = new Promise<void>(resolve => { reasoningStarted = resolve })
    let first = true
    globalThis.fetch = (async (url: string, init: RequestInit) => {
      if (new URL(url).pathname === "/api/v1/reasoning" && first) {
        first = false
        const response = await base(url, init)
        reasoningStarted()
        await new Promise<void>(resolve => { releaseOld = resolve })
        return response
      }
      return base(url, init)
    }) as typeof fetch
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      yield* entry.effect(fx.ctx as any)
      const old = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "old", { path: "/repo/a" })))
      yield* Effect.promise(() => started)
      if (change === "delivered user") {
        yield* Queue.offer(queue, { type: "session.inbox.enqueued", data: { sessionID: "root", inboxID: "user-2", item: { type: "user", payload: { text: "Fresh instruction" } } } })
        yield* Queue.offer(queue, { type: "session.inbox.delivered", data: { sessionID: "root", inboxID: "user-2" } })
      } else {
        yield* Queue.offer(queue, { type: "session.revert.committed", data: { sessionID: "root", to: "msg_before_user" } })
      }
      yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
      releaseOld()
      expect(Exit.isFailure(yield* Fiber.await(old))).toBe(true)
      expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(0)

      yield* fx.sessionHooks.compaction({ sessionID: "root" })
      yield* Queue.offer(queue, { type: "session.compaction.started", id: "evt_1", data: { sessionID: "root", reason: "auto" } })
      yield* Queue.offer(queue, { type: "session.compaction.ended", id: "evt_end", data: { sessionID: "root", reason: "auto", text: "summary" } })
      yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
      fx.setContext([{ id: "msg_1", type: "compaction", status: "completed", summary: "not a user" }])
      expect(Exit.isFailure(yield* Effect.exit(fx.hooks["execute.before"](fx.call("read", "root", "summary", { path: "/repo/b" }))))).toBe(true)
      expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(0)

      if (change === "committed revert") {
        fx.setContext([{ id: "user-1", type: "user", text: "Run the requested work" }])
        yield* fx.hooks["execute.before"](fx.call("read", "root", "same-id-retry", { path: "/repo/retry" }))
        expect(fx.requests.filter(r => r.path === "/api/v1/reasoning")).toHaveLength(2)
      }
      fx.setContext([{ id: "user-2", type: "user", text: "Fresh instruction" }])
      yield* fx.hooks["execute.before"](fx.call("read", "root", "fresh", { path: "/repo/c" }))
      expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(change === "committed revert" ? 2 : 1)
    })))
  })
}

test("reverting a cached V2 receipt requires another reasoning submission for the same native ID", async () => {
  const fx = fixture()
  const queue = await Effect.runPromise(Queue.unbounded<any>())
  ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    yield* fx.hooks["execute.before"](fx.call("read", "root", "before", { path: "/repo/a" }))
    yield* Queue.offer(queue, { type: "session.revert.committed", data: { sessionID: "root", to: "msg_before_user" } })
    yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
    yield* fx.hooks["execute.before"](fx.call("read", "root", "after", { path: "/repo/b" }))
    expect(fx.requests.filter(r => r.path === "/api/v1/reasoning")).toHaveLength(2)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(2)
  })))
})

test("revert during pending approval cannot grant CodeMode outer execution", async () => {
  const fx = fixture()
  const queue = await Effect.runPromise(Queue.unbounded<any>())
  ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
  const base = globalThis.fetch
  let evaluating!: () => void
  let release!: () => void
  const started = new Promise<void>(resolve => { evaluating = resolve })
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    if (new URL(url).pathname === "/api/v1/evaluate") {
      evaluating()
      await new Promise<void>(resolve => { release = resolve })
    }
    return base(url, init)
  }) as typeof fetch
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    const outer = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call()))
    yield* Effect.promise(() => started)
    yield* Queue.offer(queue, { type: "session.revert.committed", data: { sessionID: "root", to: "msg_before_user" } })
    yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
    release()
    expect(Exit.isFailure(yield* Fiber.await(outer))).toBe(true)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
  })))
})

test("revoked CodeMode family cannot authorize nested calls after its outer approval", async () => {
  const fx = fixture()
  const queue = await Effect.runPromise(Queue.unbounded<any>())
  ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
  let approved!: () => void
  let finish!: () => void
  const ready = new Promise<void>(resolve => { approved = resolve })
  const active = new Promise<void>(resolve => { finish = resolve })
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    const outer = fx.call()
    const owner = yield* Effect.forkChild(Effect.gen(function* () {
      yield* fx.hooks["execute.before"](outer)
      approved()
      yield* Effect.promise(() => active)
    }))
    yield* Effect.promise(() => ready)
    yield* Queue.offer(queue, { type: "session.revert.committed", data: { sessionID: "root", to: "msg_before_user" } })
    yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
    const nested = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("shell", "root", "call-1", { command: "pwd" })))
    expect(Exit.isFailure(yield* Fiber.await(nested))).toBe(true)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
    finish()
    yield* Fiber.join(owner)
  })))
})

test("suspension reactivation after revert cannot issue a second evaluation", async () => {
  const fx = fixture({ decision: "deny", session_status: "suspended", call_id: "suspended", path: "fast", latency_ms: 1 })
  const queue = await Effect.runPromise(Queue.unbounded<any>())
  ;(fx.ctx.event as any).subscribe = () => Stream.fromQueue(queue)
  const base = globalThis.fetch
  let statusStarted!: () => void
  let releaseStatus!: () => void
  const started = new Promise<void>(resolve => { statusStarted = resolve })
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    if (new URL(url).pathname === "/api/v1/session/oc-root" && init.method === "GET") {
      statusStarted()
      await new Promise<void>(resolve => { releaseStatus = resolve })
      return Response.json({ status: "active" })
    }
    return base(url, init)
  }) as typeof fetch
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* entry.effect(fx.ctx as any)
    const call = yield* Effect.forkChild(fx.hooks["execute.before"](fx.call("read", "root", "suspended", { path: "/repo/a" })))
    yield* Effect.promise(() => started)
    yield* Queue.offer(queue, { type: "session.revert.committed", data: { sessionID: "root", to: "msg_before_user" } })
    yield* Effect.promise(() => new Promise<void>(resolve => setImmediate(resolve)))
    releaseStatus()
    expect(Exit.isFailure(yield* Fiber.await(call))).toBe(true)
    expect(fx.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
  })))
})
