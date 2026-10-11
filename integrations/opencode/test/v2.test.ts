import { afterEach, expect, test } from "bun:test"
import { Effect, Exit, Stream, Cause, Option, Queue, Fiber } from "effect"
import { Tool } from "@opencode/schema/tool"
import type { Plugin } from "@opencode/plugin/effect"
import type { OpenCodeEvent } from "@opencode/protocol/groups/event"
import { adaptV2, evaluationInput, handleV2Event } from "../intaris-v2"
import { createCore } from "../src/core"

const initialFetch = globalThis.fetch
afterEach(() => { globalThis.fetch = initialFetch })

function mock(decision: unknown = { decision: "approve", call_id: "audit-1" }, session: Record<string, any> = {}, stream = Stream.never) {
  const requests: Array<{ path: string; payload: any }> = []
  globalThis.fetch = (async (url: string, request: RequestInit) => {
    const path = new URL(url).pathname
    requests.push({ path, payload: request.body ? JSON.parse(String(request.body)) : null })
    return new Response(JSON.stringify(path === "/api/v1/intention" ? { session_id: "oc-s1" } : path === "/api/v1/evaluate" ? decision : {}))
  }) as typeof fetch
  const hooks: Record<string, (event: any) => any> = {}
  const ctx = {
    tool: { hook: (name: string, callback: (event: any) => any) => Effect.sync(() => { hooks[name] = callback; return { dispose: Effect.void } }) },
    session: {
      get: () => Effect.succeed({ id: "s1", title: "Fix tests", location: { directory: "/project" }, ...session }),
      context: () => Effect.succeed(session.context ?? [{ id: "u0", type: "user", text: "admitted intention" }]),
    },
    event: { subscribe: () => stream },
  } as unknown as Plugin.Context
  return { requests, ctx, hooks }
}

const event = (tool = "shell", input: unknown = { command: "pwd" }, sessionID = "s1") => ({
  tool, input, sessionID, agent: "build", messageID: "message-1", id: "call-1",
})

test("shell name maps to bash with command and raw identity; MCP and custom names are unchanged", () => {
  expect(evaluationInput("shell", { command: "pwd" })).toEqual({ tool: "bash", args: { command: "pwd", _intaris_opencode_tool: "shell" } })
  expect(evaluationInput("server_tool", { input: "x" }).tool).toBe("server_tool")
  expect(evaluationInput("custom", { input: "x" }).tool).toBe("custom")
  expect(() => evaluationInput("shell", { code: "pwd" })).toThrow()
  expect(() => evaluationInput("execute", {})).toThrow("CodeMode")
})

test("V2 hook evaluates once before allowing and denies through Tool.Error failure channel", async () => {
  const fixture = mock()
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* adaptV2(fixture.ctx)
    yield* fixture.hooks["execute.before"](event())
    expect(fixture.requests.filter(item => item.path === "/api/v1/evaluate")).toHaveLength(1)
    expect(fixture.requests.find(item => item.path === "/api/v1/evaluate")?.payload.tool).toBe("bash")
  })))

  for (const result of [{ decision: "deny" }, { decision: "unknown" }, {}]) {
    const denied = mock(result)
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      yield* adaptV2(denied.ctx)
      const exit = yield* Effect.exit(denied.hooks["execute.before"](event()))
      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isFailure(exit)) {
        const failure = Option.getOrThrow(Cause.findErrorOption(exit.cause))
        expect(failure).toBeInstanceOf(Tool.Error)
        expect((failure as Tool.Error)._tag).toBe("Tool.Error")
      }
      expect(denied.requests.filter(item => item.path === "/api/v1/evaluate")).toHaveLength(1)
    })))
  }
})

test("execute is blocked before evaluator and missing session metadata fails closed", async () => {
  const fixture = mock()
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* adaptV2(fixture.ctx)
    const exit = yield* Effect.exit(fixture.hooks["execute.before"](event("execute", {})))
    expect(Exit.isFailure(exit)).toBe(true)
    expect(fixture.requests.filter(item => item.path === "/api/v1/evaluate")).toHaveLength(0)
  })))
  const missing = mock()
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* adaptV2(missing.ctx)
    const exit = yield* Effect.exit(missing.hooks["execute.before"](event("read", {}, "")))
    expect(Exit.isFailure(exit)).toBe(true)
    expect(missing.requests.filter(item => item.path === "/api/v1/evaluate")).toHaveLength(0)
  })))
})

test("queued or cancelled user text is not forwarded; delivered text is user-role context", async () => {
  const fixture = mock()
  const core = await createCore({ worktree: "/project" })
  const pending = new Map<string, any>()
  const announced = new Set<string>()
  const invalid = new Set<string>()
  const metadata = async () => ({ workingDirectory: "/project" })
  const send = (type: string, data: object) => handleV2Event(core, { type, data } as OpenCodeEvent, pending, announced, metadata, invalid)
  await send("session.inbox.enqueued", { sessionID: "s1", inboxID: "queued", item: { type: "user", payload: { text: "do not publish" } } })
  await send("session.inbox.cancelled", { inboxID: "queued", sessionID: "s1" })
  expect(fixture.requests.filter(r => r.path === "/api/v1/reasoning")).toHaveLength(0)
  await send("session.inbox.enqueued", { sessionID: "s1", inboxID: "admitted", item: { type: "user", payload: { text: "admitted intention" } } })
  await send("session.inbox.delivered", { sessionID: "s1", inboxID: "admitted" })
  await core["tool.execute.before"]({ tool: "read", sessionID: "s1", callID: "c1" }, { args: { path: "/project/a" } })
  expect(fixture.requests.find(r => r.path === "/api/v1/reasoning")?.payload.content).toBe("User message: admitted intention")
  core.dispose()
})

test("non-user inbox deliveries do not poison the guard; unmatched delivery blocks", async () => {
  const fixture = mock()
  const core = await createCore({ worktree: "/project" })
  const pending = new Map<string, any>()
  const announced = new Set<string>()
  const invalid = new Set<string>()
  const metadata = async () => ({ workingDirectory: "/project" })
  const send = (type: string, data: object) => handleV2Event(core, { type, data } as OpenCodeEvent, pending, announced, metadata, invalid)
  for (const kind of ["synthetic", "compaction", "move"]) {
    await send("session.inbox.enqueued", { inboxID: kind, sessionID: "s1", item: { type: kind, payload: {} } })
    await send("session.inbox.delivered", { inboxID: kind, sessionID: "s1" })
  }
  expect(invalid.size).toBe(0)
  expect(fixture.requests.filter(r => r.path === "/api/v1/reasoning")).toHaveLength(0)
  await send("session.inbox.delivered", { inboxID: "missing-user", sessionID: "s1" })
  expect(invalid.has("s1")).toBe(true)
  core.dispose()
})

test("resumed child delivered before tool retains authoritative parent and agent; fork-only stays root", async () => {
  for (const extra of [{ parentID: "parent", agent: "explorer" }, { fork: { sessionID: "source" } }]) {
    const fixture = mock(undefined, extra)
    const core = await createCore({ resolveSessionMetadata: async () => ({ workingDirectory: "/project", ...extra, title: "Fix tests" }) })
    const pending = new Map<string, any>()
    const announced = new Set<string>()
    const invalid = new Set<string>()
    const metadata = async () => ({ workingDirectory: "/project", ...extra, title: "Fix tests" })
    await handleV2Event(core, { type: "session.inbox.enqueued", data: { sessionID: "s1", inboxID: "u0", item: { type: "user", payload: { text: "admitted intention" } } } } as OpenCodeEvent, pending, announced, metadata, invalid)
    await handleV2Event(core, { type: "session.inbox.delivered", data: { sessionID: "s1", inboxID: "u0" } } as OpenCodeEvent, pending, announced, metadata, invalid)
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      yield* adaptV2(fixture.ctx)
      yield* fixture.hooks["execute.before"](event("read", { path: "/project/a" }))
    })))
    const created = fixture.requests.filter(r => r.path === "/api/v1/intention").map(r => r.payload)
    expect(created.length).toBeGreaterThan(0)
    for (const body of created) {
      expect(body.details.agent_type).toBe("agent" in extra ? extra.agent : undefined)
      expect(body.parent_session_id).toBe("parentID" in extra ? "oc-parent" : undefined)
      expect(body.details.working_directory).toBe("/project")
    }
    core.dispose()
  }
})

test("delivered user context must finish idle-resume before evaluation", async () => {
  const queue = await Effect.runPromise(Queue.unbounded<OpenCodeEvent>())
  const fixture = mock(undefined, { context: [{ id: "admitted", type: "user", text: "new intent" }] }, Stream.fromQueue(queue))
  let release!: () => void
  let resume!: () => void
  const resumed = new Promise<void>(resolve => { resume = resolve })
  const base = globalThis.fetch
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    if (new URL(url).pathname === "/api/v1/session/oc-s1/status" && JSON.parse(String(init.body)).status === "active") {
      resume()
      await new Promise<void>(resolve => { release = resolve })
    }
    return base(url, init)
  }) as typeof fetch
  const send = (type: string, data: object) => Effect.runPromise(Queue.offer(queue, { type, data, durable: { aggregateID: "s1" } } as OpenCodeEvent))
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* adaptV2(fixture.ctx)
    yield* Effect.promise(() => send("session.created", { title: "Initial" }))
    yield* Effect.promise(async () => {
      for (let i = 0; i < 100 && !fixture.requests.some(r => r.path === "/api/v1/intention"); i++) await new Promise(resolve => setTimeout(resolve, 2))
    })
    yield* Effect.promise(() => send("session.execution.succeeded", { sessionID: "s1" }))
    // The parent has gone idle before a new admitted input arrives.
    yield* Effect.promise(async () => {
      for (let i = 0; i < 100 && !fixture.requests.some(r => r.path === "/api/v1/session/oc-s1/status" && r.payload.status === "idle"); i++) await new Promise(resolve => setTimeout(resolve, 2))
    })
    expect(fixture.requests.some(r => r.path === "/api/v1/session/oc-s1/status" && r.payload.status === "idle")).toBe(true)
    yield* Effect.promise(() => send("session.inbox.enqueued", { inboxID: "admitted", sessionID: "s1", item: { type: "user", payload: { text: "new intent" } } }))
    yield* Effect.promise(() => send("session.inbox.delivered", { inboxID: "admitted", sessionID: "s1" }))
    yield* Effect.promise(() => resumed)
    const before = yield* Effect.forkChild(fixture.hooks["execute.before"](event("read", { path: "/project/a" })))
    yield* Effect.promise(() => new Promise<void>(resolve => setTimeout(resolve, 20)))
    expect(fixture.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(0)
    release()
    yield* Fiber.join(before)
    const paths = fixture.requests.map(r => r.path)
    expect(paths.indexOf("/api/v1/reasoning")).toBeGreaterThan(-1)
    expect(paths.indexOf("/api/v1/reasoning")).toBeLessThan(paths.indexOf("/api/v1/evaluate"))
    expect(fixture.requests.find(r => r.path === "/api/v1/reasoning")?.payload.content).toBe("User message: new intent")
  })))
})

test("startup without durable admitted user and unmatched delivery deny without evaluating", async () => {
  const fixture = mock(undefined, { context: [] })
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* adaptV2(fixture.ctx)
    const exit = yield* Effect.exit(fixture.hooks["execute.before"](event("read", { path: "x" })))
    expect(Exit.isFailure(exit)).toBe(true)
    if (Exit.isFailure(exit)) expect(Option.getOrThrow(Cause.findErrorOption(exit.cause))).toBeInstanceOf(Tool.Error)
    expect(fixture.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(0)
  })))
})

test("V2 fiber interruption aborts evaluator fetch; subsequent independent call succeeds", async () => {
  const fixture = mock()
  const base = globalThis.fetch
  let started!: () => void
  const inFlight = new Promise<void>(resolve => { started = resolve })
  let abortObserved = false
  let first = true
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    if (new URL(url).pathname === "/api/v1/evaluate" && first) {
      first = false
      return new Promise<Response>((_resolve, reject) => {
        init.signal?.addEventListener("abort", () => { abortObserved = true; reject(new Error("cancelled")) }, { once: true })
        started()
      })
    }
    return base(url, init)
  }) as typeof fetch
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* adaptV2(fixture.ctx)
    const fiber = yield* Effect.forkChild(fixture.hooks["execute.before"](event()))
    yield* Effect.promise(() => inFlight)
    yield* Fiber.interrupt(fiber)
    expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
    expect(abortObserved).toBe(true)
    yield* fixture.hooks["execute.before"]({ ...event(), id: "call-2" })
    expect(fixture.requests.filter(r => r.path === "/api/v1/evaluate")).toHaveLength(1)
  })))
})

test("interrupted escalation cannot release after late approval; next call proceeds", async () => {
  const fixture = mock()
  const base = globalThis.fetch
  let evaluations = 0
  let evaluated!: () => void
  const firstEvaluated = new Promise<void>(resolve => { evaluated = resolve })
  let lateApproval = false
  let auditPolls = 0
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    const path = new URL(url).pathname
    if (path === "/api/v1/evaluate") {
      evaluations++
      if (evaluations === 1) { evaluated(); return Response.json({ decision: "escalate", call_id: "audit-1" }) }
    }
    if (path.startsWith("/api/v1/audit/")) { auditPolls++; return Response.json({ user_decision: lateApproval ? "approve" : null }) }
    return base(url, init)
  }) as typeof fetch
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    yield* adaptV2(fixture.ctx)
    const fiber = yield* Effect.forkChild(fixture.hooks["execute.before"](event()))
    yield* Effect.promise(() => firstEvaluated)
    yield* Fiber.interrupt(fiber)
    expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
    lateApproval = true
    yield* fixture.hooks["execute.before"]({ ...event(), id: "call-2" })
    expect(evaluations).toBe(2)
    expect(auditPolls).toBe(0)
  })))
})
