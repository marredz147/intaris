/** OpenCode 2.0.26 Effect plugin entry; install separately from the V1 autoload entry. */
import { Plugin } from "@opencode/plugin/effect"
import { Tool } from "@opencode/schema/tool"
import type { Session } from "@opencode/schema/session"
import type { SessionInbox } from "@opencode/schema/session-inbox"
import { Effect, Stream } from "effect"
import type { OpenCodeEvent } from "@opencode/protocol/groups/event"
import { createCore } from "./src/core"

type Core = Awaited<ReturnType<typeof createCore>>
type Event = OpenCodeEvent
type Data<Type extends Event["type"]> = Extract<Event, { type: Type }>["data"]
type InboxItem = { sessionID: string; item: SessionInbox.Item }

// The V2 native shell tool has a command input; Intaris's critical-pattern
// classifier recognizes that shape under the V1-compatible name "bash".
export function evaluationInput(tool: string, input: unknown): { tool: string; args: Record<string, unknown> } {
  if (tool === "execute") throw new Error("CodeMode execute is unsupported: nested calls cannot be individually authorized")
  if (tool === "shell") {
    if (!input || typeof input !== "object" || Array.isArray(input) || typeof (input as Record<string, unknown>).command !== "string") {
      throw new Error("Unrecognized shell command input")
    }
    return { tool: "bash", args: { ...(input as Record<string, unknown>), _intaris_opencode_tool: "shell" } }
  }
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    // Unknown input shapes cannot be silently replaced with empty args.
    throw new Error("Unrecognized tool input")
  }
  return { tool, args: input as Record<string, unknown> }
}

export function adaptV2(ctx: Plugin.Context) {
  const metadata = (id: string) => Effect.runPromise(ctx.session.get({ sessionID: id as Session.ID })).then(info => ({
    workingDirectory: info.location.directory,
    parentID: info.parentID,
    agent: info.agent,
    title: info.title,
  }))
  return Effect.gen(function* () {
    const core = yield* Effect.promise(() => createCore({ resolveSessionMetadata: metadata }))
    yield* Effect.addFinalizer(() => Effect.sync(core.dispose))
    const pending = new Map<string, InboxItem>()
    const announced = new Set<string>()
    const invalid = new Set<string>()
    const ready = new Map<string, Promise<void>>()
    const seenUser = new Map<string, string>()
    yield* ctx.tool.hook("execute.before", event => Effect.gen(function* () {
      if (invalid.has(event.sessionID)) return yield* Effect.fail(new Tool.Error({ message: "[intaris] User-message context unavailable" }))
      const waiting = ready.get(event.sessionID)
      if (waiting) yield* Effect.tryPromise({ try: () => waiting, catch: () => new Tool.Error({ message: "[intaris] User-message context unavailable" }) })
      if (!announced.has(event.sessionID)) {
        const info = yield* ctx.session.get({ sessionID: event.sessionID }).pipe(
          Effect.mapError(() => new Tool.Error({ message: "[intaris] Cannot retrieve session metadata" })),
        )
        if (!info.location.directory) return yield* Effect.fail(new Tool.Error({ message: "[intaris] Missing session directory" }))
        yield* Effect.tryPromise({
          try: () => core.event({ event: { type: "session.created", properties: { info: {
            id: info.id, title: info.title, parentID: info.parentID,
          } } } }),
          catch: () => new Tool.Error({ message: "[intaris] Cannot initialize session metadata" }),
        })
        announced.add(event.sessionID)
      }
      // A new plugin can miss earlier event deliveries. The durable session
      // context is authoritative for admitted user messages (not queued input).
      const context = yield* ctx.session.context({ sessionID: event.sessionID }).pipe(
        Effect.mapError(() => new Tool.Error({ message: "[intaris] Cannot retrieve admitted user context" })),
      )
      const user = context.filter(message => message.type === "user").at(-1)
      if (!user || user.type !== "user") return yield* Effect.fail(new Tool.Error({ message: "[intaris] No admitted user context available" }))
      if (seenUser.get(event.sessionID) !== user.id) {
        yield* Effect.tryPromise({
          try: async () => {
            await core.event({ event: { type: "message.updated", properties: { info: { id: user.id, role: "user", sessionID: event.sessionID } } } })
            await core.event({ event: { type: "message.part.updated", properties: { part: { messageID: user.id, sessionID: event.sessionID, type: "text", text: user.text } } } })
          },
          catch: () => new Tool.Error({ message: "[intaris] Cannot register admitted user context" }),
        })
        seenUser.set(event.sessionID, user.id)
      }
      const current = ready.get(event.sessionID)
      if (current && current !== waiting) yield* Effect.tryPromise({ try: () => current, catch: () => new Tool.Error({ message: "[intaris] User-message context unavailable" }) })
      if (invalid.has(event.sessionID)) return yield* Effect.fail(new Tool.Error({ message: "[intaris] User-message context unavailable" }))
      const mapped = yield* Effect.try({ try: () => evaluationInput(event.tool, event.input), catch: () => new Tool.Error({ message: "[intaris] Unsupported tool input" }) })
      // Effect.tryPromise converts errors to the typed host failure. Interruption
      // remains an Effect interruption, not an approval or a swallowed exception.
      yield* Effect.tryPromise({
        try: async (signal) => {
          if (signal.aborted) throw new Error("[intaris] Evaluation interrupted")
          await core["tool.execute.before"](
            { tool: mapped.tool, sessionID: event.sessionID, callID: event.id, signal },
            { args: mapped.args },
          )
          if (signal.aborted) throw new Error("[intaris] Evaluation interrupted")
        },
        catch: (error) => new Tool.Error({ message: error instanceof Error ? error.message : "[intaris] Evaluation failed" }),
      })
    }))
    yield* ctx.tool.hook("execute.after", event => Effect.tryPromise({
      try: () => core["tool.execute.after"](
        { tool: event.tool, sessionID: event.sessionID, callID: event.id },
        event.status === "completed"
          ? { output: event.result, isError: false }
          : { output: undefined, isError: true, metadata: { error: event.error.message } },
      ),
      catch: () => undefined,
    }).pipe(Effect.ignore))
    const handle = (event: Event): Promise<void> => handleV2Event(core, event, pending, announced, metadata, invalid, seenUser)
    yield* Stream.runForEach(ctx.event.subscribe(), event => Effect.tryPromise(() => {
      if (event.type !== "session.inbox.delivered") return handle(event)
      const id = (event.data as Data<"session.inbox.delivered">).sessionID
      const prior = ready.get(id) || Promise.resolve()
      const next = prior.then(() => handle(event))
      ready.set(id, next)
      void next.finally(() => { if (ready.get(id) === next) ready.delete(id) }).catch(() => {})
      return next
    }).pipe(
      Effect.catch(() => Effect.sync(() => {
        const id = "sessionID" in event.data ? event.data.sessionID : "durable" in event ? event.durable.aggregateID : undefined
        if (typeof id === "string") invalid.add(id)
      })),
    )).pipe(Effect.forkScoped)
    yield* Effect.logInfo("Intaris V2 hooks registered (inventory: /plugins or opencode plugin list; health is separate)")
  })
}

export async function handleV2Event(
  core: Core,
  event: Event,
  pending: Map<string, InboxItem>,
  announced: Set<string>,
  metadata: (id: string) => Promise<{ workingDirectory: string; parentID?: string; agent?: string; title?: string }>,
  invalid: Set<string>,
  seenUser: Map<string, string> = new Map(),
): Promise<void> {
  if (event.type === "session.created") {
    const data = event.data as Data<"session.created">
    const id = event.durable.aggregateID
    announced.add(id)
    await core.event({ event: { type: "session.created", properties: { info: {
      id, title: data.title, parentID: data.parentID,
    } } } })
    return
  }
  if (event.type === "session.renamed") {
    const data = event.data as Data<"session.renamed">
    await core.event({ event: { type: "session.updated", properties: { info: { id: data.sessionID, title: data.title } } } })
    return
  }
  if (event.type === "session.deleted") {
    const data = event.data as Data<"session.deleted">
    announced.delete(data.sessionID)
    for (const [id, item] of pending) if (item.sessionID === data.sessionID) pending.delete(id)
    await core.event({ event: { type: "session.deleted", properties: { info: { id: data.sessionID } } } })
    return
  }
  if (event.type === "session.inbox.enqueued") {
    const data = event.data as Data<"session.inbox.enqueued">
    pending.set(data.inboxID, { sessionID: data.sessionID, item: data.item })
    return
  }
  if (event.type === "session.inbox.cancelled") {
    const data = event.data as Data<"session.inbox.cancelled">
    pending.delete(data.inboxID)
    return
  }
  if (event.type === "session.inbox.delivered") {
    const data = event.data as Data<"session.inbox.delivered">
    const item = pending.get(data.inboxID)
    pending.delete(data.inboxID)
    if (!item || item.sessionID !== data.sessionID) {
      invalid.add(data.sessionID)
      return
    }
    if (item.item.type !== "user") return
    if (!announced.has(data.sessionID)) {
      const info = await metadata(data.sessionID)
      if (!info.workingDirectory) throw new Error("Missing session directory")
      await core.event({ event: { type: "session.created", properties: { info: { id: data.sessionID, title: info.title, parentID: info.parentID } } } })
      announced.add(data.sessionID)
    }
    await core.event({ event: { type: "message.updated", properties: { info: { id: data.inboxID, role: "user", sessionID: data.sessionID } } } })
    await core.event({ event: { type: "message.part.updated", properties: { part: { messageID: data.inboxID, sessionID: data.sessionID, type: "text", text: item.item.payload.text } } } })
    seenUser.set(data.sessionID, data.inboxID)
    return
  }
  if (event.type === "session.text.ended") {
    const data = event.data as Data<"session.text.ended">
    await core.event({ event: { type: "message.updated", properties: { info: { id: data.assistantMessageID, role: "assistant", sessionID: data.sessionID } } } })
    await core.event({ event: { type: "message.part.updated", properties: { part: { messageID: data.assistantMessageID, sessionID: data.sessionID, type: "text", text: data.text } } } })
    return
  }
  if (event.type === "session.execution.succeeded") {
    const data = event.data as Data<"session.execution.succeeded">
    await core.event({ event: { type: "session.idle", properties: { sessionID: data.sessionID } } })
  }
}

export default Plugin.define({ id: "intaris", effect: adaptV2 })
