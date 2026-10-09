import assert from 'node:assert/strict'
import { test } from 'node:test'
import { IntarisPlugin } from './intaris.ts'

const response = (data, status = 200) => ({
  ok: status < 400,
  status,
  json: async () => data,
  text: async () => JSON.stringify(data),
})

async function run({ audit = [], recheck = { decision: 'approve', call_id: 'recheck', path: 'fast' },
  failOpen = false, mutate = false, mutateRecheck = false, deleteAt = '',
  initial = {}, pollStatus = 200, timeout = 10, auditDelay = 0,
  auditNetworkError = false, recheckNetworkError = false } = {}) {
  process.env.INTARIS_API_KEY = 'test-key'
  process.env.INTARIS_FAIL_OPEN = String(failOpen)
  process.env.INTARIS_CHECKPOINT_INTERVAL = '0'
  process.env.INTARIS_ESCALATION_TIMEOUT = String(timeout)
  const oldFetch = globalThis.fetch
  const oldTimeout = globalThis.setTimeout
  const oldNow = Date.now
  let clock = 0
  let evaluations = 0
  let executed = 0
  const calls = []
  const toasts = []
  let plugin
  let args
  try {
    Date.now = () => clock
    globalThis.setTimeout = (fn, ms) => { clock += ms; queueMicrotask(fn); return 1 }
    globalThis.fetch = async (url, options) => {
      const path = new URL(url).pathname
      calls.push({ path, method: options.method, body: options.body && JSON.parse(options.body) })
      if (path === '/api/v1/intention') return response({ session_id: 'oc-test' })
      if (path.endsWith('/status') || path.endsWith('/agent-summary')) return response({}, 500)
      if (path === '/api/v1/evaluate') {
        evaluations++
        if (evaluations === 2) {
          if (mutateRecheck) args.nested.value = 2
          if (deleteAt === 'recheck') await plugin.event({ event: { type: 'session.deleted', properties: { info: { id: 'test' } } } })
          if (recheckNetworkError) throw new Error('recheck offline')
        }
        return response(evaluations === 1
          ? { decision: 'deny', call_id: 'denied-id', path: 'critical', reasoning: 'blocked', ...initial }
          : recheck)
      }
      if (path === '/api/v1/audit/denied-id') {
        clock += auditDelay
        if (deleteAt === 'poll') await plugin.event({ event: { type: 'session.deleted', properties: { info: { id: 'test' } } } })
        if (auditNetworkError) throw new Error('audit offline')
        const next = audit.shift() ?? { call_id: 'denied-id', decision: 'deny' }
        if (mutate && calls.filter(c => c.path === path).length === 1) args.command = 'changed'
        return response(next, pollStatus)
      }
      throw new Error(`unexpected request: ${path}`)
    }
    plugin = await IntarisPlugin({
      client: { app: { log: async () => {} }, tui: { showToast: async ({ body }) => { toasts.push(body.message) } } },
      worktree: '/project', directory: '/project',
    })
    args = { command: 'safe-placeholder', nested: { value: 1 } }
    const input = { tool: 'bash', sessionID: 'test', callID: 'tool-call' }
    const pending = plugin['tool.execute.before'](input, { args }).then(() => { executed++ })
    // Drain asynchronous fetch continuations without advancing the controlled poll timer.
    for (let i = 0; i < 8; i++) await Promise.resolve()
    assert.equal(executed, 0, 'tool must remain blocked before the audit decision')
    let error = null
    try { await pending } catch (err) { error = err }
    assert.equal(calls.some(c => c.path === '/api/v1/decision'), false)
    return { calls, toasts, executed, error, evaluations }
  } finally {
    globalThis.fetch = oldFetch
    globalThis.setTimeout = oldTimeout
    Date.now = oldNow
  }
}

const pending = { call_id: 'denied-id', decision: 'deny' }
const approved = { ...pending, user_decision: 'approve', resolved_by: 'user' }
const userDenied = { ...pending, user_decision: 'deny', resolved_by: 'user' }

test('pending audit retains invocation, human approval rechecks same original args exactly once', async () => {
  const result = await run({ audit: [pending, approved] })
  assert.equal(result.error, null)
  assert.equal(result.executed, 1)
  assert.equal(result.evaluations, 2)
  assert.deepEqual(result.calls.filter(c => c.path === '/api/v1/evaluate').map(c => c.body.args),
    [{ command: 'safe-placeholder', nested: { value: 1 } }, { command: 'safe-placeholder', nested: { value: 1 } }])
  assert.ok(result.toasts.some(t => t.includes('denied-id')))
})

test('human denial never executes', async () => {
  const result = await run({ audit: [userDenied] })
  assert.equal(result.executed, 0)
  assert.equal(result.evaluations, 1)
  assert.match(result.error.message, /DENIED by user/)
})

test('judge denial waits for later human override', async () => {
  const escalated = { ...pending, decision: 'escalate' }
  const result = await run({ audit: [
    { ...escalated, user_decision: 'deny', resolved_by: 'judge' },
    { ...escalated, user_decision: 'approve', resolved_by: 'user' },
  ] })
  assert.equal(result.executed, 1)
  assert.equal(result.evaluations, 2)
})

test('timeout, audit error, malformed or unauthorized audit never execute even fail-open', async () => {
  for (const options of [
    { audit: [], timeout: 1 },
    { audit: [pending], pollStatus: 401 },
    { audit: [approved], auditNetworkError: true },
    { audit: [approved], timeout: 1 },
    { audit: [approved], timeout: 3, auditDelay: 1500 },
    { audit: [{ user_decision: 'approve', resolved_by: 'user' }] },
    { audit: [{ ...pending, user_decision: 'approve', resolved_by: 'judge' }] },
    { audit: [{ ...pending, user_decision: 'approve' }] },
  ]) {
    const result = await run({ ...options, failOpen: true })
    assert.equal(result.executed, 0)
    assert.equal(result.evaluations, 1)
    assert.ok(result.error)
  }
})

test('approval arriving after sleep or slow HTTP deadline never triggers re-evaluation', async () => {
  const afterSleep = await run({ audit: [approved], timeout: 1, failOpen: true })
  assert.equal(afterSleep.calls.filter(c => c.path === '/api/v1/audit/denied-id').length, 0)
  const afterHttp = await run({ audit: [approved], timeout: 3, auditDelay: 1500, failOpen: true })
  assert.equal(afterHttp.calls.filter(c => c.path === '/api/v1/audit/denied-id').length, 1)
  for (const result of [afterSleep, afterHttp]) {
    assert.equal(result.executed, 0)
    assert.equal(result.evaluations, 1)
    assert.match(result.error.message, /TIMEOUT/)
  }
})

test('recheck failures and non-approvals never execute even fail-open', async () => {
  for (const recheck of [
    { decision: 'deny', call_id: 'second', path: 'critical' },
    { decision: 'deny', call_id: 'second', path: 'fast', session_status: 'suspended' },
    { decision: 'deny', call_id: 'second', path: 'fast', session_status: 'terminated' },
    { decision: 'escalate', call_id: 'second', path: 'alignment' },
    { decision: 'approve' },
    null,
  ]) {
    const result = await run({ audit: [approved], recheck, failOpen: true })
    assert.equal(result.executed, 0)
    assert.ok(result.error)
  }
  const offline = await run({ audit: [approved], recheckNetworkError: true, failOpen: true })
  assert.equal(offline.executed, 0)
  assert.equal(offline.evaluations, 2)
  assert.ok(offline.error)
})

test('mutated arguments never execute or get re-evaluated', async () => {
  const result = await run({ audit: [approved], mutate: true })
  assert.equal(result.executed, 0)
  assert.equal(result.evaluations, 1)
  assert.match(result.error.message, /changed/)
})

test('mutation during re-evaluation blocks execution', async () => {
  const result = await run({ audit: [approved], mutateRecheck: true })
  assert.equal(result.evaluations, 2)
  assert.equal(result.executed, 0)
  assert.ok(result.error)
})

test('deleted session blocks polling and recheck even if completion request fails', async () => {
  for (const deleteAt of ['poll', 'recheck']) {
    const result = await run({ audit: [approved], deleteAt, failOpen: true })
    assert.equal(result.executed, 0)
    assert.equal(result.evaluations, deleteAt === 'poll' ? 1 : 2)
    assert.ok(result.error)
  }
})

test('ordinary escalations retain human and judge resolution behavior', async () => {
  for (const [decision, expected] of [
    [{ ...pending, user_decision: 'approve', resolved_by: 'user' }, 1],
    [{ ...pending, user_decision: 'approve', resolved_by: 'judge' }, 1],
    [{ ...pending, user_decision: 'deny', resolved_by: 'judge' }, 0],
  ]) {
    const result = await run({ initial: { decision: 'escalate' }, audit: [decision] })
    assert.equal(result.executed, expected)
    assert.equal(result.evaluations, 1)
  }
})

test('structural denial and denial missing call ID do not poll', async () => {
  for (const initial of [
    { session_status: 'terminated' },
    { session_status: 'completed', path: 'critical' },
    { call_id: '', path: 'critical' },
  ]) {
    const result = await run({ initial })
    assert.equal(result.executed, 0)
    assert.equal(result.calls.filter(c => c.path.startsWith('/api/v1/audit/')).length, 0)
  }
})
