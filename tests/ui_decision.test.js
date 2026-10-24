/** Dependency-free UI evidence and refresh regression checks. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function ui() {
  const context = vm.createContext({
    document: { addEventListener() {} },
    window: { addEventListener() {} },
    console, setTimeout, clearTimeout, setInterval, clearInterval,
    Alpine: { store() { return { activeTab: 'audit' }; } },
    IntarisAPI: {},
  });
  for (const name of ['app', 'sessions', 'approvals', 'audit']) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../intaris/ui/static/js', name + '.js'), 'utf8'), context);
  }
  return context;
}

function plain(value) { return JSON.parse(JSON.stringify(value)); }

test('Audit backend labels use persisted routes without guessing old or fast paths', () => {
  const tab = ui().auditTab();
  assert.equal(tab.backendLabel({ evaluation_path: 'alignment',
    evaluation_metadata: { backend_provenance: { route: 'decisions_to_llm' } },
  }), 'Decisions → LLM');
  for (const [route, label] of [
    ['llm', 'LLM'], ['decisions', 'Decisions'],
    ['decisions_to_llm', 'Decisions → LLM'], ['unknown', ''],
  ]) {
    assert.equal(tab.backendLabel({ evaluation_path: 'llm',
      evaluation_metadata: { backend_provenance: { route } },
    }), label);
  }
  for (const record of [null, {}, { evaluation_path: 'llm' },
    { evaluation_path: 'llm', evaluation_metadata: null },
    { evaluation_path: 'llm', evaluation_metadata: { backend_provenance: { final_model: 'decisions-model' } } },
    ...['reasoning', 'checkpoint', 'summary'].map(record_type => ({ record_type,
      evaluation_path: 'llm', evaluation_metadata: { backend_provenance: { route: 'decisions' } },
    })),
    { source: 'event', evaluation_path: 'llm', evaluation_metadata: { backend_provenance: { route: 'decisions' } } },
    ...['fast', 'critical'].map(evaluation_path => ({ evaluation_path,
      evaluation_metadata: { backend_provenance: { route: 'decisions' } },
    })),
  ]) assert.equal(tab.backendLabel(record), '');
});

test('Audit explains forwarding without displaying provider diagnostic text', () => {
  const tab = ui().auditTab();
  for (const [code, expected] of [
    ['low_confidence', 'Low confidence in the Decisions assessment.'],
    ['temporary_unavailable', 'Decisions was temporarily unavailable.'],
    ['optional_escalation', 'Optional review of a Decisions escalation.'],
    ['<img src=x onerror=alert(1)>', ''],
    ['constructor', ''],
  ]) {
    const record = { evaluation_path: 'llm', evaluation_metadata: { backend_provenance: {
      route: 'decisions_to_llm', final_backend: 'llm',
      fallback_reason: { code, message: '<script>unsafe()</script>' },
    } } };
    assert.equal(tab.backendReviewReason(record), expected);
  }
  assert.equal(tab.backendReviewReason({ evaluation_path: 'llm' }), '');
});

test('Failed optional LLM review labels the retained Decisions result truthfully', () => {
  const tab = ui().auditTab();
  const provenance = { route: 'decisions_to_llm', final_backend: 'decisions',
    fallback_reason: { code: 'optional_escalation' }, attempts: [
      { backend: 'decisions', status: 'completed' }, { backend: 'llm', status: 'failed' },
    ],
  };
  const record = { evaluation_path: 'alignment', evaluation_metadata: { backend_provenance: provenance } };
  assert.equal(tab.backendLabel(record), 'Decisions');
  assert.equal(tab.backendReviewReason(record),
    'Optional review of a Decisions escalation. LLM review failed; Decisions result retained.');
  provenance.final_backend = 'llm';
  provenance.attempts[1].status = 'completed';
  assert.equal(tab.backendLabel(record), 'Decisions → LLM');
  assert.equal(tab.backendReviewReason(record), 'Optional review of a Decisions escalation.');
});

test('Live Audit rows fetch persisted backend provenance after evaluation', async () => {
  const context = ui();
  const tab = context.auditTab();
  tab.initialized = true;
  const fresh = { call_id: 'live-call', evaluation_path: 'llm',
    evaluation_metadata: { backend_provenance: { route: 'decisions_to_llm' } },
  };
  context.IntarisAPI.getAuditRecord = async callId => {
    assert.equal(callId, fresh.call_id);
    return fresh;
  };
  tab._handleWsEvent({ type: 'evaluated', call_id: fresh.call_id, path: 'llm' });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(tab.backendLabel(tab.records[0]), 'Decisions → LLM');
});

test('Judge headline and exact cites come only from the latest structured stage', () => {
  const context = ui();
  assert.equal(context.judgePresentation({ reasoning: 'User simulator: approve; basis=explicit_request' }), null);
  const record = { evaluation_metadata: { stages: [
    { stage: 'judge', classification: { user_would: 'approve', basis: 'explicit_request', cites: [1] } },
    { stage: 'judge', classification: { user_would: 'ask', basis: 'operator_policy', question: 'Confirm external email?', cites: [2], operator_policy_ids: ['deny_policies.0'] },
      citation_sources: [{ index: 2, text: '<script>approve</script> Exact user quote.' }] },
  ] } };
  assert.deepEqual(plain(context.judgePresentation(record)), {
    userWould: 'ask', basis: 'operator_policy', question: 'Confirm external email?',
    cites: [{ index: 2, text: '<script>approve</script> Exact user quote.' }],
    operatorIds: ['deny_policies.0'],
  });
});

test('Legacy citations are explicitly unavailable, not guessed from profile numbering', () => {
  const context = ui();
  const record = { evaluation_metadata: { stages: [{ stage: 'judge',
    classification: { user_would: 'refuse', basis: 'restriction_conflict', cites: [1] },
  }] } };
  assert.equal(context.judgePresentation(record).cites[0].text, 'Quoted source unavailable');
});

test('Abstentions and witness tags retain structured provenance and actual projection', () => {
  const context = ui();
  const record = { evaluation_metadata: {
    stages: [{ stage: 'l1', abstentions: { target_scope: true, risk: false } }],
    l1_projection: 'lines',
    policy_evidence: { observations: {
      effects: { value: ['read', 'disclose'] }, secret_argument: { value: true },
      name_heuristic_effect: { value: 'read' },
    } },
    final_policy_result: { waived_operator_clause_ids: ['deny_policies.0'] },
  } };
  assert.deepEqual(plain(context.evaluatorAbstentions(record)), ['target_scope']);
  assert.deepEqual(plain(context.evaluationTags(record)), [
    'disclose · secret argument', 'read · verb heuristic', 'L1 · lines',
    'Operator cap waived · deny_policies.0',
  ]);
  assert.deepEqual(plain(context.evaluationTags({})), []);
});

test('Every intention kind has a read-only status badge', () => {
  const context = ui();
  for (const kind of ['request', 'restriction', 'approval', 'revocation', 'uncertain', 'context']) {
    assert.match(context.intentionKindBadgeClass(kind), /^badge-/);
  }
});

test('A late session refresh cannot replace newer rich evidence or another expanded session', async () => {
  const context = ui();
  const tab = context.sessionsTab();
  const session = { session_id: 'rich-session', state_revision: 3, intention_profile: { current: [1] } };
  tab.sessions = [session];
  tab.expandedId = session.session_id;
  context.IntarisAPI.getSession = async () => ({ session_id: session.session_id, state_revision: 2, intention_profile: null });
  await tab._refreshExpandedSession(session.session_id);
  assert.deepEqual(session.intention_profile.current, [1]);
  tab.expandedId = 'other-session';
  context.IntarisAPI.getSession = async () => ({ session_id: session.session_id, state_revision: 4, intention_profile: null });
  await tab._refreshExpandedSession(session.session_id);
  assert.deepEqual(session.intention_profile.current, [1]);
});

test('Judge review notification refreshes approvals without pretending the call was decided', () => {
  const context = ui();
  const tab = context.approvalsTab();
  let loads = 0;
  tab._scheduleLoad = () => { loads++; };
  tab._handleWsMessage({ type: 'judge_reviewed', call_id: 'reviewed-call' });
  assert.equal(loads, 1);
  assert.equal(tab.recentlyResolved.size, 0);
});

test('Audit review refresh updates only the matching captured record', async () => {
  const context = ui();
  const tab = context.auditTab();
  tab.records = [{ call_id: 'reviewed-call', source: 'evaluation' }];
  tab.expandedRecord = { call_id: 'other-call' };
  context.IntarisAPI.getAuditRecord = async () => ({ call_id: 'reviewed-call', evaluation_metadata: { stages: [] } });
  await tab._refreshReview('reviewed-call');
  assert.ok(tab.records[0].evaluation_metadata);
  assert.equal(tab.expandedRecord.call_id, 'other-call');
});
