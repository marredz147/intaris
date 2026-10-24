<p align="center">
  <img src="files/banner.jpg" alt="Intaris">
</p>

# intaris

Intaris helps detect when an AI agent may be drifting from the user's intention. It combines intention-aware evaluation of tool calls with behavioral analysis over a session and across sessions. Works with [OpenCode](https://opencode.ai), [Claude Code](https://docs.anthropic.com/en/docs/claude-code), [OpenClaw](https://github.com/fpytloun/openclaw/tree/v2026.3.13), and any MCP-compatible client.

**Intention guardrails.** Intaris evaluates tool use against the user's stated intention, including explicit restrictions. A read-only fast path handles routine calls; other calls can go through model evaluation and, when needed, escalation. Evaluations are recorded in an audit trail.

**Behavior analytics.** Session summaries and cross-session analysis can surface patterns that a single tool call may not show.

**Real-time.** Sub-second evaluation with a priority-ordered decision matrix. Read-only calls resolve in under 1ms. LLM evaluations complete within the 5-second circuit breaker. WebSocket streaming for live monitoring.

**Self-hosted.** Single Python process, SQLite or PostgreSQL storage, no external dependencies beyond an LLM API key. Your code and audit trail stay under your control.

Intaris is not a sandbox, an antivirus, or proof that an agent cannot construct harmful actions. Pair it with a sandbox that isolates filesystem and network access and limits available tools when working with untrusted agents.

See [Intaris purpose and threat model](docs/purpose-and-threat-model.md) for scope and deployment guidance.

Part of the [Cognara](https://github.com/fpytloun) platform (Cognis controller, Intaris guardrails, [Mnemory](https://github.com/fpytloun/mnemory) memory).

## Features

- **Tool-call assessment** -- Native OpenAI Decisions, Jev, or an OpenAI-compatible LLM assesses action facts. Low-confidence Decisions assessments receive generative L1 review before judge escalation. Assessment is not itself permission.
- **Policy-controlled decisions** -- The hybrid pipeline combines controller evidence, policy obligations and bounded model assessments. The legacy pipeline remains available; identify the mode when comparing results.
- **Source-aware review** -- Independent Judge review can consider original task instructions and scoped runtime evidence without treating generated summaries as user consent.
- **Session management** -- Hierarchical parent/child sessions with intention tracking, lifecycle states, and idle sweep.
- **Intention tracking** -- User-driven intention model with IntentionBarrier for real-time updates and AlignmentBarrier for parent/child enforcement.
- **MCP proxy** -- Sits between clients and upstream MCP servers, applying Intaris evaluation with per-tool preference overrides.
- **Audit trail** -- Every evaluation is logged with decision, reasoning, risk level, classification, latency, and redacted arguments.
- **Secret redaction** -- API keys, passwords, tokens, and connection strings are automatically redacted before audit storage.
- **Filesystem path protection** -- Working directory enforcement with approved path prefix learning from LLM approvals.
- **Session recording** -- Full-fidelity event logs with live tailing, playback, chunked ndjson storage (filesystem or S3), and a bounded local cache for immutable S3 chunks.
- **Cognis-ready auth** -- Accepts Cognis-issued ES256 JWTs for service-to-service calls while keeping standalone API key auth for direct clients.
- **Behavioral analysis** -- Three-layer system: per-call data collection, session summaries, and cross-session behavioral profiling.
- **Management UI** -- Built-in web dashboard with session tree view, audit log, approval queue, MCP server management, and real-time charts.
- **Judge review** -- A separately configured model reviews eligible escalations within policy limits. Review can approve, refuse, ask, or be unavailable; it is not a guarantee against unnecessary escalations or unsafe decisions.
- **Webhook callbacks** -- HMAC-signed escalation notifications for external approval systems.
- **Notification channels** -- Per-user push notifications (Pushover, Slack, webhook) with one-click approve/deny action links.
- **Rate limiting** -- Per-session sliding window rate limiter to prevent runaway agents.

## Quick Start

The experimental hybrid runtime also supports [single-call rich intention
generation](docs/single-call-intention.md), without a second extraction request.

Intaris needs an OpenAI-compatible API key for safety evaluation. It picks up `LLM_API_KEY` from your environment automatically.

```bash
LLM_API_KEY=sk-your-key uvx intaris
```

That's it. Intaris starts on `http://localhost:8060`, management UI at `http://localhost:8060/ui`.

Now integrate with your agent. We already ship extensions for some clients. For
example for **OpenCode**, install the plugin:

```bash
export INTARIS_URL=http://localhost:8060
cp integrations/opencode/intaris.ts ~/.config/opencode/plugins/
```

Intaris can also serve as MCP proxy with audit trail and guardrails for tool calls. To use that, configure **any MCP client** to use intaris as a single MCP server:

```json
{
  "mcpServers": {
    "intaris": {
      "type": "streamable-http",
      "url": "http://localhost:8060/mcp"
    }
  }
}
```

And add MCP servers via Intaris UI or config.

Intaris is also available via [Docker, pip, or production setup](docs/deployment.md). See the [full quick start guide](docs/quickstart.md) for more clients and options.

## Screenshots

<p align="center">
  <img src="files/screenshots/ui-dashboard.png" width="800" alt="Dashboard">
  <br><em>Dashboard -- evaluation metrics, decision distribution, performance stats, and activity timeline</em>
</p>

<p align="center">
  <img src="files/screenshots/ui-sessions.png" width="800" alt="Sessions">
  <br><em>Sessions -- hierarchical tree view with expandable session details and recent evaluations</em>
</p>

<p align="center">
  <img src="files/screenshots/ui-approvals.png" width="800" alt="Approvals">
  <br><em>Approvals -- pending escalations with reasoning, arguments, and one-click approve/deny</em>
</p>

<p align="center">
  <img src="files/screenshots/ui-analysis-profile.png" width="800" alt="Behavioral Profile">
  <br><em>Analysis -- behavioral risk profile with per-agent risk indicators and trends</em>
</p>

<p align="center">
  <img src="files/screenshots/ui-analysis-trend.png" width="800" alt="Analysis Trend">
  <br><em>Analysis -- cross-session behavioral trend tracking over time</em>
</p>

<p align="center">
  <img src="files/screenshots/ui-session-suspicious.png" width="800" alt="Suspicious Session">
  <br><em>Sessions -- suspicious session detail with evaluation reasoning and risk assessment</em>
</p>

<p align="center">
  <img src="files/screenshots/ui-critical-denied.png" width="800" alt="Critical Denied">
  <br><em>Audit -- critical tool execution denied with detailed reasoning</em>
</p>

See the [Management UI docs](docs/management-ui.md) for all tabs and features.

## Supported Clients

| Client | Integration | Setup Guide |
|---|---|---|
| [OpenCode](https://opencode.ai) | Plugin (`intaris.ts`) | [OpenCode Guide](docs/clients/opencode.md) |
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | Hooks (bash scripts) | [Claude Code Guide](docs/clients/claude-code.md) |
| [OpenClaw](https://openclaw.ai) | Plugin ([`@fpytloun/openclaw-intaris`](https://www.npmjs.com/package/@fpytloun/openclaw-intaris)) | [OpenClaw Guide](docs/clients/openclaw.md) |
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) | Plugin ([`hermes-intaris`](https://pypi.org/project/hermes-intaris/)) | [Hermes Guide](docs/clients/hermes.md) |
| Any MCP client | MCP proxy (`/mcp` endpoint) | [MCP Proxy Guide](docs/mcp-proxy.md) |

**Plugin/Hooks** give fine-grained control: custom error messages, fail-open/fail-closed behavior, session lifecycle management, and behavioral analysis. **MCP proxy** is zero-code configuration but with less UX control.

## How It Works

**Intercept.** The client integration (plugin, hooks, or MCP proxy) captures every tool call before execution and sends it to Intaris for evaluation.

**Establish context.** The hybrid pipeline distinguishes authenticated user/controller evidence from agent descriptions, tool output and generated summaries. Coverage and provenance matter; loaded skills and shell-target details are context, not grants.

**Assess and decide.** The configured evaluator supplies bounded action assessments. Deterministic policy owns permissions and outcome limits; the classifier does not become the final authority on task scope or user consent.

**Review when required.** Judge independently reviews eligible escalations using original task evidence. Unavailable review is distinct from a user refusal. Transient model and routing failures use bounded retries; retries do not authorize or execute the proposed tool.

This describes the hybrid pipeline. Legacy behavior remains selectable; see [stage separation](docs/evaluation-stage-separation.md), [configuration](docs/configuration.md), and the [release verification record](docs/release-verification-2026-10.md).

## Benchmark Results

Benchmark results apply to a specific revision, corpus, configuration and grading method—not every deployment. Earlier headline scores are not current release measurements. Historical experiments include both harmful approvals and unnecessary blocks; a targeted regression pass cannot replace a complete rerun.

The [final balanced authorization replay](docs/balanced-authorization-acceptance-2026-10-15.md) collected 225 action observations and recorded 144 history checkpoints. The earlier four forged-push approvals are now blocked, while valid scoped push, same-action human approval, local tests and lint remain approved across three trials. All 27 benign source-counterfactual observations are approved. History qualification remains incomplete: 21 checkpoint rows lack established evidence. The report preserves the baseline, failed iterations, label caveats and cumulative cost rather than presenting a universal safety score.

See the [Benchmarking docs](docs/benchmarking.md) for methodology, scenario details, and how to run your own benchmarks.
The [September Jev assessment](docs/jev-evaluator.md) and [October redesign measurements](docs/evaluation-redesign-2026-10.md) are historical comparisons, not measurements of the current release candidate.

## Documentation

| Document | Description |
|---|---|
| [Purpose and Threat Model](docs/purpose-and-threat-model.md) | Intaris's purpose, scope, and recommended deployment layers |
| [Quick Start](docs/quickstart.md) | Get running in 5 minutes |
| [Architecture](docs/architecture.md) | System design, layers, and key decisions |
| [Evaluation Pipeline](docs/evaluation-pipeline.md) | Classification, LLM evaluation, and decision matrix |
| [Jev Evaluator](docs/jev-evaluator.md) | Experimental L1 backend, measured Groq comparison, and status |
| [Configuration](docs/configuration.md) | Environment variable reference |
| [REST API](docs/rest-api.md) | Full API endpoint reference |
| [MCP Proxy](docs/mcp-proxy.md) | MCP proxy setup, tool namespacing, and preferences |
| [Management UI](docs/management-ui.md) | Built-in web dashboard |
| [Deployment](docs/deployment.md) | Production deployment guide |
| [Development](docs/development.md) | Contributing, tests, and code conventions |
| [OpenCode Integration](docs/clients/opencode.md) | OpenCode plugin setup |
| [Claude Code Integration](docs/clients/claude-code.md) | Claude Code hooks setup |
| [OpenClaw Integration](docs/clients/openclaw.md) | OpenClaw extension setup |
| [Benchmarking](docs/benchmarking.md) | Guardrails benchmark system |

## License

Business Source License 1.1 — see [LICENSE](LICENSE) for the full text.

The Licensed Work is (c) 2026 Filip Pytloun. You may use the Software for your own internal business operations free of charge. Commercial use (SaaS, managed services, or as a component of a commercial product) requires a separate license. On the Change Date (2030-03-15), the license converts to Apache License 2.0.

For alternative licensing arrangements, contact: filip@pytloun.cz
