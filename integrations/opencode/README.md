# OpenCode Integration -- Guardrails

A plugin for [OpenCode](https://opencode.ai) that evaluates supported direct tool calls through Intaris's safety pipeline before allowing execution.

## Integration Approaches

Intaris offers two integration approaches for OpenCode. Choose one -- do **not** use both simultaneously, or tool calls will be evaluated twice.

### Approach A: Plugin (Recommended)

OpenCode V1 (1.18.29) and V2 (2.0.26) have separate entries using the shared `src/core.ts` guard: `intaris.ts` uses V1 `tool.execute.before`, and `intaris-v2.ts` registers V2 Effect `execute.before` / `execute.after` hooks. **Do not load both.** The V2 supervisor may continue startup if the plugin fails to load; check built-in plugin status rather than assuming it loaded. Loaded evaluator failures still deny by default. This gives you:

- Fine-grained control over error messages
- Configurable fail-open/fail-closed behavior
- Session lifecycle management
- Structured logging via OpenCode's log system
- Behavioral analysis: periodic checkpoints, session completion signals, agent summaries

### Approach B: MCP Proxy

Configure Intaris as a remote MCP server. OpenCode connects to Intaris's `/mcp` endpoint, which transparently proxies all tool calls through the safety pipeline. This gives you:

- Zero plugin code -- just configuration
- Full MCP proxy features (tool preferences, escalation retry, tool namespacing)
- Works with any MCP-compatible client

The trade-off is less control over the UX (no custom error messages, no fail-open option).

## Setup -- Plugin (Approach A)

### 1. Environment Variables

The plugin makes direct HTTP calls to the Intaris REST API. Set these in your shell profile:

```bash
export INTARIS_URL=http://localhost:8060
export INTARIS_API_KEY=your-api-key
export INTARIS_USER_ID=your-username       # required for single-key mode (INTARIS_API_KEY)
export INTARIS_AGENT_ID=opencode           # optional, defaults to "opencode"
export INTARIS_FAIL_OPEN=false             # optional, defaults to false
export INTARIS_INTENTION=""                # optional, auto-generated from cwd
export INTARIS_ALLOW_PATHS=~/src           # optional, allow reads from sibling projects
export INTARIS_CHECKPOINT_INTERVAL=25      # optional, defaults to 25 (0=disabled)
```

| Variable | Default | Description |
|---|---|---|
| `INTARIS_URL` | `http://localhost:8060` | Intaris server URL |
| `INTARIS_API_KEY` | (empty) | API key for authentication. **Required** if Intaris has `INTARIS_API_KEY` or `INTARIS_API_KEYS` set. |
| `INTARIS_API_KEY_FILE` | (unset) | Optional path to an owner-owned, owner-readable-only, nonsymlink regular file containing a single nonempty key line. When set, errors block plugin loading; never falls back to `INTARIS_API_KEY`. |
| `INTARIS_AGENT_ID` | `opencode` | Agent ID sent to Intaris |
| `INTARIS_USER_ID` | (empty) | User ID. **Required** when using `INTARIS_API_KEY` (single-key mode) — the server needs a user identity to scope sessions and audit records. Optional if `INTARIS_API_KEYS` maps your key to a specific user. |
| `INTARIS_FAIL_OPEN` | `false` | **Unsafe legacy opt-in:** if `true`, tool calls proceed when Intaris is unreachable. Leave unset/false for fail-closed operation. |
| `INTARIS_INTENTION` | (auto) | Session intention override. Default: `"OpenCode coding session in <cwd>"` |
| `INTARIS_ALLOW_PATHS` | (empty) | Comma-separated parent directories to allow reads from without LLM evaluation. Supports `~` expansion. E.g., `~/src` allows reads from all projects under `~/src/`. |
| `INTARIS_CHECKPOINT_INTERVAL` | `25` | Number of evaluate calls between periodic checkpoints. Set to `0` to disable checkpoints. Each checkpoint consumes one rate limit slot. |

### 2. Install the Plugin

From `integrations/opencode/`, build with project-local dependencies (`bun install && bun run typecheck && bun test test/core.test.ts && bun run build:v1`). Copy **only** the bundled artifact to the OpenCode V1 plugins directory (never copy `src/core.ts` into an autoload directory):

```bash
# Global (recommended -- guardrails apply to all projects)
mkdir -p ~/.config/opencode/plugins
cp dist/intaris-v1.js ~/.config/opencode/plugins/intaris.js

# Or project-level
mkdir -p .opencode/plugins
cp dist/intaris-v1.js .opencode/plugins/intaris.js
```

Local V1 plugins are loaded automatically -- no config entry needed. Remove any older Intaris autoload entry before installing to avoid duplicate active guards.

### OpenCode V2 (2.0.26) build and installation candidate

From `integrations/opencode/`, install project-local dependencies and run `bun run typecheck && bun test test/core.test.ts test/v2.test.ts && bun run build:v2`. The V2 artifact is `dist/v2/index.js` (default export: Effect `Plugin.define({ id: "intaris", effect })`). Keep `@opencode/plugin@2.0.26`, `@opencode/schema@2.0.26` and `effect@4.0.0-rc.112` installed in an ancestor `node_modules` of this directory; they are external imports, not bundled into the artifact. OpenCode V2 resolves an explicit local plugin **directory** by its `index.js` entry, not a `.js` file path; point the V2 local plugin configuration at the absolute `dist/v2/` directory. Do not copy other `.ts` sources into an autoload plugin directory.

**Validation scope:** A Docker smoke test with pinned OpenCode 2.0.26 (image `sha256:9594cedeb9019fd376a87f04e219b5e41a609cb704391899188d05b2a1e13183`), a local synthetic streaming model and a mock evaluator reported `intaris` active via `/api/plugin`. Approve, deny and evaluator HTTP 503 each made `/intention`, `/reasoning` and exactly one `/evaluate` call; only the approved shell call created the marker. This used an ignored local smoke harness, not a distributed checkout test. Typecheck, both builds/artifact loads, 37 TypeScript tests and two focused Python policy tests also passed. **Not exercised in a real V2 runtime:** real provider/Intaris credentials, MCP, cancellation and CodeMode; V1 runtime was not exercised. This is not production acceptance.

For visible load status use V2's built-in `/plugins` or `opencode plugin list`; `GET /api/plugin` provides location inventory. The adapter also logs when its hooks register. Inventory confirms registration, **not** Intaris evaluator health. V2 retrieves the durable admitted user context before evaluation and blocks calls if that context is absent or inconsistent with an observed delivery; queued/cancelled input is not treated as active intention. The V2 `execute` CodeMode outer tool is rejected conservatively: registered nested tools do run before hooks, but CodeMode's separate HTTP extension and reused outer call ID cannot be fully correlated or mediated yet. Configure direct native tools with `codemode: false` and MCP servers with each `mcp.servers.<name>.codemode: false`; do not invent a global CodeMode field. The separate host permissions rule `[{"action":"execute","resource":"*","effect":"deny"}]` must be effective as the **last** matching rule if CodeMode is enabled elsewhere. Native V2 `shell` commands are evaluated as `bash` (original name retained as `_intaris_opencode_tool` in arguments); other native, custom and MCP tool names are sent unchanged.

### 3. Verify

Run OpenCode with `--print-logs` and look for:

```
[intaris] Plugin initialized
[intaris] Session created: oc-<session-id>
```

## Setup -- MCP Proxy (Approach B)

Add to your `~/.config/opencode/opencode.json` (global) or `opencode.json` (project):

```json
{
  "mcp": {
    "intaris": {
      "type": "remote",
      "url": "http://localhost:8060/mcp",
      "headers": {
        "Authorization": "Bearer your-api-key",
        "X-Agent-Id": "opencode",
        "X-Intaris-Intention": "OpenCode coding session"
      }
    }
  }
}
```

Configure upstream MCP servers in Intaris (via the UI, REST API, or `MCP_CONFIG_FILE`). OpenCode will see all upstream tools namespaced as `server_name:tool_name`.

See `opencode.json` in this directory for a complete example.

## OpenCode Permissions

Keep OpenCode's built-in role-specific permission restrictions. Intaris's evaluator is an additional gate, not a replacement for host permissions; setting blanket `"permission": "allow"` removes protections when the plugin is not loaded. If a specific host approval duplicates Intaris approval, review only that narrow permission with your operator before changing it. See [OpenCode Permissions](https://opencode.ai/docs/permissions/) for details.

## How It Works

### Plugin Flow

1. **`session.created`**: Creates an Intaris session via `POST /api/v1/intention` with the working directory as context. Child sessions (subagent tasks) are created with `parent_session_id` for session chain analysis.
2. **`tool.execute.before`**: Before every tool call:
   - Ensures an Intaris session exists (lazy creation for resumed sessions)
   - Calls `POST /api/v1/evaluate` with the tool name and arguments
   - **approve**: tool executes normally
   - **deny**: throws an error with reasoning (blocks execution)
   - **escalate**: polls only when `/evaluate` still returns unresolved escalation
   - Tracks per-decision statistics (approve/deny/escalate counts)
   - Sends periodic checkpoints via `POST /api/v1/checkpoint` (every N calls)
3. **`session.deleted`**: Signals session completion to Intaris:
   - `PATCH /api/v1/session/{id}/status` to `"completed"`
   - `POST /api/v1/session/{id}/agent-summary` with session statistics

During normal inactivity, the plugin uses `session.idle` to transition parent sessions to `idle` rather than `completed`. Child sessions still complete when they become idle.

### MCP Proxy Flow

1. OpenCode connects to Intaris at `/mcp` as a remote MCP server.
2. `tools/list` returns aggregated tools from all upstream servers.
3. `tools/call` evaluates each call through the safety pipeline before forwarding.

## Behavioral Analysis

The plugin supports Intaris's behavioral analysis pipeline (L1 data collection):

- **Periodic checkpoints**: Every `INTARIS_CHECKPOINT_INTERVAL` evaluate calls, the plugin sends a checkpoint with enriched content (call counts, decision breakdown, recent tool names). Checkpoints share the rate limit budget with evaluate calls.
- **Session completion**: When a session is explicitly deleted, the plugin signals completion (`PATCH /status`) and sends an agent summary with session statistics.
- **Parent session tracking**: Child sessions (subagent tasks) are created with `parent_session_id` linking them to the parent session for chain analysis.

**Limitation**: OpenCode does not fire a session-end event on normal exit. If the user closes OpenCode without deleting the session, the completion signal is not sent. The server's background sweep handles this by transitioning idle sessions after `SESSION_IDLE_TIMEOUT_MINUTES`.

## Tool Name Conventions

When using the **plugin** approach, tool names are passed as-is from OpenCode to Intaris. OpenCode uses these tool names:

- Built-in tools: `read`, `edit`, `write`, `bash`, `glob`, `grep`
- MCP tools: The MCP tool name directly (e.g., `add_memory`, `search_memories`)

When configuring Intaris session policies (fnmatch patterns), use these names:

```json
{
  "policy": {
    "allow": ["read", "glob", "grep"],
    "deny": ["bash"]
  }
}
```

When using the **MCP proxy** approach, tools are namespaced as `server_name:tool_name` (e.g., `mnemory:add_memory`).

## Troubleshooting

- **"Evaluation failed" or "Evaluation rejected" with no server logs**: The server rejected the request before reaching the evaluation pipeline (HTTP 401). The most common cause is missing user identity: if you use `INTARIS_API_KEY` (single shared key), you **must** also set `INTARIS_USER_ID`. Without it, the server cannot determine which user the request belongs to. Alternatively, switch to `INTARIS_API_KEYS` on the server with a key mapped to your user_id.
- **Tool calls blocked unexpectedly**: Check that `INTARIS_URL` and `INTARIS_API_KEY` are set correctly. Run OpenCode with `--print-logs` to see evaluation decisions.
- **"Cannot create session" errors**: Verify Intaris is running and reachable at the configured URL.
- **All tool calls allowed**: Ensure the plugin is loaded (check for "Plugin initialized" in logs). If using `INTARIS_FAIL_OPEN=true`, Intaris may be unreachable.
- **Slow first tool call**: The first tool call creates an Intaris session (~1-2s) before evaluating. Subsequent calls are faster.
- **Double evaluation**: If you see each tool call evaluated twice, you may have both the plugin and MCP proxy configured. Use only one approach.
- **Checkpoints not appearing**: Verify `INTARIS_CHECKPOINT_INTERVAL` is not set to `0`. Check that the rate limit is not exhausted (checkpoints share the budget with evaluate calls).
