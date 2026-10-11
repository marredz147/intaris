/** OpenCode V1 (1.18.29) autoload entry. Build this file for single-file installation. */
import type { Plugin } from "@opencode-ai/plugin"
import { createCore } from "./src/core"

export const IntarisPlugin: Plugin = async (context) => {
  const hooks = await createCore(context)
  return { ...hooks, dispose: async () => { hooks.dispose() } }
}
