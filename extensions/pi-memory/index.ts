/**
 * pi-memory — Persistent Agent Memory Extension
 *
 * Raw verbatim storage of conversation exchanges with semantic search.
 * Never lose context again.
 *
 * Provides:
 * - `memory_search` tool — semantic search across all stored memories
 * - `memory_save` tool — manually save a specific piece of information
 * - `memory_recall` tool — retrieve memories for a project/topic (L2)
 * - `memory_status` tool — show memory store overview
 * - Auto-capture of conversation exchanges on session shutdown/compact
 * - Wake-up context injection (L0 identity + L1 top memories) into system prompt
 * - Status widget showing memory count
 * - `/memory` command for quick operations
 */

import type {
  ExtensionAPI,
  ExtensionContext,
  Theme,
} from "@mariozechner/pi-coding-agent";
import { Text } from "@mariozechner/pi-tui";
import { Type } from "@sinclair/typebox";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MEMORY_DIR = path.join(
  process.env.HOME || process.env.USERPROFILE || "~",
  ".pi",
  "agent",
  "memory"
);
const CAPTURE_BUFFER_PATH = path.join(MEMORY_DIR, "capture_buffer.jsonl");
const CONFIG_PATH = path.join(MEMORY_DIR, "config.json");
const IDENTITY_PATH = path.join(MEMORY_DIR, "identity.txt");

// Resolve the scripts directory relative to this extension file
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BACKEND_SCRIPT = path.resolve(__dirname, "../../scripts/memory_backend.py");

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface MemoryConfig {
  /** Auto-capture conversation exchanges */
  autoCapture: boolean;
  /** Inject wake-up context into system prompt */
  wakeUpEnabled: boolean;
  /** Maximum tokens for wake-up context */
  wakeUpMaxTokens: number;
  /** Default project name (auto-detected from cwd if not set) */
  defaultProject: string | null;
  /** Python executable path */
  pythonPath: string;
}

interface MemoryRuntime {
  /** Current configuration */
  config: MemoryConfig;
  /** Buffered exchanges waiting to be flushed */
  bufferCount: number;
  /** Total memories in store (cached) */
  totalMemories: number;
  /** Per-project counts (cached) */
  projects: Record<string, number>;
  /** Whether the backend is available */
  backendAvailable: boolean;
  /** Cached wake-up text (refreshed on session_start) */
  wakeUpText: string | null;
  /** Current project context */
  currentProject: string;
  /** Whether memory mode is enabled */
  enabled: boolean;
}

interface BufferedExchange {
  content: string;
  project: string;
  topic: string;
  source: string;
  timestamp: string;
  session_id: string;
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

function defaultConfig(): MemoryConfig {
  return {
    autoCapture: true,
    wakeUpEnabled: true,
    wakeUpMaxTokens: 800,
    defaultProject: null,
    pythonPath: "python3",
  };
}

function createRuntime(): MemoryRuntime {
  return {
    config: defaultConfig(),
    bufferCount: 0,
    totalMemories: 0,
    projects: {},
    backendAvailable: false,
    wakeUpText: null,
    currentProject: "general",
    enabled: true,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function loadConfig(): MemoryConfig {
  const defaults = defaultConfig();
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      const raw = JSON.parse(fs.readFileSync(CONFIG_PATH, "utf-8"));
      return { ...defaults, ...raw };
    }
  } catch {
    // Use defaults
  }
  return defaults;
}

function saveConfig(config: MemoryConfig): void {
  fs.mkdirSync(MEMORY_DIR, { recursive: true });
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2));
}

function detectProject(cwd: string): string {
  // Try git repo name first
  const gitDir = path.join(cwd, ".git");
  if (fs.existsSync(gitDir)) {
    return path.basename(cwd);
  }
  // Fall back to directory name
  return path.basename(cwd) || "general";
}

function appendToBuffer(exchange: BufferedExchange): void {
  fs.mkdirSync(MEMORY_DIR, { recursive: true });
  fs.appendFileSync(
    CAPTURE_BUFFER_PATH,
    JSON.stringify(exchange) + "\n"
  );
}

function readBuffer(): BufferedExchange[] {
  if (!fs.existsSync(CAPTURE_BUFFER_PATH)) return [];
  try {
    const lines = fs.readFileSync(CAPTURE_BUFFER_PATH, "utf-8").trim().split("\n").filter(Boolean);
    return lines.map((line) => JSON.parse(line));
  } catch {
    return [];
  }
}

function clearBuffer(): void {
  try {
    if (fs.existsSync(CAPTURE_BUFFER_PATH)) {
      fs.unlinkSync(CAPTURE_BUFFER_PATH);
    }
  } catch {
    // Ignore
  }
}

/**
 * Call the Python memory backend.
 * Returns parsed JSON response or { error: "..." }.
 */
async function callBackend(
  pi: ExtensionAPI,
  pythonPath: string,
  command: string,
  args: Record<string, unknown>
): Promise<Record<string, unknown>> {
  try {
    const result = await pi.exec(pythonPath, [
      BACKEND_SCRIPT,
      command,
      JSON.stringify(args),
    ], { timeout: 30000 });

    if (result.code !== 0) {
      const errMsg = (result.stderr || result.stdout || "").trim();
      return { error: `Backend error (exit ${result.code}): ${errMsg.slice(0, 300)}` };
    }

    // Parse the last line of stdout as JSON (skip any progress/warning output)
    const lines = (result.stdout || "").trim().split("\n");
    const lastLine = lines[lines.length - 1];
    return JSON.parse(lastLine);
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return { error: `Backend call failed: ${msg}` };
  }
}

/**
 * Flush the capture buffer to ChromaDB via batch-store.
 */
async function flushBuffer(
  pi: ExtensionAPI,
  runtime: MemoryRuntime
): Promise<{ stored: number; duplicates: number }> {
  const items = readBuffer();
  if (items.length === 0) return { stored: 0, duplicates: 0 };

  const result = await callBackend(pi, runtime.config.pythonPath, "batch-store", { items });

  if (result.error) {
    return { stored: 0, duplicates: 0 };
  }

  clearBuffer();
  runtime.bufferCount = 0;

  return {
    stored: (result.stored as number) || 0,
    duplicates: (result.duplicates as number) || 0,
  };
}

/**
 * Extract text content from a message content block array.
 */
function extractTextFromContent(content: unknown): string {
  if (!Array.isArray(content)) return "";
  const textParts: string[] = [];
  for (const block of content) {
    if (typeof block === "object" && block !== null) {
      const b = block as Record<string, unknown>;
      if (b.type === "text" && typeof b.text === "string") {
        textParts.push(b.text);
      }
    }
  }
  return textParts.join("\n");
}

// ---------------------------------------------------------------------------
// Runtime store (per-session)
// ---------------------------------------------------------------------------

function createRuntimeStore() {
  const runtimes = new Map<string, MemoryRuntime>();

  return {
    ensure(sessionKey: string): MemoryRuntime {
      let runtime = runtimes.get(sessionKey);
      if (!runtime) {
        runtime = createRuntime();
        runtimes.set(sessionKey, runtime);
      }
      return runtime;
    },
    clear(sessionKey: string): void {
      runtimes.delete(sessionKey);
    },
  };
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function memoryExtension(pi: ExtensionAPI) {
  const runtimeStore = createRuntimeStore();
  const getSessionKey = (ctx: ExtensionContext) => ctx.sessionManager.getSessionId();
  const getRuntime = (ctx: ExtensionContext): MemoryRuntime =>
    runtimeStore.ensure(getSessionKey(ctx));

  // -----------------------------------------------------------------------
  // Widget
  // -----------------------------------------------------------------------

  const updateWidget = (ctx: ExtensionContext) => {
    if (!ctx.hasUI) return;

    const runtime = getRuntime(ctx);
    if (!runtime.enabled) {
      ctx.ui.setWidget("pi-memory", undefined);
      return;
    }

    ctx.ui.setWidget("pi-memory", (_tui, theme) => ({
      render(_width: number): string[] {
        const parts: string[] = [];
        const icon = theme.fg("accent", "🧠");
        const count = theme.fg("text", `${runtime.totalMemories} memories`);
        const project = theme.fg("dim", `│ ${runtime.currentProject}`);
        const buf = runtime.bufferCount > 0
          ? theme.fg("warning", ` │ ${runtime.bufferCount} buffered`)
          : "";

        parts.push(`${icon} ${count} ${project}${buf}`);
        return parts;
      },
      invalidate(): void {},
    }));
  };

  // -----------------------------------------------------------------------
  // State reconstruction
  // -----------------------------------------------------------------------

  const reconstructState = async (ctx: ExtensionContext) => {
    const runtime = getRuntime(ctx);
    runtime.config = loadConfig();
    runtime.currentProject = runtime.config.defaultProject || detectProject(ctx.cwd);
    runtime.enabled = true;

    // Count buffered items
    const buffer = readBuffer();
    runtime.bufferCount = buffer.length;

    // Check backend availability and get status
    const status = await callBackend(pi, runtime.config.pythonPath, "status", {});
    if (status.error) {
      runtime.backendAvailable = false;
      runtime.totalMemories = 0;
      runtime.projects = {};
    } else {
      runtime.backendAvailable = true;
      runtime.totalMemories = (status.total_memories as number) || 0;
      runtime.projects = (status.projects as Record<string, number>) || {};
    }

    // Pre-generate wake-up text
    if (runtime.config.wakeUpEnabled && runtime.backendAvailable) {
      const wakeup = await callBackend(pi, runtime.config.pythonPath, "wakeup", {
        project: runtime.currentProject,
        max_tokens: runtime.config.wakeUpMaxTokens,
      });
      runtime.wakeUpText = (wakeup.text as string) || null;
    }

    updateWidget(ctx);
  };

  // -----------------------------------------------------------------------
  // Lifecycle hooks
  // -----------------------------------------------------------------------

  pi.on("session_start", async (_e, ctx) => {
    await reconstructState(ctx);
  });

  pi.on("session_tree", async (_e, ctx) => {
    await reconstructState(ctx);
  });

  pi.on("session_shutdown", async (_e, ctx) => {
    // Flush buffer before shutdown
    const runtime = getRuntime(ctx);
    if (runtime.enabled && runtime.backendAvailable && runtime.bufferCount > 0) {
      await flushBuffer(pi, runtime);
    }
    runtimeStore.clear(getSessionKey(ctx));
  });

  pi.on("session_before_compact", async (_e, ctx) => {
    // Flush buffer before compaction to avoid losing memories
    const runtime = getRuntime(ctx);
    if (runtime.enabled && runtime.backendAvailable && runtime.bufferCount > 0) {
      await flushBuffer(pi, runtime);
    }
  });

  // Auto-capture: after each agent turn, extract and buffer the exchange
  pi.on("turn_end", async (event, ctx) => {
    const runtime = getRuntime(ctx);
    if (!runtime.enabled || !runtime.config.autoCapture) return;

    // We capture assistant messages and look back for the user message
    if (event.message?.role !== "assistant") return;

    const msg = event.message as unknown as Record<string, unknown>;
    const assistantText = extractTextFromContent(msg.content);
    if (!assistantText || assistantText.length < 20) return; // Skip trivial responses

    // Find the preceding user message from session history
    const branch = ctx.sessionManager.getBranch();
    let userText = "";
    for (let i = branch.length - 1; i >= 0; i--) {
      const entry = branch[i];
      if (entry.type === "message" && entry.message.role === "user") {
        const userMsg = entry.message as unknown as Record<string, unknown>;
        if (typeof userMsg.content === "string") {
          userText = userMsg.content;
        } else if (Array.isArray(userMsg.content)) {
          userText = extractTextFromContent(userMsg.content);
        }
        break;
      }
    }

    if (!userText || userText.length < 10) return; // Skip trivial inputs

    // Build exchange content (user + assistant pair)
    const exchange = `> ${userText}\n\n${assistantText}`;

    // Truncate very long exchanges (keep first 2000 chars)
    const content = exchange.length > 2000 ? exchange.slice(0, 2000) + "\n[truncated]" : exchange;

    // Buffer the exchange
    const buffered: BufferedExchange = {
      content,
      project: runtime.currentProject,
      topic: "general", // Could be auto-detected in the future
      source: "auto-capture",
      timestamp: new Date().toISOString(),
      session_id: getSessionKey(ctx),
    };

    appendToBuffer(buffered);
    runtime.bufferCount++;
    updateWidget(ctx);
  });

  // Inject wake-up context into system prompt
  pi.on("before_agent_start", async (event, ctx) => {
    const runtime = getRuntime(ctx);
    if (!runtime.enabled || !runtime.config.wakeUpEnabled) return;
    if (!runtime.wakeUpText) return;

    const extra =
      "\n\n## Agent Memory (ACTIVE)\n" +
      "You have persistent memory across sessions. Previous conversations and decisions are stored and searchable.\n" +
      "Use `memory_search` to find past context. Use `memory_save` to explicitly remember something important.\n" +
      "Use `memory_recall` to browse memories for a specific project or topic.\n\n" +
      runtime.wakeUpText;

    return {
      systemPrompt: event.systemPrompt + extra,
    };
  });

  // -----------------------------------------------------------------------
  // Tools
  // -----------------------------------------------------------------------

  // --- memory_search ---
  pi.registerTool({
    name: "memory_search",
    label: "Memory Search",
    description:
      "Search stored memories using semantic similarity. Finds past conversations, decisions, and context across all sessions.",
    promptSnippet: "memory_search(query, project?, topic?, n_results?) — semantic search across stored memories",
    promptGuidelines: [
      "Use when the user asks about past decisions, previous conversations, or 'what did we decide about X'",
      "Filter by project to narrow results to a specific codebase",
      "Returns ranked results with similarity scores — higher is more relevant",
    ],
    parameters: Type.Object({
      query: Type.String({ description: "What to search for (natural language)" }),
      project: Type.Optional(
        Type.String({ description: "Filter to a specific project" })
      ),
      topic: Type.Optional(
        Type.String({ description: "Filter to a specific topic" })
      ),
      n_results: Type.Optional(
        Type.Number({ description: "Number of results (default: 5, max: 20)" })
      ),
    }),

    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const runtime = getRuntime(ctx);
      const result = await callBackend(pi, runtime.config.pythonPath, "search", {
        query: params.query,
        project: params.project || null,
        topic: params.topic || null,
        n_results: params.n_results || 5,
      });

      if (result.error) {
        return {
          content: [{ type: "text" as const, text: `Memory search failed: ${result.error}` }],
          details: null,
        };
      }

      const hits = (result.results as Array<Record<string, unknown>>) || [];
      if (hits.length === 0) {
        return {
          content: [{ type: "text" as const, text: `No memories found for: "${params.query}"` }],
          details: null,
        };
      }

      let text = `Found ${hits.length} memories for "${params.query}":\n\n`;
      for (const hit of hits) {
        const sim = ((hit.similarity as number) * 100).toFixed(1);
        text += `[${hit.project}/${hit.topic}] (${sim}% match, ${hit.timestamp})\n`;
        text += `${hit.text}\n\n---\n\n`;
      }

      return {
        content: [{ type: "text" as const, text }],
        details: { query: params.query, hitCount: hits.length },
      };
    },

    renderResult(result, _options, theme) {
      const t = result.content[0];
      return new Text(t?.type === "text" ? (t as { text: string }).text : "", 0, 0);
    },
  });

  // --- memory_save ---
  pi.registerTool({
    name: "memory_save",
    label: "Memory Save",
    description:
      "Explicitly save a piece of information to persistent memory. Use for important decisions, facts, or context you want to remember across sessions.",
    promptSnippet: "memory_save(content, project?, topic?) — save to persistent memory",
    promptGuidelines: [
      "Use when the user says 'remember this' or when an important decision is made",
      "Include enough context in the content for it to be useful later",
      "Set project and topic for better organization and retrieval",
    ],
    parameters: Type.Object({
      content: Type.String({
        description: "The information to remember (include context)",
      }),
      project: Type.Optional(
        Type.String({ description: "Project this belongs to" })
      ),
      topic: Type.Optional(
        Type.String({ description: "Topic category (e.g., 'auth', 'database', 'architecture')" })
      ),
    }),

    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const runtime = getRuntime(ctx);
      const project = params.project || runtime.currentProject;

      const result = await callBackend(pi, runtime.config.pythonPath, "store", {
        content: params.content,
        project,
        topic: params.topic || "general",
        source: "manual-save",
        timestamp: new Date().toISOString(),
        session_id: getSessionKey(ctx),
      });

      if (result.error) {
        return {
          content: [{ type: "text" as const, text: `Failed to save memory: ${result.error}` }],
          details: null,
        };
      }

      if (result.status === "duplicate") {
        return {
          content: [{ type: "text" as const, text: `This memory already exists (${result.id}).` }],
          details: { status: "duplicate", id: result.id },
        };
      }

      // Update cached counts
      runtime.totalMemories++;
      runtime.projects[project] = (runtime.projects[project] || 0) + 1;
      updateWidget(ctx);

      return {
        content: [{
          type: "text" as const,
          text: `✅ Saved to memory (${result.id}) in ${project}/${params.topic || "general"}`,
        }],
        details: { status: "stored", id: result.id, project },
      };
    },

    renderResult(result, _options, theme) {
      const t = result.content[0];
      return new Text(t?.type === "text" ? (t as { text: string }).text : "", 0, 0);
    },
  });

  // --- memory_recall ---
  pi.registerTool({
    name: "memory_recall",
    label: "Memory Recall",
    description:
      "Browse memories for a specific project or topic. Returns recent/important memories filtered by metadata. Use for getting context about a project or topic area.",
    promptSnippet: "memory_recall(project?, topic?, n_results?) — browse memories by project/topic",
    promptGuidelines: [
      "Use when you need context about a specific project or topic area",
      "Good for 'what have we been working on in project X' type questions",
      "Complements memory_search — recall browses by metadata, search uses semantic similarity",
    ],
    parameters: Type.Object({
      project: Type.Optional(
        Type.String({ description: "Filter to a specific project" })
      ),
      topic: Type.Optional(
        Type.String({ description: "Filter to a specific topic" })
      ),
      n_results: Type.Optional(
        Type.Number({ description: "Number of results (default: 10, max: 50)" })
      ),
    }),

    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const runtime = getRuntime(ctx);

      const result = await callBackend(pi, runtime.config.pythonPath, "recall", {
        project: params.project || null,
        topic: params.topic || null,
        n_results: params.n_results || 10,
      });

      if (result.error) {
        return {
          content: [{ type: "text" as const, text: `Memory recall failed: ${result.error}` }],
          details: null,
        };
      }

      const items = (result.results as Array<Record<string, unknown>>) || [];
      if (items.length === 0) {
        const label = [params.project, params.topic].filter(Boolean).join("/") || "all";
        return {
          content: [{ type: "text" as const, text: `No memories found for: ${label}` }],
          details: null,
        };
      }

      let text = `${items.length} memories`;
      if (params.project) text += ` for project "${params.project}"`;
      if (params.topic) text += ` in topic "${params.topic}"`;
      text += ":\n\n";

      for (const item of items) {
        text += `[${item.project}/${item.topic}] (${item.timestamp})\n`;
        text += `${item.text}\n\n---\n\n`;
      }

      return {
        content: [{ type: "text" as const, text }],
        details: { count: items.length },
      };
    },

    renderResult(result, _options, theme) {
      const t = result.content[0];
      return new Text(t?.type === "text" ? (t as { text: string }).text : "", 0, 0);
    },
  });

  // --- memory_status ---
  pi.registerTool({
    name: "memory_status",
    label: "Memory Status",
    description:
      "Show the current state of the memory store: total memories, per-project counts, storage size, and configuration.",
    promptSnippet: "memory_status() — show memory store overview",
    parameters: Type.Object({}),

    async execute(_toolCallId, _params, _signal, _onUpdate, ctx) {
      const runtime = getRuntime(ctx);
      const result = await callBackend(pi, runtime.config.pythonPath, "status", {});

      if (result.error) {
        return {
          content: [{
            type: "text" as const,
            text: `Memory status unavailable: ${result.error}\n\nRun /skill:memory-setup to configure.`,
          }],
          details: null,
        };
      }

      // Update cached state
      runtime.totalMemories = (result.total_memories as number) || 0;
      runtime.projects = (result.projects as Record<string, number>) || {};
      updateWidget(ctx);

      let text = "## Memory Status\n\n";
      text += `- **Total memories**: ${result.total_memories}\n`;
      text += `- **Identity**: ${result.identity_exists ? "✅ configured" : "❌ not configured"}\n`;
      text += `- **Storage**: ${result.storage_size_kb} KB\n`;
      text += `- **Current project**: ${runtime.currentProject}\n`;
      text += `- **Auto-capture**: ${runtime.config.autoCapture ? "on" : "off"}\n`;
      text += `- **Wake-up**: ${runtime.config.wakeUpEnabled ? "on" : "off"}\n`;
      text += `- **Buffer**: ${runtime.bufferCount} exchanges pending\n\n`;

      const projects = result.projects as Record<string, number> | undefined;
      if (projects && Object.keys(projects).length > 0) {
        text += "### Projects\n";
        for (const [proj, count] of Object.entries(projects)) {
          text += `- ${proj}: ${count} memories\n`;
        }
      }

      return {
        content: [{ type: "text" as const, text }],
        details: { totalMemories: result.total_memories, projects: result.projects },
      };
    },

    renderResult(result, _options, theme) {
      const t = result.content[0];
      return new Text(t?.type === "text" ? (t as { text: string }).text : "", 0, 0);
    },
  });

  // -----------------------------------------------------------------------
  // Commands
  // -----------------------------------------------------------------------

  pi.registerCommand("memory", {
    description: "Memory management: status, search, flush, project, on/off",
    handler: async (args, ctx) => {
      const runtime = getRuntime(ctx);
      const parts = (args || "").trim().split(/\s+/);
      const subcmd = parts[0] || "status";

      switch (subcmd) {
        case "status": {
          const result = await callBackend(pi, runtime.config.pythonPath, "status", {});
          if (result.error) {
            ctx.ui.notify(`Memory error: ${result.error}`, "error");
          } else {
            ctx.ui.notify(
              `🧠 ${result.total_memories} memories | ${Object.keys(result.projects as Record<string, number> || {}).length} projects | ${result.storage_size_kb} KB`,
              "info"
            );
          }
          break;
        }

        case "flush": {
          if (!runtime.backendAvailable) {
            ctx.ui.notify("Memory backend not available", "error");
            break;
          }
          const flushed = await flushBuffer(pi, runtime);
          ctx.ui.notify(
            `Flushed: ${flushed.stored} stored, ${flushed.duplicates} duplicates`,
            "info"
          );
          // Refresh status
          const status = await callBackend(pi, runtime.config.pythonPath, "status", {});
          if (!status.error) {
            runtime.totalMemories = (status.total_memories as number) || 0;
            runtime.projects = (status.projects as Record<string, number>) || {};
          }
          updateWidget(ctx);
          break;
        }

        case "project": {
          const projectName = parts.slice(1).join(" ") || detectProject(ctx.cwd);
          runtime.currentProject = projectName;
          runtime.config.defaultProject = projectName;
          saveConfig(runtime.config);
          ctx.ui.notify(`Project set to: ${projectName}`, "info");
          updateWidget(ctx);
          break;
        }

        case "on": {
          runtime.enabled = true;
          runtime.config.autoCapture = true;
          runtime.config.wakeUpEnabled = true;
          saveConfig(runtime.config);
          ctx.ui.notify("Memory enabled", "info");
          updateWidget(ctx);
          break;
        }

        case "off": {
          runtime.enabled = false;
          runtime.config.autoCapture = false;
          runtime.config.wakeUpEnabled = false;
          saveConfig(runtime.config);
          ctx.ui.notify("Memory disabled", "info");
          updateWidget(ctx);
          break;
        }

        case "search": {
          const query = parts.slice(1).join(" ");
          if (!query) {
            ctx.ui.notify("Usage: /memory search <query>", "warning");
            break;
          }
          // Delegate to the agent by sending a user message
          pi.sendUserMessage(`Search my memory for: ${query}`);
          break;
        }

        default: {
          ctx.ui.notify(
            "Usage: /memory [status|flush|project <name>|search <query>|on|off]",
            "info"
          );
        }
      }
    },
    getArgumentCompletions: (prefix) => {
      const commands = ["status", "flush", "project", "search", "on", "off"];
      return commands
        .filter((c) => c.startsWith(prefix))
        .map((c) => ({ label: c, value: c, type: "text" as const }));
    },
  });
}
