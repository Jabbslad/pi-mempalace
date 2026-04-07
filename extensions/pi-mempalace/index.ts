/**
 * pi-mempalace — Persistent Agent Memory Extension
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
} from "@mariozechner/pi-coding-agent";
import { DynamicBorder } from "@mariozechner/pi-coding-agent";
import { Container, Spacer, Text } from "@mariozechner/pi-tui";
import type { MemoryStats } from "./memory_store.js";
import { Type } from "@sinclair/typebox";
import * as fs from "node:fs";
import * as path from "node:path";

import { MemoryStore } from "./memory_store.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MEMORY_DIR = path.join(
  process.env.HOME || process.env.USERPROFILE || "~",
  ".pi",
  "agent",
  "memory"
);
const CONFIG_PATH = path.join(MEMORY_DIR, "config.json");

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
}

interface MemoryRuntime {
  /** Current configuration */
  config: MemoryConfig;
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
  /** The memory store instance */
  store: MemoryStore;
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
  };
}

function createRuntime(): MemoryRuntime {
  return {
    config: defaultConfig(),
    totalMemories: 0,
    projects: {},
    backendAvailable: false,
    wakeUpText: null,
    currentProject: "general",
    enabled: true,
    store: new MemoryStore(),
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
  const gitDir = path.join(cwd, ".git");
  if (fs.existsSync(gitDir)) {
    return path.basename(cwd);
  }
  return path.basename(cwd) || "general";
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
// Shared tool helpers
// ---------------------------------------------------------------------------

function textResult(text: string, details: Record<string, unknown> | null = null) {
  return {
    content: [{ type: "text" as const, text }],
    details,
  };
}

function renderTextResult(result: any) {
  const t = result.content[0];
  return new Text(t?.type === "text" ? t.text : "", 0, 0);
}

// ---------------------------------------------------------------------------
// Stats overlay
// ---------------------------------------------------------------------------

function barChart(
  items: [string, number][],
  maxBarWidth: number,
  theme: { fg: (color: string, text: string) => string },
): string[] {
  if (items.length === 0) return ["  (none)"];
  const maxVal = Math.max(...items.map(([, v]) => v));
  const maxLabel = Math.max(...items.map(([k]) => k.length));
  return items.map(([label, count]) => {
    const barLen = maxVal > 0 ? Math.round((count / maxVal) * maxBarWidth) : 0;
    const bar = theme.fg("accent", "█".repeat(barLen)) + "░".repeat(maxBarWidth - barLen);
    const paddedLabel = label.padEnd(maxLabel);
    return `  ${theme.fg("text", paddedLabel)} ${bar} ${theme.fg("dim", String(count))}`;
  });
}

function sparkline(timeline: Record<string, number>, days: number): string {
  const sparks = " ▁▂▃▄▅▆▇█";
  const now = new Date();
  const values: number[] = [];
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(now);
    d.setDate(d.getDate() - i);
    const key = d.toISOString().slice(0, 10);
    values.push(timeline[key] || 0);
  }
  const max = Math.max(...values, 1);
  return values.map((v) => sparks[Math.round((v / max) * (sparks.length - 1))]).join("");
}

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

function daysBetween(a: string, b: string): number {
  return Math.round((new Date(b).getTime() - new Date(a).getTime()) / 86400000);
}

async function showStatsOverlay(
  ctx: ExtensionContext,
  stats: MemoryStats,
): Promise<void> {
  await ctx.ui.custom<void>((tui, theme, _kb, done) => {
    const container = new Container();

    container.addChild(new DynamicBorder((s: string) => theme.fg("accent", s)));
    container.addChild(new Text(theme.fg("accent", "  🧠 Memory Stats"), 0, 0));
    container.addChild(new DynamicBorder((s: string) => theme.fg("border", s)));

    if (stats.total === 0) {
      container.addChild(new Text(theme.fg("dim", "  No memories stored yet."), 0, 0));
    } else {
      // Overview
      const span = stats.oldest && stats.newest
        ? `${daysBetween(stats.oldest, stats.newest)}d span`
        : "";
      const lines = [
        `  ${theme.fg("text", "Total")}         ${theme.fg("accent", String(stats.total))} memories`,
        `  ${theme.fg("text", "Storage")}       ${theme.fg("accent", `${stats.storageSizeKb} KB`)}`,
        `  ${theme.fg("text", "Sessions")}      ${theme.fg("accent", String(stats.sessions))}`,
        `  ${theme.fg("text", "Avg length")}    ${theme.fg("accent", `${stats.avgContentLength}`)} chars`,
        `  ${theme.fg("text", "First memory")}  ${theme.fg("dim", formatDate(stats.oldest))}`,
        `  ${theme.fg("text", "Last memory")}   ${theme.fg("dim", formatDate(stats.newest))}${span ? `  (${theme.fg("dim", span)})` : ""}`,
      ];
      container.addChild(new Text(lines.join("\n"), 0, 0));

      // Activity sparkline (last 28 days)
      container.addChild(new Spacer(1));
      const spark = sparkline(stats.timeline, 28);
      container.addChild(new Text(
        `  ${theme.fg("text", "Activity (28d)")}  ${theme.fg("accent", spark)}`,
        0, 0,
      ));

      // Projects bar chart
      const projectEntries = Object.entries(stats.projects)
        .sort(([, a], [, b]) => b - a);
      if (projectEntries.length > 0) {
        container.addChild(new Spacer(1));
        container.addChild(new Text(theme.fg("text", "  Projects"), 0, 0));
        const projectBars = barChart(projectEntries.slice(0, 8), 20, theme);
        container.addChild(new Text(projectBars.join("\n"), 0, 0));
      }

      // Topics bar chart
      const topicEntries = Object.entries(stats.topics)
        .filter(([t]) => t !== "general")
        .sort(([, a], [, b]) => b - a);
      if (topicEntries.length > 0) {
        container.addChild(new Spacer(1));
        container.addChild(new Text(theme.fg("text", "  Topics"), 0, 0));
        const topicBars = barChart(topicEntries.slice(0, 8), 20, theme);
        container.addChild(new Text(topicBars.join("\n"), 0, 0));
      }

      // Sources breakdown
      const sourceEntries = Object.entries(stats.sources)
        .sort(([, a], [, b]) => b - a);
      if (sourceEntries.length > 0) {
        container.addChild(new Spacer(1));
        container.addChild(new Text(theme.fg("text", "  Sources"), 0, 0));
        const sourceBars = barChart(sourceEntries, 20, theme);
        container.addChild(new Text(sourceBars.join("\n"), 0, 0));
      }
    }

    container.addChild(new DynamicBorder((s: string) => theme.fg("border", s)));
    container.addChild(new Text(theme.fg("dim", "  press any key to close"), 0, 0));
    container.addChild(new DynamicBorder((s: string) => theme.fg("accent", s)));

    return {
      render: (w: number) => container.render(w),
      invalidate: () => container.invalidate(),
      handleInput: () => done(undefined),
    };
  });
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
  // State reconstruction
  // -----------------------------------------------------------------------

  const reconstructState = async (ctx: ExtensionContext) => {
    const runtime = getRuntime(ctx);
    runtime.config = loadConfig();
    runtime.currentProject = runtime.config.defaultProject || detectProject(ctx.cwd);
    runtime.enabled = true;

    // Load the memory store from disk
    try {
      runtime.store.load();
      runtime.backendAvailable = true;
      const status = runtime.store.status();
      runtime.totalMemories = status.total_memories;
      runtime.projects = status.projects;
    } catch {
      runtime.backendAvailable = false;
      runtime.totalMemories = 0;
      runtime.projects = {};
    }

    // Pre-generate wake-up text (no embedding needed — just reads from memory)
    if (runtime.config.wakeUpEnabled && runtime.backendAvailable) {
      try {
        const wakeup = runtime.store.wakeup({
          project: runtime.currentProject,
          max_tokens: runtime.config.wakeUpMaxTokens,
        });
        runtime.wakeUpText = wakeup.text || null;
      } catch {
        runtime.wakeUpText = null;
      }
    }

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
    runtimeStore.clear(getSessionKey(ctx));
  });

  // Auto-capture: after each agent turn, extract and store the exchange
  pi.on("turn_end", async (event, ctx) => {
    const runtime = getRuntime(ctx);
    if (!runtime.enabled || !runtime.config.autoCapture) return;
    if (!runtime.backendAvailable) return;

    if (event.message?.role !== "assistant") return;

    const msg = event.message as unknown as Record<string, unknown>;
    const assistantText = extractTextFromContent(msg.content);
    if (!assistantText || assistantText.length < 20) return;

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

    if (!userText || userText.length < 10) return;

    // Build exchange content
    const exchange = `> ${userText}\n\n${assistantText}`;
    const content = exchange.length > 2000
      ? exchange.slice(0, 2000) + "\n[truncated]"
      : exchange;

    try {
      const result = await runtime.store.store({
        content,
        project: runtime.currentProject,
        topic: "general",
        source: "auto-capture",
        timestamp: new Date().toISOString(),
        session_id: getSessionKey(ctx),
      });

      if (result.status === "stored") {
        runtime.totalMemories++;
        runtime.projects[runtime.currentProject] =
          (runtime.projects[runtime.currentProject] || 0) + 1;
      }
    } catch {
      // Silently fail — don't interrupt the session
    }
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

      try {
        const result = await runtime.store.search(params.query, {
          project: params.project,
          topic: params.topic,
          n_results: params.n_results,
        });

        if (result.results.length === 0) {
          return textResult(`No memories found for: "${params.query}"`);
        }

        let text = `Found ${result.results.length} memories for "${params.query}":\n\n`;
        for (const hit of result.results) {
          const sim = (hit.similarity * 100).toFixed(1);
          text += `[${hit.project}/${hit.topic}] (${sim}% match, ${hit.timestamp})\n`;
          text += `${hit.text}\n\n---\n\n`;
        }

        return textResult(text, { query: params.query, hitCount: result.results.length });
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        return textResult(`Memory search failed: ${msg}`);
      }
    },

    renderResult: renderTextResult,
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

      try {
        const result = await runtime.store.store({
          content: params.content,
          project,
          topic: params.topic || "general",
          source: "manual-save",
          timestamp: new Date().toISOString(),
          session_id: getSessionKey(ctx),
        });

        if (result.status === "duplicate") {
          return textResult(`This memory already exists (${result.id}).`, { status: "duplicate", id: result.id });
        }

        // Update cached counts
        runtime.totalMemories++;
        runtime.projects[project] = (runtime.projects[project] || 0) + 1;

        return textResult(
          `✅ Saved to memory (${result.id}) in ${project}/${params.topic || "general"}`,
          { status: "stored", id: result.id, project },
        );
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        return textResult(`Failed to save memory: ${msg}`);
      }
    },

    renderResult: renderTextResult,
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

      try {
        const result = runtime.store.recall({
          project: params.project,
          topic: params.topic,
          n_results: params.n_results,
        });

        if (result.results.length === 0) {
          const label = [params.project, params.topic].filter(Boolean).join("/") || "all";
          return textResult(`No memories found for: ${label}`);
        }

        let text = `${result.results.length} memories`;
        if (params.project) text += ` for project "${params.project}"`;
        if (params.topic) text += ` in topic "${params.topic}"`;
        text += ":\n\n";

        for (const item of result.results) {
          text += `[${item.project}/${item.topic}] (${item.timestamp})\n`;
          text += `${item.text}\n\n---\n\n`;
        }

        return textResult(text, { count: result.results.length });
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        return textResult(`Memory recall failed: ${msg}`);
      }
    },

    renderResult: renderTextResult,
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

      try {
        const result = runtime.store.status();

        // Update cached state
        runtime.totalMemories = result.total_memories;
        runtime.projects = result.projects;

        let text = "## Memory Status\n\n";
        text += `- **Total memories**: ${result.total_memories}\n`;
        text += `- **Identity**: ${result.identity_exists ? "✅ configured" : "❌ not configured"}\n`;
        text += `- **Storage**: ${result.storage_size_kb} KB\n`;
        text += `- **Current project**: ${runtime.currentProject}\n`;
        text += `- **Auto-capture**: ${runtime.config.autoCapture ? "on" : "off"}\n`;
        text += `- **Wake-up**: ${runtime.config.wakeUpEnabled ? "on" : "off"}\n`;
        text += `- **Backend**: pure TypeScript (in-process)\n\n`;

        if (result.projects && Object.keys(result.projects).length > 0) {
          text += "### Projects\n";
          for (const [proj, count] of Object.entries(result.projects)) {
            text += `- ${proj}: ${count} memories\n`;
          }
        }

        return textResult(text, { totalMemories: result.total_memories, projects: result.projects });
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        return textResult(`Memory status unavailable: ${msg}\n\nRun /skill:memory-setup to configure.`);
      }
    },

    renderResult: renderTextResult,
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
          try {
            const result = runtime.store.status();
            ctx.ui.notify(
              `🧠 ${result.total_memories} memories | ${Object.keys(result.projects).length} projects | ${result.storage_size_kb} KB`,
              "info"
            );
          } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : String(e);
            ctx.ui.notify(`Memory error: ${msg}`, "error");
          }
          break;
        }

        case "project": {
          const projectName = parts.slice(1).join(" ") || detectProject(ctx.cwd);
          runtime.currentProject = projectName;
          runtime.config.defaultProject = projectName;
          saveConfig(runtime.config);
          ctx.ui.notify(`Project set to: ${projectName}`, "info");
          break;
        }

        case "on": {
          runtime.enabled = true;
          runtime.config.autoCapture = true;
          runtime.config.wakeUpEnabled = true;
          saveConfig(runtime.config);
          ctx.ui.notify("Memory enabled", "info");
          break;
        }

        case "off": {
          runtime.enabled = false;
          runtime.config.autoCapture = false;
          runtime.config.wakeUpEnabled = false;
          saveConfig(runtime.config);
          ctx.ui.notify("Memory disabled", "info");
          break;
        }

        case "stats": {
          if (!runtime.backendAvailable) {
            ctx.ui.notify("Memory backend not available", "error");
            break;
          }
          try {
            const stats = runtime.store.computeStats();
            await showStatsOverlay(ctx, stats);
          } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : String(e);
            ctx.ui.notify(`Stats error: ${msg}`, "error");
          }
          break;
        }

        case "search": {
          const query = parts.slice(1).join(" ");
          if (!query) {
            ctx.ui.notify("Usage: /memory search <query>", "warning");
            break;
          }
          pi.sendUserMessage(`Search my memory for: ${query}`);
          break;
        }

        default: {
          ctx.ui.notify(
            "Usage: /memory [status|stats|project <name>|search <query>|on|off]",
            "info"
          );
        }
      }
    },
    getArgumentCompletions: (prefix) => {
      const commands = ["status", "stats", "project", "search", "on", "off"];
      return commands
        .filter((c) => c.startsWith(prefix))
        .map((c) => ({ label: c, value: c, type: "text" as const }));
    },
  });
}
