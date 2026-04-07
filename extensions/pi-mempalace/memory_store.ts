/**
 * memory_store.ts — Pure TypeScript memory backend.
 *
 * Replaces the Python/ChromaDB backend with:
 *   - @huggingface/transformers for local embeddings (all-MiniLM-L6-v2)
 *   - JSONL file storage with pre-computed embeddings
 *   - Brute-force cosine similarity search
 *
 * All operations are in-process — no subprocess spawning.
 */

import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MEMORY_DIR = path.join(
  process.env.HOME || process.env.USERPROFILE || "~",
  ".pi",
  "agent",
  "memory"
);
const IDENTITY_PATH = path.join(MEMORY_DIR, "identity.txt");
const MODEL_NAME = "Xenova/all-MiniLM-L6-v2";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface MemoryMetadata {
  project: string;
  topic: string;
  source: string;
  timestamp: string;
  session_id: string;
}

export interface StoredMemory {
  id: string;
  content: string;
  metadata: MemoryMetadata;
  /** Base64-encoded Float32Array of the embedding vector */
  embedding: string;
}

export interface StoreInput {
  content: string;
  project?: string;
  topic?: string;
  source?: string;
  timestamp?: string;
  session_id?: string;
}

export interface SearchResult {
  id: string;
  text: string;
  project: string;
  topic: string;
  source: string;
  timestamp: string;
  similarity: number;
}

export interface StoreResult {
  status: "stored" | "duplicate";
  id: string;
}

export interface BatchStoreResult {
  stored: number;
  duplicates: number;
  results: StoreResult[];
}

export interface StatusResult {
  memory_dir: string;
  store_path: string;
  identity_exists: boolean;
  total_memories: number;
  projects: Record<string, number>;
  storage_size_kb: number;
}

export interface WakeupResult {
  text: string;
  token_estimate: number;
}

export interface MemoryStats {
  total: number;
  projects: Record<string, number>;
  topics: Record<string, number>;
  sources: Record<string, number>;
  sessions: number;
  oldest: string | null;
  newest: string | null;
  /** Memories per day, keyed by YYYY-MM-DD */
  timeline: Record<string, number>;
  avgContentLength: number;
  storageSizeKb: number;
}

// ---------------------------------------------------------------------------
// Embeddings (lazy-loaded)
// ---------------------------------------------------------------------------

let embedder: any = null;
let embedderLoading: Promise<any> | null = null;

async function getEmbedder(): Promise<any> {
  if (embedder) return embedder;
  if (embedderLoading) return embedderLoading;

  embedderLoading = (async () => {
    const { pipeline } = await import("@huggingface/transformers");
    embedder = await pipeline("feature-extraction", MODEL_NAME, {
      dtype: "fp32" as any,
    });
    return embedder;
  })();

  return embedderLoading;
}

async function embed(text: string): Promise<Float32Array> {
  const extractor = await getEmbedder();
  const result = await extractor(text, { pooling: "mean", normalize: true });
  return new Float32Array(result.data);
}

function embeddingToBase64(vec: Float32Array): string {
  return Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength).toString(
    "base64"
  );
}

function base64ToEmbedding(b64: string): Float32Array {
  const buf = Buffer.from(b64, "base64");
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

// ---------------------------------------------------------------------------
// Cosine Similarity
// ---------------------------------------------------------------------------

/** Dot product — equivalent to cosine similarity for L2-normalized vectors. */
function dotSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

// ---------------------------------------------------------------------------
// Content Hash
// ---------------------------------------------------------------------------

function contentHash(content: string): string {
  return crypto.createHash("sha256").update(content, "utf-8").digest("hex").slice(0, 16);
}

// ---------------------------------------------------------------------------
// MemoryStore
// ---------------------------------------------------------------------------

export class MemoryStore {
  private memories: Map<string, StoredMemory> = new Map();
  private loaded = false;
  private memoryDir: string;

  constructor(memoryDir: string = MEMORY_DIR) {
    this.memoryDir = memoryDir;
  }

  get storePath(): string {
    return path.join(this.memoryDir, "memories.jsonl");
  }

  get identityPath(): string {
    return path.join(this.memoryDir, "identity.txt");
  }

  // -----------------------------------------------------------------------
  // Persistence
  // -----------------------------------------------------------------------

  /** Load all memories from JSONL into memory. */
  load(): void {
    this.memories.clear();
    const storePath = this.storePath;
    if (!fs.existsSync(storePath)) {
      this.loaded = true;
      return;
    }

    const lines = fs.readFileSync(storePath, "utf-8").trim().split("\n").filter(Boolean);
    for (const line of lines) {
      try {
        const mem: StoredMemory = JSON.parse(line);
        this.memories.set(mem.id, mem);
      } catch {
        // Skip corrupt lines
      }
    }
    this.loaded = true;
  }

  /** Ensure the store is loaded. */
  private ensureLoaded(): void {
    if (!this.loaded) this.load();
  }

  /** Append a single memory to the JSONL file. */
  private appendToDisk(mem: StoredMemory): void {
    fs.mkdirSync(this.memoryDir, { recursive: true });
    fs.appendFileSync(this.storePath, JSON.stringify(mem) + "\n");
  }

  /** Rewrite the entire JSONL file (used after delete). */
  private rewriteDisk(): void {
    fs.mkdirSync(this.memoryDir, { recursive: true });
    const lines = Array.from(this.memories.values())
      .map((m) => JSON.stringify(m))
      .join("\n");
    fs.writeFileSync(this.storePath, lines ? lines + "\n" : "");
  }

  // -----------------------------------------------------------------------
  // Commands
  // -----------------------------------------------------------------------

  async store(input: StoreInput): Promise<StoreResult> {
    this.ensureLoaded();

    const content = (input.content || "").trim();
    if (!content) {
      throw new Error("Empty content");
    }

    const cHash = contentHash(content);
    const docId = `mem_${cHash}`;

    // Check for duplicate
    if (this.memories.has(docId)) {
      return { status: "duplicate", id: docId };
    }

    const vec = await embed(content);
    const mem: StoredMemory = {
      id: docId,
      content,
      metadata: {
        project: input.project || "general",
        topic: input.topic || "general",
        source: input.source || "auto-capture",
        timestamp: input.timestamp || new Date().toISOString(),
        session_id: input.session_id || "",
      },
      embedding: embeddingToBase64(vec),
    };

    this.memories.set(docId, mem);
    this.appendToDisk(mem);

    return { status: "stored", id: docId };
  }

  async batchStore(items: StoreInput[]): Promise<BatchStoreResult> {
    if (!items || items.length === 0) {
      throw new Error("No items provided");
    }

    const results: StoreResult[] = [];
    let stored = 0;
    let duplicates = 0;

    for (const item of items) {
      if (!(item.content || "").trim()) continue;
      const result = await this.store(item);
      results.push(result);
      if (result.status === "stored") stored++;
      else duplicates++;
    }

    return { stored, duplicates, results };
  }

  async search(
    query: string,
    options?: { project?: string; topic?: string; n_results?: number }
  ): Promise<{ query: string; filters: Record<string, string | null>; results: SearchResult[] }> {
    this.ensureLoaded();

    if (!query || !query.trim()) {
      throw new Error("Empty query");
    }

    const project = options?.project || null;
    const topic = options?.topic || null;
    const nResults = Math.min(options?.n_results || 5, 20);

    const queryVec = await embed(query.trim());

    // Score all memories with optional filtering
    const scored: { mem: StoredMemory; similarity: number }[] = [];
    for (const mem of this.memories.values()) {
      if (project && mem.metadata.project !== project) continue;
      if (topic && mem.metadata.topic !== topic) continue;

      const memVec = base64ToEmbedding(mem.embedding);
      const sim = dotSimilarity(queryVec, memVec);
      scored.push({ mem, similarity: sim });
    }

    // Sort by similarity descending, take top N
    scored.sort((a, b) => b.similarity - a.similarity);
    const topN = scored.slice(0, nResults);

    const results: SearchResult[] = topN.map(({ mem, similarity }) => ({
      id: mem.id,
      text: mem.content,
      project: mem.metadata.project,
      topic: mem.metadata.topic,
      source: mem.metadata.source,
      timestamp: mem.metadata.timestamp,
      similarity: Math.round(similarity * 10000) / 10000,
    }));

    return {
      query,
      filters: { project, topic },
      results,
    };
  }

  wakeup(options?: { project?: string; max_tokens?: number }): WakeupResult {
    this.ensureLoaded();

    const project = options?.project || null;
    const maxTokens = options?.max_tokens || 800;
    const parts: string[] = [];

    // L0: Identity
    if (fs.existsSync(this.identityPath)) {
      const identity = fs.readFileSync(this.identityPath, "utf-8").trim();
      parts.push(`## Memory — Identity\n${identity}`);
    } else {
      parts.push(
        "## Memory — Identity\nNo identity configured. Use /skill:memory-setup to set up."
      );
    }

    // L1: Top memories (recent + important)
    const allMems = Array.from(this.memories.values());
    if (allMems.length > 0) {
      // Filter by project if specified
      let filtered = project
        ? allMems.filter((m) => m.metadata.project === project)
        : allMems;

      // Sort by timestamp descending
      filtered.sort((a, b) =>
        (b.metadata.timestamp || "").localeCompare(a.metadata.timestamp || "")
      );

      // Group by project
      const byProject: Record<string, StoredMemory[]> = {};
      for (const mem of filtered) {
        const proj = mem.metadata.project || "general";
        if (!byProject[proj]) byProject[proj] = [];
        byProject[proj].push(mem);
      }

      parts.push("\n## Memory — Recent Context");
      let totalChars = 0;
      const maxChars = maxTokens * 4;

      for (const [proj, entries] of Object.entries(byProject).sort()) {
        if (totalChars > maxChars) break;
        parts.push(`\n[${proj}]`);
        for (const mem of entries.slice(0, 5)) {
          let snippet = mem.content.trim().replace(/\n/g, " ");
          if (snippet.length > 200) snippet = snippet.slice(0, 197) + "...";
          const topic = mem.metadata.topic || "";
          let line =
            topic && topic !== "general"
              ? `  - [${topic}] ${snippet}`
              : `  - ${snippet}`;
          totalChars += line.length;
          if (totalChars > maxChars) {
            parts.push("  ... (use memory_search for more)");
            break;
          }
          parts.push(line);
        }
      }
    } else {
      parts.push("\n## Memory — Recent Context\nNo memories stored yet.");
    }

    const text = parts.join("\n");
    return { text, token_estimate: Math.ceil(text.length / 4) };
  }

  status(): StatusResult {
    this.ensureLoaded();
    const projects = this.countByProject();

    let storageSizeKb = 0;
    try {
      if (fs.existsSync(this.storePath)) {
        storageSizeKb = Math.round((fs.statSync(this.storePath).size / 1024) * 10) / 10;
      }
    } catch {
      // Ignore
    }

    return {
      memory_dir: this.memoryDir,
      store_path: this.storePath,
      identity_exists: fs.existsSync(this.identityPath),
      total_memories: this.memories.size,
      projects,
      storage_size_kb: storageSizeKb,
    };
  }

  delete(id: string): { status: string; id: string } {
    this.ensureLoaded();

    if (!id || !id.trim()) {
      throw new Error("No id provided");
    }

    if (!this.memories.has(id)) {
      throw new Error(`Memory not found: ${id}`);
    }

    this.memories.delete(id);
    this.rewriteDisk();
    return { status: "deleted", id };
  }

  listProjects(): { projects: Record<string, number>; total: number } {
    const { projects, total_memories: total } = this.status();
    return { projects, total };
  }

  recall(options?: {
    project?: string;
    topic?: string;
    n_results?: number;
  }): { filters: Record<string, string | null>; count: number; results: SearchResult[] } {
    this.ensureLoaded();

    const project = options?.project || null;
    const topic = options?.topic || null;
    const nResults = Math.min(options?.n_results || 10, 50);

    let filtered = Array.from(this.memories.values());
    if (project) filtered = filtered.filter((m) => m.metadata.project === project);
    if (topic) filtered = filtered.filter((m) => m.metadata.topic === topic);

    // Sort by timestamp descending
    filtered.sort((a, b) =>
      (b.metadata.timestamp || "").localeCompare(a.metadata.timestamp || "")
    );
    filtered = filtered.slice(0, nResults);

    const results: SearchResult[] = filtered.map((mem) => ({
      id: mem.id,
      text: mem.content,
      project: mem.metadata.project,
      topic: mem.metadata.topic,
      source: mem.metadata.source,
      timestamp: mem.metadata.timestamp,
      similarity: 0, // Not applicable for recall
    }));

    return { filters: { project, topic }, count: results.length, results };
  }

  // -----------------------------------------------------------------------
  // Stats
  // -----------------------------------------------------------------------

  computeStats(): MemoryStats {
    this.ensureLoaded();

    const allMems = Array.from(this.memories.values());
    const total = allMems.length;

    if (total === 0) {
      return {
        total: 0,
        projects: {},
        topics: {},
        sources: {},
        sessions: 0,
        oldest: null,
        newest: null,
        timeline: {},
        avgContentLength: 0,
        storageSizeKb: 0,
      };
    }

    const projects: Record<string, number> = {};
    const topics: Record<string, number> = {};
    const sources: Record<string, number> = {};
    const sessions = new Set<string>();
    const timeline: Record<string, number> = {};
    let totalContentLength = 0;
    let oldest = allMems[0].metadata.timestamp;
    let newest = allMems[0].metadata.timestamp;

    for (const mem of allMems) {
      const m = mem.metadata;

      // Counts
      projects[m.project || "general"] = (projects[m.project || "general"] || 0) + 1;
      topics[m.topic || "general"] = (topics[m.topic || "general"] || 0) + 1;
      sources[m.source || "unknown"] = (sources[m.source || "unknown"] || 0) + 1;

      // Sessions
      if (m.session_id) sessions.add(m.session_id);

      // Content length
      totalContentLength += mem.content.length;

      // Timeline (by day)
      if (m.timestamp) {
        const day = m.timestamp.slice(0, 10); // YYYY-MM-DD
        timeline[day] = (timeline[day] || 0) + 1;
        if (m.timestamp < oldest) oldest = m.timestamp;
        if (m.timestamp > newest) newest = m.timestamp;
      }
    }

    let storageSizeKb = 0;
    try {
      if (fs.existsSync(this.storePath)) {
        storageSizeKb = Math.round((fs.statSync(this.storePath).size / 1024) * 10) / 10;
      }
    } catch { /* ignore */ }

    return {
      total,
      projects,
      topics,
      sources,
      sessions: sessions.size,
      oldest,
      newest,
      timeline,
      avgContentLength: Math.round(totalContentLength / total),
      storageSizeKb,
    };
  }

  // -----------------------------------------------------------------------
  // Utility
  // -----------------------------------------------------------------------

  private countByProject(): Record<string, number> {
    const projects: Record<string, number> = {};
    for (const mem of this.memories.values()) {
      const proj = mem.metadata.project || "general";
      projects[proj] = (projects[proj] || 0) + 1;
    }
    return projects;
  }

  /** Get total memory count (no disk read if already loaded). */
  get size(): number {
    this.ensureLoaded();
    return this.memories.size;
  }

  /** Check if a memory exists by ID. */
  has(id: string): boolean {
    this.ensureLoaded();
    return this.memories.has(id);
  }
}
