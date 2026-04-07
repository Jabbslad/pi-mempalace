/**
 * memory_store.ts — SQLite + sqlite-vec memory backend.
 *
 * Replaces the JSONL flat-file backend with:
 *   - better-sqlite3 for persistent storage
 *   - sqlite-vec for vector similarity search
 *   - @huggingface/transformers for local embeddings (all-MiniLM-L6-v2)
 *
 * Implements the MemPalace 4-Layer Memory Stack:
 *   L0: Identity (static file)
 *   L1: Essential Story (top 15 memories, cached per session)
 *   L2: On-Demand Project Context (filtered retrieval)
 *   L3: Deep Semantic Search (sqlite-vec vector search)
 *
 * All operations are in-process — no subprocess spawning.
 */

import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";

// @ts-ignore — better-sqlite3 types may not be perfect
import Database from "better-sqlite3";
// @ts-ignore — sqlite-vec has no type declarations
import * as sqliteVec from "sqlite-vec";

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
const EMBEDDING_DIM = 384;

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
  importance?: number;
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
// Database row types
// ---------------------------------------------------------------------------

interface MemoryRow {
  rowid: number;
  id: string;
  content: string;
  content_hash: string;
  project: string;
  topic: string;
  source: string;
  timestamp: string;
  session_id: string;
  importance: number;
}

interface VecSearchRow {
  rowid: number;
  distance: number;
}

interface CountRow {
  project?: string;
  topic?: string;
  source?: string;
  session_id?: string;
  cnt: number;
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
// Content Hash
// ---------------------------------------------------------------------------

function contentHash(content: string): string {
  return crypto
    .createHash("sha256")
    .update(content, "utf-8")
    .digest("hex")
    .slice(0, 16);
}

// ---------------------------------------------------------------------------
// Distance ↔ Similarity Conversion
// ---------------------------------------------------------------------------

/**
 * Convert sqlite-vec L2 distance to cosine similarity.
 * For L2-normalized vectors: similarity = 1 - (distance² / 2)
 * sqlite-vec returns the actual L2 distance (not squared).
 */
function distanceToSimilarity(distance: number): number {
  return 1 - (distance * distance) / 2;
}

// ---------------------------------------------------------------------------
// MemoryStore
// ---------------------------------------------------------------------------

export class MemoryStore {
  private db: any = null;
  private loaded = false;
  private memoryDir: string;
  private cachedL1: string | null = null;

  // Prepared statements (initialized in load)
  private stmtInsertMemory: any = null;
  private stmtInsertVec: any = null;
  private stmtFindByHash: any = null;
  private stmtFindById: any = null;
  private stmtDeleteMemory: any = null;
  private stmtDeleteVec: any = null;
  private stmtCountAll: any = null;
  private stmtHasId: any = null;

  constructor(memoryDir: string = MEMORY_DIR) {
    this.memoryDir = memoryDir;
  }

  get dbPath(): string {
    return path.join(this.memoryDir, "memories.db");
  }

  /** Legacy JSONL path — used for migration detection */
  get storePath(): string {
    return path.join(this.memoryDir, "memories.jsonl");
  }

  get identityPath(): string {
    return path.join(this.memoryDir, "identity.txt");
  }

  // -----------------------------------------------------------------------
  // Database Lifecycle
  // -----------------------------------------------------------------------

  /** Open the database, create tables, and run migration if needed. */
  load(): void {
    if (this.loaded) return;

    fs.mkdirSync(this.memoryDir, { recursive: true });

    this.db = new Database(this.dbPath);
    sqliteVec.load(this.db);

    // WAL mode for better concurrent read performance
    this.db.pragma("journal_mode = WAL");

    // Create schema
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS memories (
        rowid INTEGER PRIMARY KEY AUTOINCREMENT,
        id TEXT NOT NULL UNIQUE,
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL UNIQUE,
        project TEXT NOT NULL DEFAULT 'general',
        topic TEXT NOT NULL DEFAULT 'general',
        source TEXT NOT NULL DEFAULT 'auto-capture',
        timestamp TEXT NOT NULL,
        session_id TEXT NOT NULL DEFAULT '',
        importance REAL DEFAULT 0.5
      );
      CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
      CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic);
      CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
      CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash);
    `);

    // sqlite-vec virtual table — created separately since CREATE VIRTUAL TABLE
    // doesn't support IF NOT EXISTS in all versions; catch the error if it exists.
    try {
      this.db.exec(
        `CREATE VIRTUAL TABLE vec_memories USING vec0(embedding float[${EMBEDDING_DIM}])`
      );
    } catch (e: any) {
      // Table already exists — that's fine
      if (!String(e.message).includes("already exists")) {
        throw e;
      }
    }

    // Prepare statements
    this.stmtInsertMemory = this.db.prepare(`
      INSERT INTO memories (id, content, content_hash, project, topic, source, timestamp, session_id, importance)
      VALUES (@id, @content, @content_hash, @project, @topic, @source, @timestamp, @session_id, @importance)
    `);

    this.stmtInsertVec = this.db.prepare(`
      INSERT INTO vec_memories (rowid, embedding) VALUES (?, ?)
    `);

    this.stmtFindByHash = this.db.prepare(
      `SELECT id FROM memories WHERE content_hash = ?`
    );

    this.stmtFindById = this.db.prepare(
      `SELECT * FROM memories WHERE id = ?`
    );

    this.stmtDeleteMemory = this.db.prepare(
      `DELETE FROM memories WHERE id = ?`
    );

    this.stmtDeleteVec = this.db.prepare(
      `DELETE FROM vec_memories WHERE rowid = ?`
    );

    this.stmtCountAll = this.db.prepare(
      `SELECT COUNT(*) as cnt FROM memories`
    );

    this.stmtHasId = this.db.prepare(
      `SELECT 1 FROM memories WHERE id = ? LIMIT 1`
    );

    // Run migration from JSONL if old file exists and DB is empty
    if (fs.existsSync(this.storePath) && this.countAll() === 0) {
      this.migrateFromJsonl();
    }

    this.loaded = true;
  }

  /** Ensure the store is loaded. */
  private ensureLoaded(): void {
    if (!this.loaded) this.load();
  }

  // -----------------------------------------------------------------------
  // Internal Helpers
  // -----------------------------------------------------------------------

  private countAll(): number {
    return (this.stmtCountAll.get() as CountRow).cnt;
  }

  /**
   * Insert a memory + its embedding in a single transaction.
   * Returns the rowid of the inserted memory.
   */
  private insertMemoryAndVec(
    id: string,
    content: string,
    cHash: string,
    project: string,
    topic: string,
    source: string,
    timestamp: string,
    sessionId: string,
    importance: number,
    embedding: Float32Array
  ): number {
    const insertBoth = this.db.transaction(() => {
      const info = this.stmtInsertMemory.run({
        id,
        content,
        content_hash: cHash,
        project,
        topic,
        source,
        timestamp,
        session_id: sessionId,
        importance,
      });
      const rowid = Number(info.lastInsertRowid);
      this.stmtInsertVec.run(BigInt(rowid), embedding);
      return rowid;
    });
    return insertBoth();
  }

  // -----------------------------------------------------------------------
  // Migration
  // -----------------------------------------------------------------------

  /** Migrate memories from legacy JSONL file to SQLite. */
  migrateFromJsonl(): void {
    const jsonlPath = this.storePath;
    if (!fs.existsSync(jsonlPath)) return;

    const lines = fs
      .readFileSync(jsonlPath, "utf-8")
      .trim()
      .split("\n")
      .filter(Boolean);

    if (lines.length === 0) return;

    const migrate = this.db.transaction(() => {
      for (const line of lines) {
        let mem: StoredMemory;
        try {
          mem = JSON.parse(line);
        } catch {
          continue; // Skip corrupt lines
        }

        const cHash = contentHash(mem.content);

        // Skip if already migrated
        if (this.stmtFindByHash.get(cHash)) continue;

        // Decode existing embedding from base64
        let embedding: Float32Array;
        try {
          embedding = base64ToEmbedding(mem.embedding);
          if (embedding.length !== EMBEDDING_DIM) continue; // Skip bad embeddings
        } catch {
          continue;
        }

        try {
          this.insertMemoryAndVec(
            mem.id,
            mem.content,
            cHash,
            mem.metadata.project || "general",
            mem.metadata.topic || "general",
            mem.metadata.source || "auto-capture",
            mem.metadata.timestamp || new Date().toISOString(),
            mem.metadata.session_id || "",
            0.5, // Default importance for migrated memories
            embedding
          );
        } catch {
          // Skip duplicates or other insertion errors
        }
      }
    });

    migrate();

    // Rename old file to backup
    const bakPath = jsonlPath + ".bak";
    try {
      fs.renameSync(jsonlPath, bakPath);
    } catch {
      // If rename fails, leave it — migration is still done
    }
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
    if (this.stmtFindByHash.get(cHash)) {
      return { status: "duplicate", id: docId };
    }

    const vec = await embed(content);

    this.insertMemoryAndVec(
      docId,
      content,
      cHash,
      input.project || "general",
      input.topic || "general",
      input.source || "auto-capture",
      input.timestamp || new Date().toISOString(),
      input.session_id || "",
      input.importance ?? 0.5,
      vec
    );

    // Invalidate L1 cache when new memory is stored
    this.cachedL1 = null;

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

  /**
   * L3: Deep Semantic Search via sqlite-vec.
   *
   * When project/topic filters are specified, performs vector search on a
   * larger candidate set and post-filters by metadata in JS.
   */
  async search(
    query: string,
    options?: { project?: string; topic?: string; n_results?: number }
  ): Promise<{
    query: string;
    filters: Record<string, string | null>;
    results: SearchResult[];
  }> {
    this.ensureLoaded();

    if (!query || !query.trim()) {
      throw new Error("Empty query");
    }

    const project = options?.project || null;
    const topic = options?.topic || null;
    const nResults = Math.min(options?.n_results || 5, 20);

    const queryVec = await embed(query.trim());

    // If filtering, search a wider set and post-filter
    const hasFilters = project || topic;
    const searchLimit = hasFilters ? Math.max(nResults * 10, 50) : nResults;

    const vecRows = this.db
      .prepare(
        `SELECT rowid, distance FROM vec_memories
         WHERE embedding MATCH ?
         ORDER BY distance
         LIMIT ?`
      )
      .all(queryVec, searchLimit) as VecSearchRow[];

    if (vecRows.length === 0) {
      return { query, filters: { project, topic }, results: [] };
    }

    // Fetch metadata for matched rowids
    const rowids = vecRows.map((r) => r.rowid);
    const distanceMap = new Map(vecRows.map((r) => [r.rowid, r.distance]));

    // Build IN clause — parameterized via individual placeholders
    const placeholders = rowids.map(() => "?").join(",");
    const memRows = this.db
      .prepare(
        `SELECT rowid, id, content, project, topic, source, timestamp
         FROM memories WHERE rowid IN (${placeholders})`
      )
      .all(...rowids) as MemoryRow[];

    // Apply post-filters and build results
    let results: SearchResult[] = [];
    for (const row of memRows) {
      if (project && row.project !== project) continue;
      if (topic && row.topic !== topic) continue;

      const distance = distanceMap.get(row.rowid) ?? Infinity;
      const similarity = distanceToSimilarity(distance);

      results.push({
        id: row.id,
        text: row.content,
        project: row.project,
        topic: row.topic,
        source: row.source,
        timestamp: row.timestamp,
        similarity: Math.round(similarity * 10000) / 10000,
      });
    }

    // Sort by similarity descending, take top N
    results.sort((a, b) => b.similarity - a.similarity);
    results = results.slice(0, nResults);

    return { query, filters: { project, topic }, results };
  }

  /**
   * Wakeup: L0 Identity + L1 Essential Story.
   *
   * L0: Read from identity.txt (always loaded, static).
   * L1: Top 15 memories by importance + recency, grouped by project.
   *     Generated once per session and cached.
   */
  wakeup(options?: { project?: string; max_tokens?: number }): WakeupResult {
    this.ensureLoaded();

    const project = options?.project || null;
    const maxTokens = options?.max_tokens || 800;
    const maxChars = maxTokens * 4;
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

    // L1: Essential Story (cached)
    if (this.cachedL1 === null) {
      this.cachedL1 = this.generateL1(project, maxChars);
    }
    parts.push(this.cachedL1);

    const text = parts.join("\n");
    return { text, token_estimate: Math.ceil(text.length / 4) };
  }

  /**
   * Generate L1 Essential Story: top 15 memories by importance + recency,
   * grouped by project with compact formatting.
   */
  private generateL1(project: string | null, maxChars: number): string {
    const total = this.countAll();
    if (total === 0) {
      return "\n## Memory — Recent Context\nNo memories stored yet.";
    }

    let rows: MemoryRow[];
    if (project) {
      rows = this.db
        .prepare(
          `SELECT content, project, topic, timestamp, importance
           FROM memories WHERE project = ?
           ORDER BY importance DESC, timestamp DESC
           LIMIT 15`
        )
        .all(project) as MemoryRow[];
    } else {
      rows = this.db
        .prepare(
          `SELECT content, project, topic, timestamp, importance
           FROM memories
           ORDER BY importance DESC, timestamp DESC
           LIMIT 15`
        )
        .all() as MemoryRow[];
    }

    if (rows.length === 0) {
      return "\n## Memory — Recent Context\nNo memories stored yet.";
    }

    // Group by project
    const byProject: Record<string, MemoryRow[]> = {};
    for (const row of rows) {
      const proj = row.project || "general";
      if (!byProject[proj]) byProject[proj] = [];
      byProject[proj].push(row);
    }

    const lines: string[] = ["\n## Memory — Recent Context"];
    let totalChars = 0;

    for (const [proj, entries] of Object.entries(byProject).sort()) {
      if (totalChars > maxChars) break;
      lines.push(`\n[${proj}]`);
      for (const row of entries.slice(0, 5)) {
        let snippet = row.content.trim().replace(/\n/g, " ");
        if (snippet.length > 200) snippet = snippet.slice(0, 197) + "...";
        const topic = row.topic || "";
        const line =
          topic && topic !== "general"
            ? `  - [${topic}] ${snippet}`
            : `  - ${snippet}`;
        totalChars += line.length;
        if (totalChars > maxChars) {
          lines.push("  ... (use memory_search for more)");
          break;
        }
        lines.push(line);
      }
    }

    return lines.join("\n");
  }

  status(): StatusResult {
    this.ensureLoaded();

    const projects = this.countByProject();
    const total = this.countAll();

    let storageSizeKb = 0;
    try {
      if (fs.existsSync(this.dbPath)) {
        storageSizeKb =
          Math.round((fs.statSync(this.dbPath).size / 1024) * 10) / 10;
      }
    } catch {
      // Ignore
    }

    return {
      memory_dir: this.memoryDir,
      store_path: this.dbPath,
      identity_exists: fs.existsSync(this.identityPath),
      total_memories: total,
      projects,
      storage_size_kb: storageSizeKb,
    };
  }

  delete(id: string): { status: string; id: string } {
    this.ensureLoaded();

    if (!id || !id.trim()) {
      throw new Error("No id provided");
    }

    const row = this.stmtFindById.get(id) as MemoryRow | undefined;
    if (!row) {
      throw new Error(`Memory not found: ${id}`);
    }

    // Delete from both tables in a transaction
    const deleteTransaction = this.db.transaction(() => {
      this.stmtDeleteVec.run(BigInt(row.rowid));
      this.stmtDeleteMemory.run(id);
    });
    deleteTransaction();

    // Invalidate L1 cache
    this.cachedL1 = null;

    return { status: "deleted", id };
  }

  listProjects(): { projects: Record<string, number>; total: number } {
    const { projects, total_memories: total } = this.status();
    return { projects, total };
  }

  /**
   * L2: On-Demand Project Context.
   * Filtered retrieval by project/topic, ordered by timestamp descending.
   */
  recall(options?: {
    project?: string;
    topic?: string;
    n_results?: number;
  }): {
    filters: Record<string, string | null>;
    count: number;
    results: SearchResult[];
  } {
    this.ensureLoaded();

    const project = options?.project || null;
    const topic = options?.topic || null;
    const nResults = Math.min(options?.n_results || 10, 50);

    // Build dynamic query
    const conditions: string[] = [];
    const params: any[] = [];

    if (project) {
      conditions.push("project = ?");
      params.push(project);
    }
    if (topic) {
      conditions.push("topic = ?");
      params.push(topic);
    }

    const whereClause =
      conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

    const rows = this.db
      .prepare(
        `SELECT id, content, project, topic, source, timestamp
         FROM memories ${whereClause}
         ORDER BY timestamp DESC
         LIMIT ?`
      )
      .all(...params, nResults) as MemoryRow[];

    const results: SearchResult[] = rows.map((row) => ({
      id: row.id,
      text: row.content,
      project: row.project,
      topic: row.topic,
      source: row.source,
      timestamp: row.timestamp,
      similarity: 0, // Not applicable for recall
    }));

    return { filters: { project, topic }, count: results.length, results };
  }

  // -----------------------------------------------------------------------
  // Stats
  // -----------------------------------------------------------------------

  computeStats(): MemoryStats {
    this.ensureLoaded();

    const total = this.countAll();

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

    // Project counts
    const projects: Record<string, number> = {};
    const projectRows = this.db
      .prepare(
        `SELECT project, COUNT(*) as cnt FROM memories GROUP BY project`
      )
      .all() as CountRow[];
    for (const r of projectRows) {
      projects[r.project || "general"] = r.cnt;
    }

    // Topic counts
    const topics: Record<string, number> = {};
    const topicRows = this.db
      .prepare(`SELECT topic, COUNT(*) as cnt FROM memories GROUP BY topic`)
      .all() as CountRow[];
    for (const r of topicRows) {
      topics[r.topic || "general"] = r.cnt;
    }

    // Source counts
    const sources: Record<string, number> = {};
    const sourceRows = this.db
      .prepare(`SELECT source, COUNT(*) as cnt FROM memories GROUP BY source`)
      .all() as CountRow[];
    for (const r of sourceRows) {
      sources[r.source || "unknown"] = r.cnt;
    }

    // Session count
    const sessionCount = (
      this.db
        .prepare(
          `SELECT COUNT(DISTINCT session_id) as cnt FROM memories WHERE session_id != ''`
        )
        .get() as CountRow
    ).cnt;

    // Oldest/newest timestamps
    const oldest = (
      this.db.prepare(`SELECT MIN(timestamp) as val FROM memories`).get() as {
        val: string | null;
      }
    ).val;
    const newest = (
      this.db.prepare(`SELECT MAX(timestamp) as val FROM memories`).get() as {
        val: string | null;
      }
    ).val;

    // Timeline: memories per day
    const timeline: Record<string, number> = {};
    const timelineRows = this.db
      .prepare(
        `SELECT SUBSTR(timestamp, 1, 10) as day, COUNT(*) as cnt
         FROM memories GROUP BY day ORDER BY day`
      )
      .all() as { day: string; cnt: number }[];
    for (const r of timelineRows) {
      timeline[r.day] = r.cnt;
    }

    // Average content length
    const avgLen = (
      this.db
        .prepare(`SELECT AVG(LENGTH(content)) as val FROM memories`)
        .get() as { val: number }
    ).val;

    // Storage size
    let storageSizeKb = 0;
    try {
      if (fs.existsSync(this.dbPath)) {
        storageSizeKb =
          Math.round((fs.statSync(this.dbPath).size / 1024) * 10) / 10;
      }
    } catch {
      /* ignore */
    }

    return {
      total,
      projects,
      topics,
      sources,
      sessions: sessionCount,
      oldest,
      newest,
      timeline,
      avgContentLength: Math.round(avgLen || 0),
      storageSizeKb,
    };
  }

  // -----------------------------------------------------------------------
  // Utility
  // -----------------------------------------------------------------------

  private countByProject(): Record<string, number> {
    const projects: Record<string, number> = {};
    const rows = this.db
      .prepare(
        `SELECT project, COUNT(*) as cnt FROM memories GROUP BY project`
      )
      .all() as CountRow[];
    for (const r of rows) {
      projects[r.project || "general"] = r.cnt;
    }
    return projects;
  }

  /** Get total memory count. */
  get size(): number {
    this.ensureLoaded();
    return this.countAll();
  }

  /** Check if a memory exists by ID. */
  has(id: string): boolean {
    this.ensureLoaded();
    return !!this.stmtHasId.get(id);
  }
}
