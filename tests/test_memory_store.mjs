#!/usr/bin/env node
/**
 * Tests for the MemoryStore TypeScript backend.
 *
 * Usage:
 *   node tests/test_memory_store.mjs
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { strict as assert } from "node:assert";

// We need to import the compiled store — pi handles TS transpilation,
// but for tests we use tsx or the raw .ts via a loader.
// For now, test the module via dynamic import with tsx.

const TESTS = [];
let passed = 0;
let failed = 0;

function test(name, fn) {
  TESTS.push({ name, fn });
}

async function runTests() {
  // Dynamic import — requires tsx or node with ts loader
  let MemoryStore;
  try {
    const mod = await import("../extensions/pi-mempalace/memory_store.ts");
    MemoryStore = mod.MemoryStore;
  } catch {
    try {
      const mod = await import("../extensions/pi-mempalace/memory_store.js");
      MemoryStore = mod.MemoryStore;
    } catch (e) {
      console.error("Could not import MemoryStore. Try running with tsx:");
      console.error("  npx tsx tests/test_memory_store.mjs");
      console.error(e);
      process.exit(1);
    }
  }

  // Helper: create a fresh temp store
  function createTempStore() {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-memory-test-"));
    return { store: new MemoryStore(dir), dir };
  }

  function cleanup(dir) {
    fs.rmSync(dir, { recursive: true, force: true });
  }

  // -------------------------------------------------------------------
  // Store tests
  // -------------------------------------------------------------------

  test("store: basic store and retrieve", async () => {
    const { store, dir } = createTempStore();
    try {
      const result = await store.store({ content: "TypeScript is great" });
      assert.equal(result.status, "stored");
      assert.ok(result.id.startsWith("mem_"));
      assert.equal(store.size, 1);
    } finally {
      cleanup(dir);
    }
  });

  test("store: deduplication", async () => {
    const { store, dir } = createTempStore();
    try {
      const r1 = await store.store({ content: "duplicate content" });
      const r2 = await store.store({ content: "duplicate content" });
      assert.equal(r1.status, "stored");
      assert.equal(r2.status, "duplicate");
      assert.equal(r1.id, r2.id);
      assert.equal(store.size, 1);
    } finally {
      cleanup(dir);
    }
  });

  test("store: empty content throws", async () => {
    const { store, dir } = createTempStore();
    try {
      await assert.rejects(() => store.store({ content: "" }), /Empty content/);
      await assert.rejects(() => store.store({ content: "   " }), /Empty content/);
    } finally {
      cleanup(dir);
    }
  });

  test("store: metadata defaults", async () => {
    const { store, dir } = createTempStore();
    try {
      await store.store({ content: "test metadata" });
      const status = store.status();
      assert.equal(status.projects["general"], 1);
    } finally {
      cleanup(dir);
    }
  });

  test("store: custom metadata", async () => {
    const { store, dir } = createTempStore();
    try {
      await store.store({
        content: "auth decision",
        project: "myapp",
        topic: "auth",
        source: "manual-save",
      });
      const status = store.status();
      assert.equal(status.projects["myapp"], 1);
    } finally {
      cleanup(dir);
    }
  });

  // -------------------------------------------------------------------
  // Batch store tests
  // -------------------------------------------------------------------

  test("batchStore: multiple items", async () => {
    const { store, dir } = createTempStore();
    try {
      const result = await store.batchStore([
        { content: "item one" },
        { content: "item two" },
        { content: "item three" },
      ]);
      assert.equal(result.stored, 3);
      assert.equal(result.duplicates, 0);
      assert.equal(store.size, 3);
    } finally {
      cleanup(dir);
    }
  });

  test("batchStore: handles duplicates", async () => {
    const { store, dir } = createTempStore();
    try {
      await store.store({ content: "existing item" });
      const result = await store.batchStore([
        { content: "existing item" },
        { content: "new item" },
      ]);
      assert.equal(result.stored, 1);
      assert.equal(result.duplicates, 1);
      assert.equal(store.size, 2);
    } finally {
      cleanup(dir);
    }
  });

  test("batchStore: empty throws", async () => {
    const { store, dir } = createTempStore();
    try {
      await assert.rejects(() => store.batchStore([]), /No items/);
    } finally {
      cleanup(dir);
    }
  });

  // -------------------------------------------------------------------
  // Search tests
  // -------------------------------------------------------------------

  test("search: finds semantically similar", async () => {
    const { store, dir } = createTempStore();
    try {
      await store.store({ content: "We decided to use PostgreSQL for the database" });
      await store.store({ content: "The authentication system uses JWT tokens" });
      await store.store({ content: "Frontend is built with React and TypeScript" });

      const result = await store.search("database choice");
      assert.ok(result.results.length > 0);
      assert.ok(result.results[0].text.includes("PostgreSQL"));
      assert.ok(result.results[0].similarity > 0.3);
    } finally {
      cleanup(dir);
    }
  });

  test("search: filters by project", async () => {
    const { store, dir } = createTempStore();
    try {
      await store.store({ content: "project A stuff", project: "projA" });
      await store.store({ content: "project B stuff", project: "projB" });

      const result = await store.search("stuff", { project: "projA" });
      assert.equal(result.results.length, 1);
      assert.equal(result.results[0].project, "projA");
    } finally {
      cleanup(dir);
    }
  });

  test("search: empty query throws", async () => {
    const { store, dir } = createTempStore();
    try {
      await assert.rejects(() => store.search(""), /Empty query/);
    } finally {
      cleanup(dir);
    }
  });

  test("search: no results returns empty", async () => {
    const { store, dir } = createTempStore();
    try {
      const result = await store.search("anything");
      assert.equal(result.results.length, 0);
    } finally {
      cleanup(dir);
    }
  });

  // -------------------------------------------------------------------
  // Wakeup tests
  // -------------------------------------------------------------------

  test("wakeup: with identity and memories", async () => {
    const { store, dir } = createTempStore();
    try {
      fs.writeFileSync(path.join(dir, "identity.txt"), "I am a test assistant.");
      await store.store({ content: "important decision", project: "testproj" });

      const result = store.wakeup({ project: "testproj" });
      assert.ok(result.text.includes("I am a test assistant"));
      assert.ok(result.text.includes("important decision"));
      assert.ok(result.token_estimate > 0);
    } finally {
      cleanup(dir);
    }
  });

  test("wakeup: without identity", async () => {
    const { store, dir } = createTempStore();
    try {
      const result = store.wakeup();
      assert.ok(result.text.includes("No identity configured"));
    } finally {
      cleanup(dir);
    }
  });

  test("wakeup: empty store", async () => {
    const { store, dir } = createTempStore();
    try {
      const result = store.wakeup();
      assert.ok(result.text.includes("No memories stored yet"));
    } finally {
      cleanup(dir);
    }
  });

  // -------------------------------------------------------------------
  // Status tests
  // -------------------------------------------------------------------

  test("status: empty store", async () => {
    const { store, dir } = createTempStore();
    try {
      const result = store.status();
      assert.equal(result.total_memories, 0);
      assert.deepEqual(result.projects, {});
      assert.equal(result.identity_exists, false);
    } finally {
      cleanup(dir);
    }
  });

  test("status: with memories", async () => {
    const { store, dir } = createTempStore();
    try {
      await store.store({ content: "memory one", project: "proj1" });
      await store.store({ content: "memory two", project: "proj1" });
      await store.store({ content: "memory three", project: "proj2" });

      const result = store.status();
      assert.equal(result.total_memories, 3);
      assert.equal(result.projects["proj1"], 2);
      assert.equal(result.projects["proj2"], 1);
      assert.ok(result.storage_size_kb > 0);
    } finally {
      cleanup(dir);
    }
  });

  // -------------------------------------------------------------------
  // Delete tests
  // -------------------------------------------------------------------

  test("delete: removes memory", async () => {
    const { store, dir } = createTempStore();
    try {
      const r = await store.store({ content: "to be deleted" });
      assert.equal(store.size, 1);
      store.delete(r.id);
      assert.equal(store.size, 0);
    } finally {
      cleanup(dir);
    }
  });

  test("delete: empty id throws", async () => {
    const { store, dir } = createTempStore();
    try {
      assert.throws(() => store.delete(""), /No id/);
    } finally {
      cleanup(dir);
    }
  });

  test("delete: nonexistent id throws", async () => {
    const { store, dir } = createTempStore();
    try {
      assert.throws(() => store.delete("mem_nonexistent"), /not found/);
    } finally {
      cleanup(dir);
    }
  });

  // -------------------------------------------------------------------
  // Recall tests
  // -------------------------------------------------------------------

  test("recall: filters by project", async () => {
    const { store, dir } = createTempStore();
    try {
      await store.store({ content: "proj A memory", project: "projA" });
      await store.store({ content: "proj B memory", project: "projB" });

      const result = store.recall({ project: "projA" });
      assert.equal(result.count, 1);
      assert.equal(result.results[0].project, "projA");
    } finally {
      cleanup(dir);
    }
  });

  test("recall: filters by topic", async () => {
    const { store, dir } = createTempStore();
    try {
      await store.store({ content: "auth memory", topic: "auth" });
      await store.store({ content: "db memory", topic: "database" });

      const result = store.recall({ topic: "auth" });
      assert.equal(result.count, 1);
      assert.equal(result.results[0].topic, "auth");
    } finally {
      cleanup(dir);
    }
  });

  // -------------------------------------------------------------------
  // List projects tests
  // -------------------------------------------------------------------

  test("listProjects: returns project counts", async () => {
    const { store, dir } = createTempStore();
    try {
      await store.store({ content: "a", project: "p1" });
      await store.store({ content: "b", project: "p1" });
      await store.store({ content: "c", project: "p2" });

      const result = store.listProjects();
      assert.equal(result.projects["p1"], 2);
      assert.equal(result.projects["p2"], 1);
      assert.equal(result.total, 3);
    } finally {
      cleanup(dir);
    }
  });

  // -------------------------------------------------------------------
  // Stats tests
  // -------------------------------------------------------------------

  test("computeStats: empty store", async () => {
    const { store, dir } = createTempStore();
    try {
      const stats = store.computeStats();
      assert.equal(stats.total, 0);
      assert.deepEqual(stats.projects, {});
      assert.equal(stats.sessions, 0);
      assert.equal(stats.oldest, null);
    } finally {
      cleanup(dir);
    }
  });

  test("computeStats: with memories", async () => {
    const { store, dir } = createTempStore();
    try {
      await store.store({ content: "first memory", project: "p1", topic: "auth", source: "manual-save", session_id: "s1" });
      await store.store({ content: "second memory", project: "p1", topic: "db", source: "auto-capture", session_id: "s1" });
      await store.store({ content: "third memory", project: "p2", topic: "auth", source: "auto-capture", session_id: "s2" });

      const stats = store.computeStats();
      assert.equal(stats.total, 3);
      assert.equal(stats.projects["p1"], 2);
      assert.equal(stats.projects["p2"], 1);
      assert.equal(stats.topics["auth"], 2);
      assert.equal(stats.topics["db"], 1);
      assert.equal(stats.sources["manual-save"], 1);
      assert.equal(stats.sources["auto-capture"], 2);
      assert.equal(stats.sessions, 2);
      assert.ok(stats.avgContentLength > 0);
      assert.ok(stats.oldest);
      assert.ok(stats.newest);
      assert.ok(Object.keys(stats.timeline).length > 0);
    } finally {
      cleanup(dir);
    }
  });

  // -------------------------------------------------------------------
  // Persistence tests
  // -------------------------------------------------------------------

  test("persistence: survives reload", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-memory-test-"));
    try {
      const store1 = new MemoryStore(dir);
      await store1.store({ content: "persistent memory" });
      assert.equal(store1.size, 1);

      // Create a new store from the same directory
      const store2 = new MemoryStore(dir);
      store2.load();
      assert.equal(store2.size, 1);
    } finally {
      cleanup(dir);
    }
  });

  test("persistence: delete persists", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-memory-test-"));
    try {
      const store1 = new MemoryStore(dir);
      const r = await store1.store({ content: "to delete" });
      store1.delete(r.id);

      const store2 = new MemoryStore(dir);
      store2.load();
      assert.equal(store2.size, 0);
    } finally {
      cleanup(dir);
    }
  });

  // -------------------------------------------------------------------
  // Chunking tests
  // -------------------------------------------------------------------

  test("chunking: short content stored as single memory", async () => {
    const { store, dir } = createTempStore();
    try {
      const r = await store.store({ content: "short content" });
      assert.equal(r.status, "stored");
      assert.equal(store.size, 1);
    } finally {
      cleanup(dir);
    }
  });

  test("chunking: long content split into multiple chunks", async () => {
    const { store, dir } = createTempStore();
    try {
      // Create content > 800 chars
      const longContent = "This is a paragraph about databases. ".repeat(30) + "\n\n" +
        "This is a paragraph about authentication. ".repeat(30) + "\n\n" +
        "This is a paragraph about deployment. ".repeat(30);
      assert.ok(longContent.length > 800, "Content should exceed chunk size");

      const r = await store.store({ content: longContent, project: "chunky" });
      assert.equal(r.status, "stored");
      // Should have multiple chunks
      assert.ok(store.size > 1, `Expected multiple chunks, got ${store.size}`);
    } finally {
      cleanup(dir);
    }
  });

  test("chunking: chunks are searchable", async () => {
    const { store, dir } = createTempStore();
    try {
      const longContent = "PostgreSQL is our primary database for all transactional workloads. ".repeat(20) + "\n\n" +
        "Redis is used for caching and session storage across all services. ".repeat(20);

      await store.store({ content: longContent, project: "infra" });
      const result = await store.search("database choice");
      assert.ok(result.results.length > 0, "Should find chunks via search");
    } finally {
      cleanup(dir);
    }
  });

  // -------------------------------------------------------------------
  // Palace Graph tests
  // -------------------------------------------------------------------

  test("palace graph: empty store", async () => {
    const { store, dir } = createTempStore();
    try {
      const graph = store.getPalaceGraph();
      assert.equal(graph.nodes.length, 0);
      assert.equal(graph.edges.length, 0);
    } finally {
      cleanup(dir);
    }
  });

  test("palace graph: discovers tunnels", async () => {
    const { store, dir } = createTempStore();
    try {
      // Two projects sharing "auth" topic
      await store.store({ content: "auth in proj A", project: "projA", topic: "auth" });
      await store.store({ content: "auth in proj B", project: "projB", topic: "auth" });
      // A unique topic
      await store.store({ content: "db in proj A", project: "projA", topic: "database" });

      const tunnels = store.discoverTunnels();
      assert.equal(tunnels.length, 1);
      assert.equal(tunnels[0].topic, "auth");
      assert.ok(tunnels[0].projects.includes("projA"));
      assert.ok(tunnels[0].projects.includes("projB"));
    } finally {
      cleanup(dir);
    }
  });

  test("palace graph: traverse tunnel", async () => {
    const { store, dir } = createTempStore();
    try {
      await store.store({ content: "auth logic in A", project: "projA", topic: "auth" });
      await store.store({ content: "auth logic in B", project: "projB", topic: "auth" });

      const results = store.traverseTunnel("auth", "projA", "projB");
      assert.equal(results.length, 2);
      const projects = results.map(r => r.project).sort();
      assert.deepEqual(projects, ["projA", "projB"]);
    } finally {
      cleanup(dir);
    }
  });

  test("palace graph: full graph structure", async () => {
    const { store, dir } = createTempStore();
    try {
      await store.store({ content: "a1", project: "p1", topic: "auth" });
      await store.store({ content: "a2", project: "p2", topic: "auth" });
      await store.store({ content: "d1", project: "p1", topic: "database" });

      const graph = store.getPalaceGraph();
      assert.equal(graph.nodes.length, 2);
      assert.equal(graph.edges.length, 1);
      assert.equal(graph.edges[0].topic, "auth");
    } finally {
      cleanup(dir);
    }
  });

  // -------------------------------------------------------------------
  // Knowledge Graph tests
  // -------------------------------------------------------------------

  test("knowledge: add entity", async () => {
    const { store, dir } = createTempStore();
    try {
      const r = store.addEntity({ name: "PostgreSQL", entity_type: "technology" });
      assert.equal(r.status, "created");
      assert.ok(r.id.startsWith("ent_"));

      // Update
      const r2 = store.addEntity({ name: "PostgreSQL", entity_type: "database", id: r.id });
      assert.equal(r2.status, "updated");
    } finally {
      cleanup(dir);
    }
  });

  test("knowledge: add triple", async () => {
    const { store, dir } = createTempStore();
    try {
      const r = store.addTriple({
        subject: "myapp",
        predicate: "uses",
        object: "PostgreSQL",
        valid_from: "2025-01-01",
        project: "myapp",
      });
      assert.equal(r.status, "created");
      assert.ok(r.id > 0);
    } finally {
      cleanup(dir);
    }
  });

  test("knowledge: query entity facts", async () => {
    const { store, dir } = createTempStore();
    try {
      store.addTriple({ subject: "myapp", predicate: "uses", object: "PostgreSQL" });
      store.addTriple({ subject: "myapp", predicate: "uses", object: "Redis" });
      store.addTriple({ subject: "Alice", predicate: "created", object: "myapp" });

      const result = store.queryEntity("myapp");
      assert.ok(result.entity);
      assert.equal(result.facts.length, 3);
    } finally {
      cleanup(dir);
    }
  });

  test("knowledge: temporal query", async () => {
    const { store, dir } = createTempStore();
    try {
      store.addTriple({
        subject: "myapp", predicate: "uses", object: "MySQL",
        valid_from: "2024-01-01", valid_to: "2025-06-01",
      });
      store.addTriple({
        subject: "myapp", predicate: "uses", object: "PostgreSQL",
        valid_from: "2025-06-01",
      });

      // Query at 2024-06
      const past = store.queryEntity("myapp", { at_time: "2024-06-01" });
      assert.equal(past.facts.length, 1);
      assert.equal(past.facts[0].object, "MySQL");

      // Query at 2025-09
      const present = store.queryEntity("myapp", { at_time: "2025-09-01" });
      assert.equal(present.facts.length, 1);
      assert.equal(present.facts[0].object, "PostgreSQL");
    } finally {
      cleanup(dir);
    }
  });

  test("knowledge: query by predicate", async () => {
    const { store, dir } = createTempStore();
    try {
      store.addTriple({ subject: "app1", predicate: "uses", object: "React" });
      store.addTriple({ subject: "app2", predicate: "uses", object: "Vue" });
      store.addTriple({ subject: "app1", predicate: "depends_on", object: "Node" });

      const uses = store.queryByPredicate("uses");
      assert.equal(uses.length, 2);
    } finally {
      cleanup(dir);
    }
  });

  test("knowledge: invalidate triple", async () => {
    const { store, dir } = createTempStore();
    try {
      const r = store.addTriple({
        subject: "myapp", predicate: "uses", object: "MySQL",
        valid_from: "2024-01-01",
      });
      store.invalidateTriple(r.id, "2025-06-01");

      // Should not appear in present query
      const result = store.queryEntity("myapp", { at_time: "2025-09-01" });
      assert.equal(result.facts.length, 0);
    } finally {
      cleanup(dir);
    }
  });

  test("knowledge: stats", async () => {
    const { store, dir } = createTempStore();
    try {
      store.addTriple({ subject: "app", predicate: "uses", object: "React" });
      store.addTriple({ subject: "app", predicate: "uses", object: "Node" });

      const stats = store.knowledgeStats();
      assert.equal(stats.entityCount, 3); // app, React, Node
      assert.equal(stats.tripleCount, 2);
      assert.equal(stats.activeTriples, 2);
      assert.equal(stats.predicates["uses"], 2);
    } finally {
      cleanup(dir);
    }
  });

  // -------------------------------------------------------------------
  // Run all tests
  // -------------------------------------------------------------------

  console.log(`Running ${TESTS.length} tests...\n`);

  for (const { name, fn } of TESTS) {
    try {
      await fn();
      passed++;
      console.log(`  ✅ ${name}`);
    } catch (err) {
      failed++;
      console.log(`  ❌ ${name}`);
      console.log(`     ${err.message}`);
    }
  }

  console.log(`\n${passed} passed, ${failed} failed, ${TESTS.length} total`);
  process.exit(failed > 0 ? 1 : 0);
}

runTests();
