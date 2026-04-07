#!/usr/bin/env node
/**
 * LongMemEval Benchmark for pi-mempalace
 *
 * Reproduces MemPalace's raw-mode benchmark:
 *   - Store each session's user turns as a single memory (verbatim)
 *   - Query with the question text
 *   - Check if any gold session appears in top-K results
 *   - Compute Recall@5, Recall@10, NDCG@10
 *
 * Usage:
 *   npx tsx benchmarks/longmemeval_bench.mjs [options]
 *
 * Options:
 *   --limit N        Only run first N questions (default: all 500)
 *   --top-k K        Max results to retrieve (default: 50)
 *   --data PATH      Path to longmemeval_s_cleaned.json
 *   --out PATH       Write per-question results to JSONL file
 *   --quiet          Suppress per-question output
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

// ---------------------------------------------------------------------------
// CLI Args
// ---------------------------------------------------------------------------

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    limit: 0,
    topK: 50,
    data: path.join(import.meta.dirname, "data", "longmemeval_s_cleaned.json"),
    out: "",
    quiet: false,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--limit":
        opts.limit = parseInt(args[++i], 10);
        break;
      case "--top-k":
        opts.topK = parseInt(args[++i], 10);
        break;
      case "--data":
        opts.data = args[++i];
        break;
      case "--out":
        opts.out = args[++i];
        break;
      case "--quiet":
        opts.quiet = true;
        break;
    }
  }
  return opts;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/** Recall@K: is any gold ID in the top-K retrieved IDs? */
function recallAtK(retrievedIds, goldIds, k) {
  const topK = new Set(retrievedIds.slice(0, k));
  return goldIds.some((id) => topK.has(id)) ? 1 : 0;
}

/** DCG@K: Discounted Cumulative Gain */
function dcgAtK(relevances, k) {
  let dcg = 0;
  for (let i = 0; i < Math.min(relevances.length, k); i++) {
    dcg += relevances[i] / Math.log2(i + 2);
  }
  return dcg;
}

/** NDCG@K: Normalized DCG */
function ndcgAtK(retrievedIds, goldIds, k) {
  const goldSet = new Set(goldIds);

  // Actual relevances in retrieved order
  const relevances = retrievedIds
    .slice(0, k)
    .map((id) => (goldSet.has(id) ? 1 : 0));

  // Ideal relevances (all gold first)
  const idealRelevances = Array(Math.min(goldIds.length, k))
    .fill(1)
    .concat(Array(Math.max(0, k - goldIds.length)).fill(0));

  const dcg = dcgAtK(relevances, k);
  const idcg = dcgAtK(idealRelevances, k);

  return idcg === 0 ? 0 : dcg / idcg;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const opts = parseArgs();

  // Load dataset
  if (!fs.existsSync(opts.data)) {
    console.error(`Dataset not found: ${opts.data}`);
    console.error(
      "Download with: curl -fsSL -o benchmarks/data/longmemeval_s_cleaned.json \\\n" +
        "  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
    );
    process.exit(1);
  }

  console.log("Loading dataset...");
  const dataset = JSON.parse(fs.readFileSync(opts.data, "utf-8"));
  const questions =
    opts.limit > 0 ? dataset.slice(0, opts.limit) : dataset;
  console.log(
    `Loaded ${dataset.length} questions, running ${questions.length}\n`
  );

  // Import MemoryStore
  const { MemoryStore } = await import(
    "../extensions/pi-mempalace/memory_store.ts"
  );

  // Tracking
  let totalR5 = 0,
    totalR10 = 0,
    totalNDCG10 = 0;
  const perType = {};
  const failures = [];
  const t0 = performance.now();
  let outStream = null;

  if (opts.out) {
    outStream = fs.createWriteStream(opts.out);
  }

  for (let qi = 0; qi < questions.length; qi++) {
    const q = questions[qi];
    const qType = q.question_type;

    // Create a fresh store per question (each question has its own haystack)
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "lme-bench-"));
    const store = new MemoryStore(tmpDir);
    store.load();

    // Store each session as a single memory (user turns only, concatenated)
    // This matches MemPalace's raw mode: one document per session.
    // Content may be chunked (>800 chars), so we map all chunk IDs
    // back to the original session ID.
    const memToSession = new Map(); // memory_id -> session_id

    for (let si = 0; si < q.haystack_sessions.length; si++) {
      const session = q.haystack_sessions[si];
      const sessionId = q.haystack_session_ids[si];
      const userTurns = session
        .filter((t) => t.role === "user")
        .map((t) => t.content)
        .join("\n");

      if (!userTurns || userTurns.length < 10) continue;

      const sizeBefore = store.size;
      const result = await store.store({
        content: userTurns,
        project: "longmemeval",
        topic: qType,
        source: "benchmark",
        timestamp: q.haystack_dates?.[si] || new Date().toISOString(),
      });

      // Map all newly created IDs (including chunks) to this session
      if (result.status === "stored") {
        // The returned id is the parent; chunks share the base hash.
        // Map both the parent and any chunk variants.
        const baseId = result.id;
        memToSession.set(baseId, sessionId);

        // If chunks were created, map them too (mem_HASH_c0, _c1, ...)
        if (baseId.includes("_c")) {
          const baseHash = baseId.replace(/_c\d+$/, "");
          // Map all chunk IDs that could exist
          const chunksCreated = store.size - sizeBefore;
          for (let ci = 0; ci < chunksCreated; ci++) {
            memToSession.set(`${baseHash}_c${ci}`, sessionId);
          }
        }
      }
    }

    // Search with wider k to account for multiple chunks per session
    const searchResult = await store.search(q.question, {
      n_results: opts.topK,
    });

    // Map memory IDs back to session IDs, dedup while preserving order
    const seen = new Set();
    const retrievedSessionIds = [];
    for (const r of searchResult.results) {
      const sid = memToSession.get(r.id) || r.id;
      if (!seen.has(sid)) {
        seen.add(sid);
        retrievedSessionIds.push(sid);
      }
    }
    const goldIds = q.answer_session_ids;

    // Score
    const r5 = recallAtK(retrievedSessionIds, goldIds, 5);
    const r10 = recallAtK(retrievedSessionIds, goldIds, 10);
    const ndcg10 = ndcgAtK(retrievedSessionIds, goldIds, 10);

    totalR5 += r5;
    totalR10 += r10;
    totalNDCG10 += ndcg10;

    // Per-type tracking
    if (!perType[qType]) perType[qType] = { count: 0, r5: 0, r10: 0 };
    perType[qType].count++;
    perType[qType].r5 += r5;
    perType[qType].r10 += r10;

    if (r5 === 0) {
      failures.push({
        qi,
        question_id: q.question_id,
        question_type: qType,
        question: q.question.slice(0, 100),
        gold: goldIds,
        retrieved: retrievedSessionIds.slice(0, 5),
      });
    }

    // Per-question output
    if (!opts.quiet) {
      const status = r5 ? "✅" : "❌";
      const progress = `[${qi + 1}/${questions.length}]`;
      console.log(
        `${progress} ${status} R@5=${r5} R@10=${r10} NDCG@10=${ndcg10.toFixed(3)} | ${qType} | ${q.question.slice(0, 60)}...`
      );
    }

    // Write per-question JSONL
    if (outStream) {
      outStream.write(
        JSON.stringify({
          question_id: q.question_id,
          question_type: qType,
          r5,
          r10,
          ndcg10: Math.round(ndcg10 * 1000) / 1000,
          gold_ids: goldIds,
          retrieved_top5: retrievedSessionIds.slice(0, 5),
        }) + "\n"
      );
    }

    // Cleanup temp store
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }

  const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

  if (outStream) outStream.end();

  // ---------------------------------------------------------------------------
  // Results
  // ---------------------------------------------------------------------------

  const n = questions.length;
  console.log("\n" + "=".repeat(70));
  console.log("  LongMemEval Benchmark Results — pi-mempalace (raw mode)");
  console.log("=".repeat(70));
  console.log();
  console.log(`  Questions:  ${n}`);
  console.log(`  Time:       ${elapsed}s (${(parseFloat(elapsed) / n).toFixed(2)}s/question)`);
  console.log();
  console.log(`  Recall@5:   ${(totalR5 / n * 100).toFixed(1)}%  (${totalR5}/${n})`);
  console.log(`  Recall@10:  ${(totalR10 / n * 100).toFixed(1)}%  (${totalR10}/${n})`);
  console.log(`  NDCG@10:    ${(totalNDCG10 / n).toFixed(3)}`);
  console.log();

  // Per-type breakdown
  console.log("  Per-type breakdown:");
  const typeEntries = Object.entries(perType).sort(
    ([, a], [, b]) => b.r5 / b.count - a.r5 / a.count
  );
  for (const [type, stats] of typeEntries) {
    const r5pct = ((stats.r5 / stats.count) * 100).toFixed(1);
    const r10pct = ((stats.r10 / stats.count) * 100).toFixed(1);
    console.log(
      `    ${type.padEnd(30)} R@5: ${r5pct.padStart(5)}%  R@10: ${r10pct.padStart(5)}%  (${stats.count} questions)`
    );
  }

  // Failures
  if (failures.length > 0) {
    console.log(`\n  Failures (${failures.length}):`);
    for (const f of failures.slice(0, 20)) {
      console.log(
        `    #${f.qi} [${f.question_type}] ${f.question}...`
      );
      console.log(`       gold: ${f.gold.join(", ")}`);
      console.log(`       got:  ${f.retrieved.join(", ")}`);
    }
    if (failures.length > 20) {
      console.log(`    ... and ${failures.length - 20} more`);
    }
  }

  console.log();
  console.log("  MemPalace reference (raw mode): R@5=96.6% R@10=98.2% NDCG@10=0.889");
  console.log("=".repeat(70));

  // Exit with error if below threshold
  process.exit(failures.length > 0 && totalR5 / n < 0.9 ? 1 : 0);
}

main().catch((err) => {
  console.error("Benchmark failed:", err);
  process.exit(1);
});
