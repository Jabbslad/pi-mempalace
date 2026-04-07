# 🧠 pi-mempalace

**Your AI forgot everything again. How delightful.**

Every conversation you've ever had with an AI — every architectural decision, every late-night debugging eureka, every "let's use Postgres because..." — *poof*. Gone the moment you close the tab. Your AI has the long-term memory of a goldfish at a rave.

**pi-mempalace fixes that.** It gives [pi](https://github.com/badlogic/pi-mono) agents persistent, cross-session memory. Store everything. Search it later. Never re-explain your life choices to a machine again.

---

## 🏰 Standing on the Shoulders of a Memory Palace

This project is directly inspired by the wonderful [MemPalace](https://www.mempalace.tech) — built by **Milla Jovovich** (yes, *that* Milla Jovovich — Leeloo from *The Fifth Element*, Alice from *Resident Evil*) and developer **Ben Sigman**.

Milla's origin story is painfully relatable: after thousands of conversations with AI, she realized every new session was a clean slate. All her decisions, reasoning, creative ideas — thrown into the void. Existing memory tools like Mem0 and Zep tried to help, but they had a fatal flaw: they used AI to decide what was worth remembering. The nuance, the "why," the reasoning behind decisions — exactly the stuff that matters — was the first to go.

So Milla and Ben spent months building MemPalace with Claude Code, and landed on a beautifully simple idea:

> **Don't let AI decide what to forget — store everything, then make it findable.**

Their MemPalace scored **96.6% on LongMemEval** (the standard benchmark for AI memory) in raw mode — the highest published result requiring zero API calls. The full system hit 100% with hybrid reranking, sparking a [glorious internet debate](https://www.mempalace.tech) about benchmarks and bragging rights. 7,000+ GitHub stars in 48 hours. Not bad for an actress and a developer with an idea.

pi-mempalace takes that core philosophy — **verbatim storage + semantic search** — and reimagines it as a native [pi](https://github.com/badlogic/pi-mono) extension. No Python. No ChromaDB. No pip install nightmares. Just pure TypeScript running in-process, fast enough to forget that forgetting was ever a problem.

---

## ✨ What It Does

- **🔄 Auto-capture** — Every conversation exchange is stored automatically after each turn. You don't have to remember to remember.
- **🌅 Wake-up context** — Each new session starts with a whisper of who you are and what you've been up to (~600-900 tokens of "previously on your life").
- **🔍 Semantic search** — Find past decisions by *meaning*, not keywords. "Why did we pick that database?" just works.
- **📁 Project-aware** — Memories are tagged by project (auto-detected from your directory) and topic. Your work stays organized even if you don't.
- **🏠 Fully local** — Embeddings computed in-process via `all-MiniLM-L6-v2`. No cloud calls. No API keys. No surveillance capitalism. Just you and your memories.
- **📊 Beautiful stats** — Sparkline activity charts, bar graphs by project/topic, and a TUI overlay that makes you feel like a hacker in a 90s movie.

---

## 🚀 Install

```bash
# Install from GitHub
pi install https://github.com/Jabbslad/pi-mempalace

# Or from a local checkout
pi install /path/to/pi-mempalace
```

That's it. `pi install` runs `npm install` automatically, which pulls in all three runtime dependencies:

| Dependency | What It Does | Native? |
|-----------|-------------|--------|
| `@huggingface/transformers` | Local embeddings (all-MiniLM-L6-v2, 384 dims) | No — pure JS, downloads model on first use (~80MB) |
| `better-sqlite3` | SQLite database access | Yes — compiles native addon via `node-gyp` |
| `sqlite-vec` | Vector similarity search for SQLite | Yes — ships prebuilt binary per platform |

No Python. No conda. No Docker. No ChromaDB server. No API keys. No sacrificial offerings to the dependency gods.

### Prerequisites

- **Node.js** (required by pi)
- **C++ toolchain** for `better-sqlite3`'s native build:
  - macOS: `xcode-select --install`
  - Linux: `sudo apt install build-essential` (or equivalent)
  - Windows: Visual Studio Build Tools

Most dev machines already have this. If `pi install` fails with gyp errors, that's the fix.

---

## 🎮 Quick Start

```bash
# Set up your identity and verify everything works
/skill:memory-setup

# Explicitly save something important
memory_save("We chose PostgreSQL for concurrent write support", project: "myapp", topic: "database")

# Search later, when you've inevitably forgotten
memory_search("why did we pick the database?")

# Browse a project's collected wisdom
memory_recall(project: "myapp")

# Check on your growing brain
memory_status()
```

---

## 🧰 Tools

### Memory Tools
| Tool | What It Does |
|------|-------------|
| `memory_search` | Semantic search across all stored memories — find things by meaning |
| `memory_save` | Explicitly save important info — for those "remember this" moments |
| `memory_recall` | Browse memories by project/topic — like flipping through a journal |
| `memory_status` | Memory store overview — how big is your brain now? |

### Palace Graph Tools
| Tool | What It Does |
|------|-------------|
| `memory_graph` | Visualize the palace — projects as wings, shared topics as tunnel connections |
| `memory_tunnel` | Traverse a tunnel between two projects via a shared topic |

### Knowledge Graph Tools
| Tool | What It Does |
|------|-------------|
| `knowledge_add` | Add structured facts — "myapp uses PostgreSQL since 2025-01" |
| `knowledge_query` | Query facts about an entity, with temporal filtering — "what did we use in 2024?" |
| `knowledge_status` | Knowledge graph stats — entities, facts, predicates |

## ⌨️ Commands

| Command | What It Does |
|---------|-------------|
| `/memory status` | Quick status overview |
| `/memory stats` | Full stats overlay with sparklines and bar charts |
| `/memory project <name>` | Set current project context |
| `/memory search <query>` | Quick search shortcut |
| `/memory graph` | Show palace graph with cross-project connections |
| `/memory knowledge <entity>` | Query knowledge graph for an entity |
| `/memory on` / `off` | Enable/disable memory (for those private moments) |

---

## 🏗️ Architecture

Following the [MemPalace](https://www.mempalace.tech) 4-layer memory stack:

```
┌─────────────────────────────────────────────────────────┐
│  L0: IDENTITY (~100 tokens)                             │
│  Always loaded. ~/.pi/agent/memory/identity.txt         │
├─────────────────────────────────────────────────────────┤
│  L1: ESSENTIAL STORY (~500-800 tokens)                  │
│  Top 15 memories by importance + recency.               │
│  Grouped by project. Injected at session start.         │
├─────────────────────────────────────────────────────────┤
│  L2: ON-DEMAND PROJECT CONTEXT                          │
│  Filtered by project/topic via SQL indexes.             │
│  Loaded only when you ask about a specific area.        │
├─────────────────────────────────────────────────────────┤
│  L3: DEEP SEMANTIC SEARCH                               │
│  Full vector similarity via sqlite-vec ANN index.       │
│  Searches 100K+ memories in milliseconds.               │
└─────────────────────────────────────────────────────────┘
```

```
pi (TypeScript — everything in-process)
┌──────────────────────────────────────────────────┐
│  extensions/pi-mempalace/                        │
│                                                  │
│  index.ts                memory_store.ts         │
│  ┌──────────────────┐    ┌────────────────────┐  │
│  │ turn_end →        │    │ MemoryStore        │  │
│  │   auto-capture    │───│                    │  │
│  │ before_agent →    │    │ SQLite + sqlite-vec│  │
│  │   L0+L1 inject    │←──│ ANN vector search  │  │
│  │ Tools + Commands  │    │ Metadata indexes   │  │
│  │ Stats TUI overlay │    │ WAL mode           │  │
│  └──────────────────┘    │                    │  │
│                           │ ~/.pi/agent/       │  │
│                           │   memory/          │  │
│                           │     memories.db    │  │
│                           └────────────────────┘  │
│                                                  │
│  @huggingface/transformers                       │
│    all-MiniLM-L6-v2 (384 dimensions)             │
│                                                  │
│  better-sqlite3 + sqlite-vec                     │
│    Indexed storage + vector similarity search    │
└──────────────────────────────────────────────────┘
```

### ⚡ Performance

| Operation | Time | Vibes |
|-----------|------|-------|
| Model load (once per session) | ~660ms | ☕ Sip of coffee |
| Embed text | ~4ms | ⚡ Blink and you'll miss it |
| Search 1,000 memories | ~5ms | 🚀 Faster than you can forget |
| Search 100,000 memories | ~15ms | 🚀 Still faster than you can forget |
| Store 1 memory | ~4ms | 💾 Practically instant |
| Startup (L0+L1 only) | ~250ms | 🌅 Dawn of a new session |

### 💾 Storage

Memories live in a SQLite database at `~/.pi/agent/memory/memories.db`, powered by [sqlite-vec](https://github.com/asg017/sqlite-vec) for vector similarity search. Metadata is indexed (project, topic, timestamp) for fast pre-filtering before vector search kicks in. Deduplication via SHA-256 hash means you can't accidentally remember the same thing twice — unlike that embarrassing story you keep retelling at parties.

> **Migrating from v0.1?** If you have an existing `memories.jsonl` file, it's automatically migrated to SQLite on first load. Your old file is renamed to `.bak`. No data lost, no action needed.

---

## 🏰 The Full MemPalace Architecture

The original [MemPalace](https://www.mempalace.tech) uses a gorgeous metaphorical architecture — **Wings** (top-level containers), **Rooms** (topics), **Halls** (corridors by memory type), **Closets** (compressed summaries), and **Drawers** (verbatim source files). It runs on Python with ChromaDB and includes AAAK, a custom 30x compression dialect that any LLM can read natively.

pi-mempalace faithfully implements the core MemPalace architecture in TypeScript:

| MemPalace Concept | pi-mempalace Implementation |
|---|---|
| **Wings** (projects/people) | `project` field — auto-detected from git repo |
| **Rooms** (topics within wings) | `topic` field — set per memory |
| **Drawers** (verbatim chunks) | 800-char chunks with 100-char overlap, each with its own embedding |
| **Tunnels** (cross-wing connections) | `memory_graph` discovers shared topics across projects |
| **4-Layer Stack** | L0 Identity → L1 Essential Story → L2 On-Demand → L3 Deep Search |
| **Knowledge Graph** | Temporal triples with `valid_from`/`valid_to` — "what was true when?" |
| **ChromaDB** | SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) — same HNSW indexing, no Python |

The one thing we deliberately skip is **AAAK compression** — MemPalace's own benchmarks show it drops accuracy from 96.6% to 84.2%. We'll take the storage cost over the precision loss.

The soul is the same: **your AI should remember you.**

If you want the full palace experience with all its wings and halls and drawers, go check out [mempalace.tech](https://www.mempalace.tech). Milla and Ben built something special.

---

## 🤔 Why Not Just Use MemPalace Directly?

You absolutely could! MemPalace is great. But if you're already living in the pi ecosystem:

- **No Python required** — pi-mempalace is pure TypeScript, runs in-process
- **Full MemPalace architecture** — 4-layer stack, chunking, palace graph, knowledge graph
- **SQLite instead of ChromaDB** — one file, no server, native Node.js via better-sqlite3
- **Palace graph with tunnels** — discover cross-project connections via shared topics
- **Temporal knowledge graph** — structured facts with time validity ("what was true when?")
- **Auto-chunking** — long content split into 800-char chunks with overlap, just like MemPalace
- **Native pi integration** — hooks into pi's extension system, session lifecycle, and TUI
- **Auto-capture built in** — no manual memory management needed
- **Wake-up context** — L0 identity + L1 essential story, injected before you even ask
- **Zero config** — install it and go. It just works.

---

## 📜 License

MIT — because memories should be free.

---

<p align="center">
  <i>
    "I wanted my AI to remember the way I remember — not just the conclusions, but the journey."
    <br/>
    — The philosophy behind MemPalace, which inspired this project
  </i>
</p>

<p align="center">
  🏰 Inspired by <a href="https://www.mempalace.tech">MemPalace</a> by Milla Jovovich & Ben Sigman
  <br/>
  🧠 Built for <a href="https://github.com/badlogic/pi-mono">pi</a> with ❤️ and a fear of forgetting
</p>
