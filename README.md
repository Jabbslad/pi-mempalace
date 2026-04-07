# pi-memory

Persistent agent memory for [pi](https://github.com/badlogic/pi-mono). Never lose context again.

Every conversation you have with an AI — every decision, every debugging session, every architecture debate — disappears when the session ends. pi-memory stores your exchanges verbatim and makes them searchable across sessions using semantic similarity.

## How it works

**Store everything, search it later.** No LLM summarization, no lossy extraction. Raw verbatim text + local embeddings (all-MiniLM-L6-v2 via ChromaDB) = high-accuracy retrieval at zero API cost.

- **Auto-capture**: Conversation exchanges are buffered during sessions and stored on shutdown
- **Wake-up context**: Each new session starts with your identity + recent memories (~600-900 tokens)
- **Semantic search**: Find past decisions by meaning, not just keywords
- **Project-aware**: Memories are tagged by project (auto-detected from directory) and topic
- **Fully local**: ChromaDB runs locally, embeddings computed locally, no cloud dependency

## Install

```bash
# Install chromadb (the only dependency)
pip3 install 'chromadb>=0.4.0,<1'

# Install the pi extension
pi install /path/to/pi-memory
# or
pi install git:github.com/your-username/pi-memory
```

## Quick Start

```bash
# Set up identity and verify
/skill:memory-setup

# Save something important
memory_save("We chose PostgreSQL for concurrent write support", project: "myapp", topic: "database")

# Search later
memory_search("why did we pick the database?")

# Browse a project's memories
memory_recall(project: "myapp")
```

## Tools

| Tool | Purpose |
|------|---------|
| `memory_search` | Semantic search across all stored memories |
| `memory_save` | Explicitly save important information |
| `memory_recall` | Browse memories by project/topic |
| `memory_status` | Show memory store overview |

## Commands

| Command | Purpose |
|---------|---------|
| `/memory status` | Quick status overview |
| `/memory flush` | Force-flush buffered exchanges |
| `/memory project <name>` | Set current project context |
| `/memory search <query>` | Quick search |
| `/memory on` / `off` | Enable/disable memory |

## Architecture

```
pi (TypeScript extension)          Python backend (ChromaDB)
┌─────────────────────┐           ┌──────────────────────┐
│ turn_end → buffer    │           │ memory_backend.py     │
│ session_end → flush  │──JSON──→ │                        │
│ before_agent_start   │           │ store / search /       │
│   → inject wake-up   │←──JSON──│ wakeup / status         │
│ Tools: search/save/  │           │                        │
│   recall/status      │           │ ChromaDB + MiniLM      │
│ Widget: 🧠 N memories│           │ ~/.pi/agent/memory/    │
└─────────────────────┘           └──────────────────────┘
```

## Inspired by

[mempalace](https://github.com/your-username/mempalace) — which demonstrated that raw verbatim storage + semantic search achieves 96.6% retrieval accuracy on LongMemEval, beating LLM-based extraction approaches.

## License

MIT
