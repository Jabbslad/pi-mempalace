---
name: memory-setup
description: Set up persistent agent memory. Configures identity, checks dependencies, and prepares the memory store. Use when asked to "set up memory", "configure memory", or when memory_status shows unconfigured state.
---

# Memory Setup

Set up pi-mempalace for persistent agent memory across sessions.

## Prerequisites

No external dependencies required. Everything runs in Node.js via `@huggingface/transformers`.

The first operation that needs embeddings (search or store) will download the all-MiniLM-L6-v2 model (~90MB, cached automatically).

## Setup Steps

### 1. Create memory directory

```bash
mkdir -p ~/.pi/agent/memory
```

### 2. Configure identity

Ask the user about themselves and write `~/.pi/agent/memory/identity.txt`. This is loaded into every session (~100 tokens). Keep it concise.

Template:
```
I am an AI assistant for [Name].
Projects: [list active projects with brief descriptions]
Preferences: [coding style, frameworks, communication style]
Team: [key people and their roles, if applicable]
```

Example:
```
I am an AI assistant for Alice.
Projects: Driftwood (SaaS analytics dashboard, React + PostgreSQL), Orion (CLI tool in Rust).
Preferences: TypeScript, functional style, clear error messages, test-driven.
Team: Bob (backend lead), Maya (design), Soren (DevOps).
```

### 3. Set default project

```bash
# The memory extension auto-detects from the current directory,
# but you can set a default:
```

Use `/memory project <name>` to set the current project context.

### 4. Verify setup

Run the `memory_status` tool to confirm everything is working:
- Identity should show ✅
- Total memories should be 0 (fresh install)
- Backend should show "pure TypeScript (in-process)"

### 5. Optional: Import existing conversations

If the user has exported conversations (from Claude, ChatGPT, etc.), help them store key exchanges:

```bash
# Read the export file, extract important exchanges, and use memory_save
# for each one. Focus on decisions, architecture choices, and key context.
```

## How Memory Works

Once set up, memory works automatically:

- **Auto-capture**: Every conversation turn is embedded and stored directly (no buffering)
- **Wake-up**: Each new session starts with your identity + recent context (~600-900 tokens)
- **Search**: Use `memory_search` to find past conversations by meaning
- **Save**: Use `memory_save` to explicitly remember important decisions
- **Recall**: Use `memory_recall` to browse by project or topic

## Configuration

Config is stored at `~/.pi/agent/memory/config.json`:

```json
{
  "autoCapture": true,
  "wakeUpEnabled": true,
  "wakeUpMaxTokens": 800,
  "defaultProject": null
}
```

## Commands

- `/memory status` — show memory overview
- `/memory project <name>` — set current project
- `/memory on` / `/memory off` — enable/disable memory
- `/memory search <query>` — quick search
