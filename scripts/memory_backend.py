#!/usr/bin/env python3
"""
memory_backend.py — ChromaDB-backed memory storage for pi-memory.

Accepts JSON commands via CLI, returns JSON to stdout.
Single dependency: chromadb (includes local embeddings).

Usage:
    python3 memory_backend.py store '{"content": "...", "project": "myapp", "topic": "auth"}'
    python3 memory_backend.py search '{"query": "database decision", "project": "myapp"}'
    python3 memory_backend.py wakeup '{"project": "myapp"}'
    python3 memory_backend.py status '{}'
    python3 memory_backend.py delete '{"id": "mem_abc123"}'
    python3 memory_backend.py list-projects '{}'
    python3 memory_backend.py batch-store '{"items": [...]}'
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Suppress chromadb telemetry and noisy logs
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import logging
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("onnxruntime").setLevel(logging.WARNING)

import chromadb


MEMORY_DIR = os.path.expanduser("~/.pi/agent/memory")
PALACE_DIR = os.path.join(MEMORY_DIR, "palace")
COLLECTION_NAME = "pi_memory"
IDENTITY_PATH = os.path.join(MEMORY_DIR, "identity.txt")


def get_collection():
    """Get or create the ChromaDB collection."""
    client = chromadb.PersistentClient(path=PALACE_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def content_hash(content: str) -> str:
    """SHA-256 hash of content for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def cmd_store(args: dict) -> dict:
    """Store a memory in ChromaDB."""
    content = args.get("content", "").strip()
    if not content:
        return {"error": "Empty content"}

    project = args.get("project", "general")
    topic = args.get("topic", "general")
    source = args.get("source", "auto-capture")
    timestamp = args.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    session_id = args.get("session_id", "")

    # Generate ID from content hash for deduplication
    c_hash = content_hash(content)
    doc_id = f"mem_{c_hash}"

    col = get_collection()

    # Check for duplicate
    try:
        existing = col.get(ids=[doc_id])
        if existing and existing["ids"]:
            return {"status": "duplicate", "id": doc_id}
    except Exception:
        pass

    col.add(
        ids=[doc_id],
        documents=[content],
        metadatas=[{
            "project": project,
            "topic": topic,
            "source": source,
            "timestamp": timestamp,
            "session_id": session_id,
        }],
    )

    return {"status": "stored", "id": doc_id}


def cmd_batch_store(args: dict) -> dict:
    """Store multiple memories in one batch."""
    items = args.get("items", [])
    if not items:
        return {"error": "No items provided"}

    col = get_collection()
    results = []
    ids_to_add = []
    docs_to_add = []
    metas_to_add = []

    for item in items:
        content = item.get("content", "").strip()
        if not content:
            continue

        c_hash = content_hash(content)
        doc_id = f"mem_{c_hash}"

        # Check duplicate
        try:
            existing = col.get(ids=[doc_id])
            if existing and existing["ids"]:
                results.append({"status": "duplicate", "id": doc_id})
                continue
        except Exception:
            pass

        project = item.get("project", "general")
        topic = item.get("topic", "general")
        source = item.get("source", "auto-capture")
        timestamp = item.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        session_id = item.get("session_id", "")

        ids_to_add.append(doc_id)
        docs_to_add.append(content)
        metas_to_add.append({
            "project": project,
            "topic": topic,
            "source": source,
            "timestamp": timestamp,
            "session_id": session_id,
        })
        results.append({"status": "stored", "id": doc_id})

    if ids_to_add:
        col.add(ids=ids_to_add, documents=docs_to_add, metadatas=metas_to_add)

    return {"stored": len(ids_to_add), "duplicates": len(items) - len(ids_to_add), "results": results}


def cmd_search(args: dict) -> dict:
    """Semantic search across memories."""
    query = args.get("query", "").strip()
    if not query:
        return {"error": "Empty query"}

    project = args.get("project")
    topic = args.get("topic")
    n_results = min(args.get("n_results", 5), 20)

    col = get_collection()

    # Build metadata filter
    where = None
    if project and topic:
        where = {"$and": [{"project": project}, {"topic": topic}]}
    elif project:
        where = {"project": project}
    elif topic:
        where = {"topic": topic}

    kwargs = {
        "query_texts": [query],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    try:
        results = col.query(**kwargs)
    except Exception as e:
        return {"error": f"Search failed: {e}"}

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    ids = results["ids"][0]

    hits = []
    for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
        hits.append({
            "id": doc_id,
            "text": doc,
            "project": meta.get("project", "unknown"),
            "topic": meta.get("topic", "unknown"),
            "source": meta.get("source", "unknown"),
            "timestamp": meta.get("timestamp", ""),
            "similarity": round(1 - dist, 4),
        })

    return {
        "query": query,
        "filters": {"project": project, "topic": topic},
        "results": hits,
    }


def cmd_wakeup(args: dict) -> dict:
    """Generate wake-up context: L0 (identity) + L1 (top memories)."""
    project = args.get("project")
    max_tokens = args.get("max_tokens", 800)

    parts = []

    # L0: Identity
    if os.path.exists(IDENTITY_PATH):
        with open(IDENTITY_PATH, "r") as f:
            identity = f.read().strip()
        parts.append(f"## Memory — Identity\n{identity}")
    else:
        parts.append("## Memory — Identity\nNo identity configured. Use /skill:memory-setup to set up.")

    # L1: Top memories (recent + important)
    try:
        col = get_collection()
        kwargs = {"include": ["documents", "metadatas"]}
        if project:
            kwargs["where"] = {"project": project}

        all_results = col.get(**kwargs)
        docs = all_results.get("documents", [])
        metas = all_results.get("metadatas", [])

        if docs:
            # Sort by timestamp descending (most recent first)
            paired = list(zip(docs, metas))
            paired.sort(key=lambda x: x[1].get("timestamp", ""), reverse=True)

            # Group by project
            by_project = {}
            for doc, meta in paired:
                proj = meta.get("project", "general")
                if proj not in by_project:
                    by_project[proj] = []
                by_project[proj].append((doc, meta))

            parts.append("\n## Memory — Recent Context")
            total_chars = 0
            max_chars = max_tokens * 4  # rough token-to-char ratio

            for proj, entries in sorted(by_project.items()):
                if total_chars > max_chars:
                    break
                parts.append(f"\n[{proj}]")
                for doc, meta in entries[:5]:  # top 5 per project
                    snippet = doc.strip().replace("\n", " ")
                    if len(snippet) > 200:
                        snippet = snippet[:197] + "..."
                    topic = meta.get("topic", "")
                    line = f"  - {snippet}"
                    if topic and topic != "general":
                        line = f"  - [{topic}] {snippet}"
                    total_chars += len(line)
                    if total_chars > max_chars:
                        parts.append("  ... (use memory_search for more)")
                        break
                    parts.append(line)
        else:
            parts.append("\n## Memory — Recent Context\nNo memories stored yet.")

    except Exception:
        parts.append("\n## Memory — Recent Context\nMemory store not initialized.")

    text = "\n".join(parts)
    return {
        "text": text,
        "token_estimate": len(text) // 4,
    }


def cmd_status(args: dict) -> dict:
    """Return memory store status."""
    result = {
        "memory_dir": MEMORY_DIR,
        "palace_dir": PALACE_DIR,
        "identity_exists": os.path.exists(IDENTITY_PATH),
        "total_memories": 0,
        "projects": {},
        "storage_size_kb": 0,
    }

    try:
        col = get_collection()
        count = col.count()
        result["total_memories"] = count

        if count > 0:
            all_results = col.get(include=["metadatas"])
            metas = all_results.get("metadatas", [])

            projects = {}
            for meta in metas:
                proj = meta.get("project", "general")
                projects[proj] = projects.get(proj, 0) + 1
            result["projects"] = projects
    except Exception:
        pass

    # Storage size
    palace_path = Path(PALACE_DIR)
    if palace_path.exists():
        total_size = sum(f.stat().st_size for f in palace_path.rglob("*") if f.is_file())
        result["storage_size_kb"] = round(total_size / 1024, 1)

    return result


def cmd_delete(args: dict) -> dict:
    """Delete a specific memory by ID."""
    doc_id = args.get("id", "").strip()
    if not doc_id:
        return {"error": "No id provided"}

    col = get_collection()
    try:
        col.delete(ids=[doc_id])
        return {"status": "deleted", "id": doc_id}
    except Exception as e:
        return {"error": f"Delete failed: {e}"}


def cmd_list_projects(args: dict) -> dict:
    """List all projects with memory counts."""
    try:
        col = get_collection()
        all_results = col.get(include=["metadatas"])
        metas = all_results.get("metadatas", [])

        projects = {}
        for meta in metas:
            proj = meta.get("project", "general")
            projects[proj] = projects.get(proj, 0) + 1

        return {"projects": projects, "total": sum(projects.values())}
    except Exception:
        return {"projects": {}, "total": 0}


def cmd_recall(args: dict) -> dict:
    """L2 on-demand retrieval: get memories filtered by project/topic."""
    project = args.get("project")
    topic = args.get("topic")
    n_results = min(args.get("n_results", 10), 50)

    col = get_collection()

    where = None
    if project and topic:
        where = {"$and": [{"project": project}, {"topic": topic}]}
    elif project:
        where = {"project": project}
    elif topic:
        where = {"topic": topic}

    kwargs = {"include": ["documents", "metadatas"]}
    if where:
        kwargs["where"] = where

    try:
        results = col.get(**kwargs)
    except Exception as e:
        return {"error": f"Recall failed: {e}"}

    docs = results.get("documents", [])
    metas = results.get("metadatas", [])

    # Sort by timestamp descending
    paired = list(zip(docs, metas))
    paired.sort(key=lambda x: x[1].get("timestamp", ""), reverse=True)
    paired = paired[:n_results]

    items = []
    for doc, meta in paired:
        items.append({
            "text": doc,
            "project": meta.get("project", "unknown"),
            "topic": meta.get("topic", "unknown"),
            "source": meta.get("source", "unknown"),
            "timestamp": meta.get("timestamp", ""),
        })

    return {
        "filters": {"project": project, "topic": topic},
        "count": len(items),
        "results": items,
    }


# ---------------------------------------------------------------------------
# CLI dispatcher
# ---------------------------------------------------------------------------

COMMANDS = {
    "store": cmd_store,
    "batch-store": cmd_batch_store,
    "search": cmd_search,
    "wakeup": cmd_wakeup,
    "status": cmd_status,
    "delete": cmd_delete,
    "list-projects": cmd_list_projects,
    "recall": cmd_recall,
}


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: memory_backend.py <command> [json_args]"}))
        sys.exit(1)

    command = sys.argv[1]
    if command not in COMMANDS:
        print(json.dumps({"error": f"Unknown command: {command}. Available: {list(COMMANDS.keys())}"}))
        sys.exit(1)

    # Parse args
    args = {}
    if len(sys.argv) > 2:
        try:
            args = json.loads(sys.argv[2])
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON args: {e}"}))
            sys.exit(1)

    # Ensure memory directory exists
    os.makedirs(PALACE_DIR, exist_ok=True)

    try:
        result = COMMANDS[command](args)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": f"{command} failed: {e}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
