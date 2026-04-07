#!/usr/bin/env python3
"""Tests for memory_backend.py"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "memory_backend.py")


@pytest.fixture(autouse=True)
def temp_memory_dir(monkeypatch, tmp_path):
    """Use a temporary directory for all memory storage."""
    memory_dir = str(tmp_path / "memory")
    palace_dir = os.path.join(memory_dir, "palace")
    identity_path = os.path.join(memory_dir, "identity.txt")

    # Patch the module-level constants by setting env and reimporting
    # Instead, we'll just set the env vars that the script checks
    monkeypatch.setenv("HOME", str(tmp_path))

    # Create the memory directory structure
    os.makedirs(os.path.join(str(tmp_path), ".pi", "agent", "memory"), exist_ok=True)

    return tmp_path


def run_backend(command, args=None, home=None):
    """Run the backend script and return parsed JSON."""
    cmd = [sys.executable, SCRIPT, command]
    if args:
        cmd.append(json.dumps(args))

    env = os.environ.copy()
    if home:
        env["HOME"] = str(home)
        # Preserve user site-packages so chromadb can be found
        import site
        user_site = site.getusersitepackages()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{user_site}:{existing}" if existing else user_site

    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=30
    )

    if result.returncode != 0:
        # Try to parse error JSON from stdout
        try:
            return json.loads(result.stdout.strip().split("\n")[-1])
        except (json.JSONDecodeError, IndexError):
            raise RuntimeError(f"Backend failed: {result.stderr}")

    # Parse last line as JSON (skip progress bars etc.)
    lines = result.stdout.strip().split("\n")
    return json.loads(lines[-1])


class TestStore:
    def test_store_basic(self, temp_memory_dir):
        result = run_backend("store", {
            "content": "We chose PostgreSQL for concurrent writes.",
            "project": "myapp",
            "topic": "database",
        }, home=temp_memory_dir)
        assert result["status"] == "stored"
        assert result["id"].startswith("mem_")

    def test_store_dedup(self, temp_memory_dir):
        content = "Duplicate content test."
        r1 = run_backend("store", {"content": content}, home=temp_memory_dir)
        r2 = run_backend("store", {"content": content}, home=temp_memory_dir)
        assert r1["status"] == "stored"
        assert r2["status"] == "duplicate"
        assert r1["id"] == r2["id"]

    def test_store_empty(self, temp_memory_dir):
        result = run_backend("store", {"content": ""}, home=temp_memory_dir)
        assert "error" in result

    def test_store_defaults(self, temp_memory_dir):
        result = run_backend("store", {
            "content": "Some content without explicit metadata.",
        }, home=temp_memory_dir)
        assert result["status"] == "stored"


class TestBatchStore:
    def test_batch_store(self, temp_memory_dir):
        result = run_backend("batch-store", {
            "items": [
                {"content": "Item one", "project": "a"},
                {"content": "Item two", "project": "b"},
                {"content": "Item three", "project": "a"},
            ]
        }, home=temp_memory_dir)
        assert result["stored"] == 3
        assert result["duplicates"] == 0

    def test_batch_store_with_duplicates(self, temp_memory_dir):
        run_backend("store", {"content": "Already exists"}, home=temp_memory_dir)
        result = run_backend("batch-store", {
            "items": [
                {"content": "Already exists"},
                {"content": "New item"},
            ]
        }, home=temp_memory_dir)
        assert result["stored"] == 1
        assert result["duplicates"] == 1

    def test_batch_store_empty(self, temp_memory_dir):
        result = run_backend("batch-store", {"items": []}, home=temp_memory_dir)
        assert "error" in result


class TestSearch:
    def test_search_basic(self, temp_memory_dir):
        run_backend("store", {
            "content": "PostgreSQL was chosen for concurrent write support.",
            "project": "myapp", "topic": "database",
        }, home=temp_memory_dir)
        run_backend("store", {
            "content": "React is our frontend framework with TypeScript.",
            "project": "myapp", "topic": "frontend",
        }, home=temp_memory_dir)

        result = run_backend("search", {
            "query": "what database do we use?",
        }, home=temp_memory_dir)

        assert "results" in result
        assert len(result["results"]) > 0
        # The database memory should be the top result
        assert "PostgreSQL" in result["results"][0]["text"]

    def test_search_with_project_filter(self, temp_memory_dir):
        run_backend("store", {
            "content": "App uses PostgreSQL.",
            "project": "myapp",
        }, home=temp_memory_dir)
        run_backend("store", {
            "content": "CLI uses SQLite.",
            "project": "cli-tool",
        }, home=temp_memory_dir)

        result = run_backend("search", {
            "query": "database",
            "project": "cli-tool",
        }, home=temp_memory_dir)

        assert all(r["project"] == "cli-tool" for r in result["results"])

    def test_search_empty_query(self, temp_memory_dir):
        result = run_backend("search", {"query": ""}, home=temp_memory_dir)
        assert "error" in result


class TestWakeup:
    def test_wakeup_no_identity(self, temp_memory_dir):
        result = run_backend("wakeup", {}, home=temp_memory_dir)
        assert "text" in result
        assert "No identity configured" in result["text"]

    def test_wakeup_with_identity(self, temp_memory_dir):
        identity_path = os.path.join(
            str(temp_memory_dir), ".pi", "agent", "memory", "identity.txt"
        )
        with open(identity_path, "w") as f:
            f.write("I am an AI assistant for Alice.")

        result = run_backend("wakeup", {}, home=temp_memory_dir)
        assert "Alice" in result["text"]
        assert result["token_estimate"] > 0

    def test_wakeup_with_memories(self, temp_memory_dir):
        run_backend("store", {
            "content": "Important decision about auth.",
            "project": "myapp", "topic": "auth",
        }, home=temp_memory_dir)

        result = run_backend("wakeup", {"project": "myapp"}, home=temp_memory_dir)
        assert "auth" in result["text"].lower()


class TestStatus:
    def test_status_empty(self, temp_memory_dir):
        result = run_backend("status", {}, home=temp_memory_dir)
        assert result["total_memories"] == 0
        assert result["projects"] == {}

    def test_status_with_data(self, temp_memory_dir):
        run_backend("store", {"content": "A", "project": "p1"}, home=temp_memory_dir)
        run_backend("store", {"content": "B", "project": "p1"}, home=temp_memory_dir)
        run_backend("store", {"content": "C", "project": "p2"}, home=temp_memory_dir)

        result = run_backend("status", {}, home=temp_memory_dir)
        assert result["total_memories"] == 3
        assert result["projects"]["p1"] == 2
        assert result["projects"]["p2"] == 1


class TestDelete:
    def test_delete(self, temp_memory_dir):
        stored = run_backend("store", {"content": "To be deleted"}, home=temp_memory_dir)
        doc_id = stored["id"]

        result = run_backend("delete", {"id": doc_id}, home=temp_memory_dir)
        assert result["status"] == "deleted"

        # Verify it's gone
        status = run_backend("status", {}, home=temp_memory_dir)
        assert status["total_memories"] == 0

    def test_delete_empty_id(self, temp_memory_dir):
        result = run_backend("delete", {"id": ""}, home=temp_memory_dir)
        assert "error" in result


class TestRecall:
    def test_recall_by_project(self, temp_memory_dir):
        run_backend("store", {"content": "A fact about auth", "project": "myapp", "topic": "auth"}, home=temp_memory_dir)
        run_backend("store", {"content": "A fact about DB", "project": "myapp", "topic": "db"}, home=temp_memory_dir)
        run_backend("store", {"content": "Other project", "project": "other"}, home=temp_memory_dir)

        result = run_backend("recall", {"project": "myapp"}, home=temp_memory_dir)
        assert result["count"] == 2
        assert all(r["project"] == "myapp" for r in result["results"])


class TestListProjects:
    def test_list_projects(self, temp_memory_dir):
        run_backend("store", {"content": "A", "project": "p1"}, home=temp_memory_dir)
        run_backend("store", {"content": "B", "project": "p2"}, home=temp_memory_dir)

        result = run_backend("list-projects", {}, home=temp_memory_dir)
        assert "p1" in result["projects"]
        assert "p2" in result["projects"]
        assert result["total"] == 2


class TestCLI:
    def test_unknown_command(self, temp_memory_dir):
        result = run_backend("nonexistent", {}, home=temp_memory_dir)
        assert "error" in result

    def test_invalid_json(self, temp_memory_dir):
        cmd = [sys.executable, SCRIPT, "status", "not-json"]
        env = os.environ.copy()
        env["HOME"] = str(temp_memory_dir)
        import site
        user_site = site.getusersitepackages()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{user_site}:{existing}" if existing else user_site
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        output = json.loads(proc.stdout.strip().split("\n")[-1])
        assert "error" in output
