"""Deterministic tests, requiring no Ollama and no Milvus service."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

BASE = Path(__file__).resolve().parents[1] / "05_memory"
sys.path.insert(0, str(BASE))
from memory_core import MemoryStore, compact_messages, consolidate_candidates, estimate_tokens


def test_compaction_uses_budget_and_retains_latest():
    messages = [
        {"role": "system", "content": "You are a careful assistant."},
        *[{"role": "user", "content": f"Turn {i}: " + "x" * 50} for i in range(12)],
        {"role": "user", "content": "Latest important question?"},
    ]
    compacted = compact_messages(messages, 100)
    assert compacted[0] == messages[0]
    assert compacted[-1] == messages[-1]
    assert any("compacted" in m["content"] for m in compacted)
    assert sum(estimate_tokens(m["content"]) for m in compacted) <= 100


def test_typed_memory_forget_and_provenance(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.sqlite"))
    try:
        memory = store.remember("semantic", "User prefers Python", "explicit_user_statement")
        assert memory in store.list_active("semantic")
        assert store.forget(memory.id)
        assert store.list_active() == []
        assert not store.forget(memory.id)
        with pytest.raises(ValueError):
            store.remember("unknown", "x", "source")
    finally:
        store.close()


def test_consolidation_only_explicit_messages():
    turns = [
        {"role": "user", "content": "FACT: Demo user prefers Python"},
        {"role": "user", "content": "Maybe I like Java?"},
        {"role": "assistant", "content": "FACT: This must never be written as a user fact"},
        {"role": "user", "content": "PROCEDURE: Request approval before release"},
    ]
    result = consolidate_candidates(turns)
    assert [x["kind"] for x in result] == ["semantic", "procedural"]
    assert all(x["source"].startswith("user_turn_") for x in result)


def test_memory_survives_reopen(tmp_path):
    path = str(tmp_path / "memory.sqlite")
    with_store = MemoryStore(path)
    with_store.remember("episodic", "A sample review finished", "demo")
    with_store.close()
    reopened = MemoryStore(path)
    assert reopened.list_active()[0].content == "A sample review finished"
    reopened.close()
