"""Pure-Python Agent memory primitives: context budgeting + durable typed memories.

No network, LLM, or Milvus required. SQLite makes memory survive restarts.
Educational baseline: messages are approximated with character-based token
estimates; production systems should use the model's actual tokenizer.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

KINDS = ("semantic", "episodic", "procedural")


def estimate_tokens(text: str) -> int:
    """Rough proxy ONLY; not a tokenizer or exact context usage."""
    return max(1, (len(text) + 3) // 4)


def compact_messages(messages: list[dict], max_estimated_tokens: int = 120) -> list[dict]:
    """Keep the system instruction, a deterministic summary of old turns,
    and the latest turns. No LLM invoked here.
    """
    if not messages:
        return []
    head = [messages[0]] if messages[0].get("role") == "system" else []
    rest = messages[len(head):]
    budget = max_estimated_tokens - sum(estimate_tokens(m["content"]) for m in head)
    if budget < 1:
        raise ValueError("The system instruction exceeds the context budget")
    if sum(estimate_tokens(m["content"]) for m in rest) <= budget:
        return messages.copy()

    tail: list[dict] = []
    remaining = max(1, int(budget * 0.65))
    for msg in reversed(rest):
        cost = estimate_tokens(msg["content"])
        if tail and cost > remaining:
            break
        if cost > remaining and not tail:
            tail.append({**msg, "content": msg["content"][-4 * remaining:]})
            remaining = 0
            break
        tail.append(msg)
        remaining -= cost
    tail.reverse()
    dropped = rest[:len(rest) - len(tail)]
    if not dropped:
        return head + tail

    # Deterministic metadata, not a semantic summary: no fabricated facts.
    synopsis = "; ".join(f"{m['role']} turn" for m in dropped)
    summary = {
        "role": "system",
        "content": (
            f"Conversation history compacted: {len(dropped)} older turns omitted "
            f"({synopsis[:80]}). Prior details are NOT available; ask if needed."
        ),
    }
    # Reserve space for summary; truncate oldest tail if necessary.
    while tail and sum(estimate_tokens(m["content"]) for m in head + [summary] + tail) > max_estimated_tokens:
        tail.pop(0)
    if sum(estimate_tokens(m["content"]) for m in head + [summary] + tail) > max_estimated_tokens:
        raise ValueError("Context budget too small for header and compression marker")
    return head + [summary] + tail


@dataclass(frozen=True)
class Memory:
    id: str
    kind: str
    content: str
    source: str
    created_at: float
    expires_at: float | None
    importance: float


class MemoryStore:
    def __init__(self, path: str = "./data/agent_memories.sqlite") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            "id TEXT PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL, "
            "source TEXT NOT NULL, created_at REAL NOT NULL, "
            "expires_at REAL, importance REAL NOT NULL)"
        )
        self.conn.commit()

    def remember(
        self, kind: str, content: str, source: str,
        *, importance: float = 0.5, ttl_seconds: int | None = None,
    ) -> Memory:
        if kind not in KINDS:
            raise ValueError("Memory kind must be semantic, episodic or procedural")
        if not content.strip() or not source.strip():
            raise ValueError("Memory requires nonempty content and provenance")
        if not 0 <= importance <= 1:
            raise ValueError("importance must be in [0, 1]")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        now = time.time()
        record = Memory(
            id=str(uuid.uuid4()), kind=kind, content=content.strip(),
            source=source.strip(), created_at=now,
            expires_at=(now + ttl_seconds if ttl_seconds is not None else None),
            importance=importance,
        )
        self.conn.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?)",
            (record.id, record.kind, record.content, record.source,
             record.created_at, record.expires_at, record.importance),
        )
        self.conn.commit()
        return record

    def list_active(self, kind: str | None = None) -> list[Memory]:
        if kind is not None and kind not in KINDS:
            raise ValueError("Invalid memory kind")
        sql = ("SELECT id,kind,content,source,created_at,expires_at,importance "
               "FROM memories WHERE (expires_at IS NULL OR expires_at > ?)")
        params: list = [time.time()]
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        sql += " ORDER BY importance DESC, created_at DESC"
        return [Memory(*row) for row in self.conn.execute(sql, params)]

    def forget(self, memory_id: str) -> bool:
        cursor = self.conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def prune(self) -> int:
        cursor = self.conn.execute(
            "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (time.time(),),
        )
        self.conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        self.conn.close()


def consolidate_candidates(messages: list[dict]) -> list[dict]:
    """Simple explicit-markers extractor, NOT autonomous memory inference.

    Saves only statements explicitly prefixed FACT:, EVENT:, or PROCEDURE:.
    This reduces accidental memory creation from arbitrary user text.
    """
    prefix_map = {"FACT:": "semantic", "EVENT:": "episodic", "PROCEDURE:": "procedural"}
    records = []
    for index, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "").strip()
        for prefix, kind in prefix_map.items():
            if content.startswith(prefix):
                value = content[len(prefix):].strip()
                if value:
                    records.append({"kind": kind, "content": value, "source": f"user_turn_{index}"})
                break
    return records
