"""Run all memory concepts on safe, synthetic sample content.

python 05_memory/demo.py
python 05_memory/demo.py --milvus   # requires Ollama + Milvus Lite/server
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict

from memory_core import MemoryStore, compact_messages, consolidate_candidates, estimate_tokens


class ChatState(TypedDict):
    messages: list[str]
    last_reply: str


def reply_node(state: ChatState) -> dict:
    return {"last_reply": "Recorded: " + state["messages"][-1]}


def checkpoint_demo(db_path: str) -> None:
    builder = StateGraph(ChatState)
    builder.add_node("reply", reply_node)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    config = {"configurable": {"thread_id": "memory-lab-chat"}}
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        graph.invoke({"messages": ["First message"], "last_reply": ""}, config)
        first = graph.get_state(config).values
        print("[Checkpoint] persisted last_reply:", first["last_reply"])
    # Open a new SQLite connection to demonstrate process-like restoration.
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        restored = graph.get_state(config).values
        print("[Checkpoint] reopened and restored:", restored["last_reply"])
        assert restored["last_reply"] == first["last_reply"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milvus", action="store_true")
    parser.add_argument("--db", default="./data/memory_lab.sqlite")
    args = parser.parse_args()

    # Context window is a hard limit on input + output tokens; this is only
    # a rough input budget demonstration, not a real tokenizer or KV cache.
    messages = [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": "Explain semantic memory briefly."},
        {"role": "assistant", "content": "Semantic memory stores facts."},
        {"role": "user", "content": "Explain episodic memory with a concrete example."},
        {"role": "assistant", "content": "Episodic memory records a particular past event."},
        {"role": "user", "content": "Now summarize the difference in one sentence."},
    ]
    compact = compact_messages(messages, max_estimated_tokens=75)
    print("[Context] turns before/after:", len(messages), len(compact))
    print("[Context] estimated tokens before/after:",
          sum(estimate_tokens(m["content"]) for m in messages),
          sum(estimate_tokens(m["content"]) for m in compact))

    store = MemoryStore(args.db)
    try:
        examples = [
            {"role": "user", "content": "FACT: The demo user's favorite programming language is Python."},
            {"role": "user", "content": "EVENT: Yesterday the demo agent retrieved a welding rule."},
            {"role": "user", "content": "PROCEDURE: Before publishing a report, ask for human approval."},
        ]
        candidates = consolidate_candidates(examples)
        for candidate in candidates:
            record = store.remember(**candidate, ttl_seconds=7 * 24 * 3600)
            print(f"[Consolidate] {record.kind}: {record.content} [{record.id[:8]}]")
        print("[Long term] active count:", len(store.list_active()))

        checkpoint_demo("./data/memory_chat_checkpoint.sqlite")

        if args.milvus:
            from milvus_memory import MilvusMemoryIndex
            index = MilvusMemoryIndex()
            try:
                index.rebuild(store)
                hits = index.search("What must happen before publishing a report?", top_k=3)
                print("[Milvus] semantic hits:", hits)
            finally:
                index.close()

        # Demonstrate explicit forgetting with an exact ID.
        first_id = store.list_active()[-1].id
        print("[Forget] deleted exact ID:", store.forget(first_id))
        print("[Long term] after deletion:", len(store.list_active()))
        if args.milvus:
            print("[Milvus] Note: rebuild the index after forget() to clear stale vectors.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
