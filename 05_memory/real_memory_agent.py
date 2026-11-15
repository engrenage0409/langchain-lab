"""Real Qwen + LangGraph + Milvus + SQLite memory chat.

Run:
    python 05_memory/real_memory_agent.py --thread chat-a
    python 05_memory/real_memory_agent.py --thread chat-b

Ollama must be running with qwen3-vl:2b-instruct and nomic-embed-text.
The two threads have different short-term history but the same long-term
memories. /remember proposes candidate facts for explicit approval.
"""
from __future__ import annotations

import argparse
import json
import operator
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph

from llm_memory import propose_memories
from memory_core import MemoryStore, compact_messages, estimate_tokens
from milvus_memory import MilvusMemoryIndex


class ChatState(TypedDict, total=False):
    # LangGraph reducer appends new turns to the durable thread history.
    messages: Annotated[list[dict[str, str]], operator.add]
    query: str
    retrieved: list[dict]
    answer: str


def make_graph(model: ChatOllama, index: MilvusMemoryIndex, checkpointer,
               max_context_tokens: int = 400):
    builder = StateGraph(ChatState)

    def retrieve(state: ChatState) -> dict:
        hits = index.search(state["query"], top_k=3)
        print("\n[LONG-TERM RETRIEVAL / Milvus]")
        for hit in hits:
            print(f"  {hit['kind']} score={hit['score']}: {hit['content']} ({hit['memory_id'][:8]})")
        if not hits:
            print("  (no memories yet)")
        return {"retrieved": hits}

    def answer(state: ChatState) -> dict:
        # Messages are checkpointed per thread. Do not include memories from
        # other users here: this is a single-user educational demo.
        past = state.get("messages", [])
        compacted = compact_messages(
            [{"role": "system", "content": "You are a concise, careful assistant."}] + past,
            max_estimated_tokens=max_context_tokens,
        )
        recalled = "\n".join(
            f"- [{h['kind']}] {h['content']} (memory_id={h['memory_id']})"
            for h in state.get("retrieved", [])
        ) or "(none)"
        messages = [{
            "role": "system",
            "content": (
                "Answer in Chinese. Use retrieved memories as historical context, "
                "not absolute truth. If memory conflicts with the current user's "
                "statement, trust the current statement and point out the conflict. "
                "Never invent memories.\nRelevant long-term memories:\n" + recalled
            ),
        }] + compacted
        print("\n[SHORT-TERM / LangGraph]")
        print(f"  history turns={len(past)}; after context compaction={len(compacted)-1}")
        print("  approximate input tokens=",
              sum(estimate_tokens(m["content"]) for m in messages))
        response = model.invoke(messages)
        raw = response.content
        reply = raw if isinstance(raw, str) else str(raw)
        print("\n[QWEN ANSWER]\n" + reply)
        return {
            "answer": reply,
            "messages": [{"role": "assistant", "content": reply}],
        }

    builder.add_node("retrieve", retrieve)
    builder.add_node("answer", answer)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=checkpointer)


def show_history(graph, config):
    state = graph.get_state(config).values
    messages = state.get("messages", [])
    print(f"\n[CHECKPOINT] persisted history: {len(messages)} messages")
    for m in messages[-8:]:
        print(f"  {m['role']}: {m['content'][:180]}")


def review_and_save(graph, config, model, store, index):
    messages = graph.get_state(config).values.get("messages", [])
    if not messages:
        print("No conversation yet. Chat first.")
        return
    print("\n[CONSOLIDATION] Qwen proposes long-term memories from this thread...")
    try:
        # JSON mode improves compliance on small Qwen models; the parser
        # still validates and reports malformed model responses.
        extraction_model = ChatOllama(
            model="qwen3-vl:2b-instruct", temperature=0,
            num_ctx=4096, format="json",
        )
        proposed = propose_memories(extraction_model, messages)
    except Exception as e:
        print(f"Memory extraction failed: {type(e).__name__}: {e}")
        return
    print("Summary (not automatically saved):", proposed["summary"])
    candidates = proposed["candidates_for_review"]
    if not candidates:
        print("No memory candidates. Nothing was saved.")
        return
    added = 0
    for candidate in candidates:
        print("\nCandidate type:", candidate["kind"])
        print("  Fact:", candidate["content"])
        print("  Source:", candidate["source"], candidate["source_text"][:300])
        approve = input("Store this memory? [yes/no]: ").strip().lower()
        if approve in {"yes", "y"}:
            record = store.remember(
                candidate["kind"], candidate["content"], candidate["source"],
            )
            print("  [SAVED SQLite]", record.id)
            added += 1
        else:
            print("  [SKIPPED]")
    if added:
        index.rebuild(store)
        print(f"[Milvus] rebuilt vector index after {added} approved memories")


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default="demo-a", help="Short-term checkpoint thread_id")
    parser.add_argument("--checkpoint", default="./data/real_chat_checkpoints.sqlite")
    parser.add_argument("--memory-db", default="./data/real_agent_memories.sqlite")
    parser.add_argument("--context-budget", type=int, default=400,
                        help="Approximate token budget for session messages")
    args = parser.parse_args()
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    model = ChatOllama(
        model="qwen3-vl:2b-instruct", temperature=0, num_ctx=4096
    )
    store = MemoryStore(args.memory_db)
    index = None
    try:
        index = MilvusMemoryIndex(collection="real_agent_memory_vectors")
        # This demo rebuilds the index on launch to reconcile expired/deleted
        # SQLite memories, so stale vectors cannot be returned after restart.
        index.rebuild(store)
        with SqliteSaver.from_conn_string(args.checkpoint) as checkpointer:
            graph = make_graph(model, index, checkpointer, args.context_budget)
            config = {"configurable": {"thread_id": args.thread}}
            print(f"[READY] thread={args.thread}")
            print("Commands: /remember (Qwen extraction + approval), "
                  "/history, /list, /forget MEMORY_ID, /exit")
            print("Try: I prefer Python for computer vision projects.")
            while True:
                try:
                    user_text = input("\nYou> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting.")
                    break
                if not user_text:
                    continue
                if user_text in {"/exit", "/quit"}:
                    break
                if user_text == "/history":
                    show_history(graph, config)
                    continue
                if user_text == "/list":
                    for m in store.list_active():
                        print(f"{m.id} [{m.kind}] {m.content} (source={m.source})")
                    continue
                if user_text == "/remember":
                    review_and_save(graph, config, model, store, index)
                    continue
                if user_text.startswith("/forget "):
                    memory_id = user_text[len("/forget "):].strip()
                    if store.forget(memory_id):
                        index.rebuild(store)
                        print("[FORGOTTEN] SQLite record removed and Milvus index rebuilt")
                    else:
                        print("[NOT FOUND] No memory with that ID")
                    continue
                if user_text.startswith("/"):
                    print("Unknown command. Use /remember, /history, /list, /forget ID, /exit")
                    continue
                try:
                    graph.invoke({
                        "query": user_text,
                        "messages": [{"role": "user", "content": user_text}],
                    }, config=config)
                    print("[CHECKPOINT] turn saved for thread:", args.thread)
                except Exception as e:
                    print(f"[ERROR] {type(e).__name__}: {e}")
    finally:
        if index is not None:
            index.close()
        store.close()


if __name__ == "__main__":
    run()
