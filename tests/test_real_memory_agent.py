"""Offline plumbing tests of the real chat graph using fake model/index.

The actual Qwen + Milvus run still requires a local integration test.
"""
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1] / "05_memory"
sys.path.insert(0, str(BASE))

from langgraph.checkpoint.memory import InMemorySaver
from real_memory_agent import make_graph


class FakeReply:
    content = "Simulated Qwen answer"


class FakeModel:
    def invoke(self, messages):
        assert any("Relevant long-term memories" in m["content"] for m in messages)
        return FakeReply()


class FakeIndex:
    def __init__(self):
        self.queries = []

    def search(self, query, top_k=3):
        self.queries.append(query)
        return [{
            "kind": "semantic", "content": "Prefers Python",
            "memory_id": "example-id", "score": 0.83,
        }]


def test_short_term_checkpoint_and_long_term_retrieval_are_distinct():
    index = FakeIndex()
    graph = make_graph(FakeModel(), index, InMemorySaver())
    a = {"configurable": {"thread_id": "a"}}
    b = {"configurable": {"thread_id": "b"}}

    graph.invoke({"query": "First", "messages": [{"role": "user", "content": "First"}]}, a)
    graph.invoke({"query": "Second", "messages": [{"role": "user", "content": "Second"}]}, a)
    graph.invoke({"query": "Other", "messages": [{"role": "user", "content": "Other"}]}, b)

    turns_a = graph.get_state(a).values["messages"]
    turns_b = graph.get_state(b).values["messages"]
    assert [m["role"] for m in turns_a] == ["user", "assistant", "user", "assistant"]
    assert [m["role"] for m in turns_b] == ["user", "assistant"]
    assert index.queries == ["First", "Second", "Other"]
