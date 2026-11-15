"""Offline graph tests: no Ollama or Milvus required.

Run: python -m pytest -q tests/test_full_graph.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_full_agent"))

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from workflow import build_graph, initial_state


class FakeWorkers:
    def __init__(self, decision: str = "geometry"):
        self.calls = []
        self.decision = decision

    def run(self, agent_name, task, description, context=""):
        self.calls.append(agent_name)
        return {
            "agent": agent_name,
            "answer": f"result from {agent_name}",
            "tool_used": True,
            "tool_calls": [{"name": "fake_tool"}],
            "tool_results": ["fake evidence"],
        }

    def choose(self, task, description, findings):
        self.calls.append("supervisor")
        return self.decision


def execute_until_approval(decision="geometry", thread="test-001"):
    workers = FakeWorkers(decision)
    graph = build_graph(workers, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": thread}}
    result = graph.invoke(initial_state("Review both types", "symbol missing; 25 mm vs 30 mm"), config)
    assert "__interrupt__" in result
    assert graph.get_state(config).next == ("approval",)
    return workers, graph, config


def test_parallel_sequential_route_and_approve():
    workers, graph, config = execute_until_approval()
    # Parallel order is deliberately not fixed.
    assert set(workers.calls[:2]) == {"symbol", "geometry"}
    assert workers.calls[2:] == ["research", "validate", "supervisor", "geometry"]
    completed = graph.invoke(Command(resume=True), config)
    assert completed["approved"] is True
    assert len(completed["parallel_results"]) == 2
    assert completed["chosen_agent"] == "geometry"
    assert "APPROVED" in completed["report"]
    assert graph.get_state(config).next == ()


def test_symbol_route_and_reject():
    workers, graph, config = execute_until_approval("symbol", "test-002")
    assert workers.calls[-1] == "symbol"
    completed = graph.invoke(Command(resume=False), config)
    assert completed["approved"] is False
    assert completed["chosen_agent"] == "symbol"
    assert "REJECTED" in completed["report"]
    assert graph.get_state(config).next == ()
