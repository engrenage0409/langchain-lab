"""Offline tests for LLM-planned DAG scheduling (no Ollama / Milvus needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_full_agent"))

from dynamic_workflow import initial_state, make_graph
from planner import validate_plan


class Planner:
    def __init__(self, tasks):
        self.tasks = tasks

    def plan(self, task, description):
        return {"tasks": self.tasks}


class Workers:
    def __init__(self):
        self.calls = []

    def run(self, agent_name, task, description, context=""):
        self.calls.append((agent_name, task))
        return {
            "agent": agent_name,
            "answer": f"{agent_name}: {task}",
            "tool_used": True,
            "tool_calls": [],
            "tool_results": [],
        }


def t(id, agent, deps=()):
    return {
        "id": id,
        "agent": agent,
        "instruction": f"Inspect relevant {agent} evidence",
        "depends_on": list(deps),
    }


def test_parallel_then_serial_and_approval_resume():
    tasks = [
        t("T1", "symbol"), t("T2", "geometry"),
        t("T3", "research", ["T1", "T2"]),
        t("T4", "validate", ["T3"]),
    ]
    workers = Workers()
    graph = make_graph(workers, Planner(tasks), InMemorySaver())
    config = {"configurable": {"thread_id": "parallel-sequential"}}

    result = graph.invoke(initial_state("both then rules then verify", "TEXT"), config)
    assert "__interrupt__" in result
    assert set(result["results"]) == {"T1", "T2", "T3", "T4"}
    assert {name for name, _ in workers.calls[:2]} == {"symbol", "geometry"}
    assert [name for name, _ in workers.calls[2:]] == ["research", "validate"]
    assert graph.get_state(config).next == ("approval",)

    done = graph.invoke(Command(resume=True), config)
    assert done["approved"] is True
    assert "APPROVED" in done["report"]
    assert graph.get_state(config).next == ()


def test_single_agent_route_and_rejection():
    workers = Workers()
    graph = make_graph(workers, Planner([t("T1", "symbol")]), InMemorySaver())
    config = {"configurable": {"thread_id": "single"}}
    pause = graph.invoke(initial_state("symbol only", "TEXT"), config)
    assert "__interrupt__" in pause
    assert [name for name, _ in workers.calls] == ["symbol"]
    done = graph.invoke(Command(resume=False), config)
    assert done["approved"] is False
    assert "REJECTED" in done["report"]


@pytest.mark.parametrize("bad_plan", [
    {"tasks": []},
    {"tasks": [t("T1", "unknown")]},
    {"tasks": [t("T1", "symbol", ["T9"])]},
    {"tasks": [t("T1", "symbol", ["T2"]), t("T2", "geometry", ["T1"])]},
    {"tasks": [t("T1", "symbol"), t("T1", "geometry")]},
    {"tasks": [t("T1", "symbol", ["T1"])]},
    {"tasks": [t("T1", "symbol", ["T2"]), t("T2", "geometry", ["T3"]),
               t("T3", "research", ["T1"])]},
])
def test_invalid_dag_is_rejected(bad_plan):
    with pytest.raises(ValueError):
        validate_plan(bad_plan)


def test_independent_tasks_can_run_without_dependencies():
    tasks = [t("T1", "geometry"), t("T2", "symbol"), t("T3", "research")]
    workers = Workers()
    graph = make_graph(workers, Planner(tasks), InMemorySaver())
    config = {"configurable": {"thread_id": "independent"}}
    result = graph.invoke(initial_state("independent checks", "TEXT"), config)
    assert "__interrupt__" in result
    assert len(result["results"]) == 3
    assert {agent for agent, _ in workers.calls} == {"symbol", "geometry", "research"}
