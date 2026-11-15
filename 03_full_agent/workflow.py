"""LangGraph orchestration: parallel -> sequential -> LLM route -> approval.

Nodes can be injected with fake workers for offline unit tests.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


class WorkflowState(TypedDict):
    task: str
    description: str
    parallel_results: Annotated[list[dict], operator.add]
    retrieved: dict
    validation: dict
    chosen_agent: str
    final_review: dict
    approved: bool
    report: str


class WorkerProtocol:
    def run(self, agent_name: str, task: str, description: str, context: str = "") -> dict: ...
    def choose(self, task: str, description: str, findings: str) -> str: ...


def build_graph(workers: WorkerProtocol, checkpointer: Any):
    builder = StateGraph(WorkflowState)

    def symbol_parallel(state: WorkflowState) -> dict:
        result = workers.run("symbol", state["task"], state["description"])
        return {"parallel_results": [result]}

    def geometry_parallel(state: WorkflowState) -> dict:
        result = workers.run("geometry", state["task"], state["description"])
        return {"parallel_results": [result]}

    def research_sequential(state: WorkflowState) -> dict:
        prior = str(state["parallel_results"])
        return {
            "retrieved": workers.run(
                "research", state["task"], state["description"],
                context=f"Parallel specialist findings: {prior}",
            )
        }

    def validate_sequential(state: WorkflowState) -> dict:
        context = (
            f"Specialist findings: {state['parallel_results']}\n"
            f"Retrieved rules: {state['retrieved']}"
        )
        return {
            "validation": workers.run(
                "validate", state["task"], state["description"], context=context
            )
        }

    def supervisor(state: WorkflowState) -> dict:
        findings = (
            f"Parallel: {state['parallel_results']}\n"
            f"Research: {state['retrieved']}\n"
            f"Validation: {state['validation']}"
        )
        chosen = workers.choose(state["task"], state["description"], findings)
        if chosen not in ("symbol", "geometry"):
            raise ValueError(f"Invalid agent: {chosen}")
        return {"chosen_agent": chosen}

    def route_specialist(state: WorkflowState) -> str:
        return state["chosen_agent"]

    def final_symbol(state: WorkflowState) -> dict:
        context = f"Previous work: {state['parallel_results']}; {state['retrieved']}; {state['validation']}"
        return {"final_review": workers.run("symbol", "Final targeted review: " + state["task"], state["description"], context)}

    def final_geometry(state: WorkflowState) -> dict:
        context = f"Previous work: {state['parallel_results']}; {state['retrieved']}; {state['validation']}"
        return {"final_review": workers.run("geometry", "Final targeted review: " + state["task"], state["description"], context)}

    def approval(state: WorkflowState) -> dict:
        decision = interrupt({
            "question": "Approve this report for release?",
            "task": state["task"],
            "parallel_results": state["parallel_results"],
            "retrieved": state["retrieved"],
            "validation": state["validation"],
            "selected_specialist": state["chosen_agent"],
            "final_review": state["final_review"],
        })
        # Only accept actual boolean approval, never arbitrary truthy strings.
        if not isinstance(decision, bool):
            raise ValueError("Approval resume value must be boolean")
        return {"approved": decision}

    def approval_route(state: WorkflowState) -> str:
        return "approved" if state["approved"] else "rejected"

    def approved_report(state: WorkflowState) -> dict:
        lines = [
            "# Engineering Drawing Review — APPROVED",
            f"Task: {state['task']}",
            "",
            "## Parallel specialist findings",
        ]
        for result in state["parallel_results"]:
            lines.append(f"- {result['agent']}: {result['answer']} (tool_used={result['tool_used']})")
        lines += [
            "",
            "## Sequential Milvus retrieval",
            str(state["retrieved"]["answer"]),
            "",
            "## Sequential evidence check",
            str(state["validation"]["answer"]),
            "",
            f"## Supervisor choice: {state['chosen_agent']}",
            str(state["final_review"]["answer"]),
            "",
            "Note: This demo analyzes a TEXT description, not a DWG/PDF image. "
            "Approval is a human decision; it is not an automatic correctness guarantee.",
        ]
        return {"report": "\n".join(lines)}

    def rejected_report(state: WorkflowState) -> dict:
        return {"report": "REJECTED: the human reviewer did not approve this report. No release was performed."}

    for name, handler in [
        ("symbol_parallel", symbol_parallel),
        ("geometry_parallel", geometry_parallel),
        ("research", research_sequential),
        ("validate", validate_sequential),
        ("supervisor", supervisor),
        ("final_symbol", final_symbol),
        ("final_geometry", final_geometry),
        ("approval", approval),
        ("approved_report", approved_report),
        ("rejected_report", rejected_report),
    ]:
        builder.add_node(name, handler)

    # Phase 1: real parallel fan-out and reducer fan-in
    builder.add_edge(START, "symbol_parallel")
    builder.add_edge(START, "geometry_parallel")
    builder.add_edge(["symbol_parallel", "geometry_parallel"], "research")

    # Phase 2: strict sequential researcher -> evidence validator
    builder.add_edge("research", "validate")

    # Phase 3: an LLM supervisor dynamically chooses ONE final specialist
    builder.add_edge("validate", "supervisor")
    builder.add_conditional_edges(
        "supervisor", route_specialist,
        {"symbol": "final_symbol", "geometry": "final_geometry"},
    )
    builder.add_edge("final_symbol", "approval")
    builder.add_edge("final_geometry", "approval")

    # Always stop for human review before report release.
    builder.add_conditional_edges(
        "approval", approval_route,
        {"approved": "approved_report", "rejected": "rejected_report"},
    )
    builder.add_edge("approved_report", END)
    builder.add_edge("rejected_report", END)

    return builder.compile(checkpointer=checkpointer)


def initial_state(task: str, description: str) -> WorkflowState:
    return {
        "task": task,
        "description": description,
        "parallel_results": [],
        "retrieved": {},
        "validation": {},
        "chosen_agent": "",
        "final_review": {},
        "approved": False,
        "report": "",
    }
