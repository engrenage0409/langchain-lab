"""Dynamic LangGraph DAG runner: planner -> ready-task fanout -> join -> approval.

The LLM generates task dependencies; Python validates them and safely schedules
all ready tasks concurrently using Send. No fixed parallel/serial layout.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from planner import PlannerProtocol, validate_plan


class DynamicState(TypedDict):
    request: str
    description: str
    plan: dict
    results: Annotated[dict[str, dict], operator.or_]
    approved: bool
    report: str


class WorkerState(TypedDict):
    task: dict
    request: str
    description: str
    prior: dict[str, dict]


def make_graph(workers: Any, planner: PlannerProtocol, checkpointer: Any):
    builder = StateGraph(DynamicState)

    def plan_node(state: DynamicState) -> dict:
        proposed = planner.plan(state["request"], state["description"])
        plan = validate_plan(proposed)
        print("[Plan] execution DAG:", flush=True)
        for item in plan["tasks"]:
            print(
                f"  {item['id']}: agent={item['agent']} "
                f"depends_on={item['depends_on']} instruction={item['instruction']}",
                flush=True,
            )
        return {"plan": plan}

    def scheduler(state: DynamicState) -> dict:
        # Deliberately does not update State; only routes after dependencies resolve.
        return {}

    def dispatch(state: DynamicState):
        tasks = state["plan"]["tasks"]
        completed = set(state["results"])
        remaining = [t for t in tasks if t["id"] not in completed]
        if not remaining:
            return "approval"

        ready = [
            t for t in remaining
            if set(t["depends_on"]).issubset(completed)
        ]
        if not ready:
            raise RuntimeError("No ready tasks: invalid DAG or missing result")

        print(f"[Scheduler] ready: {[t['id'] for t in ready]}", flush=True)
        return [
            Send("worker", {
                "task": t,
                "request": state["request"],
                "description": state["description"],
                "prior": {
                    dep: state["results"][dep]
                    for dep in t["depends_on"]
                },
            })
            for t in ready
        ]

    def worker(state: WorkerState) -> dict:
        task = state["task"]
        tid = task["id"]
        print(f"[Worker {tid}] START agent={task['agent']}", flush=True)
        result = workers.run(
            agent_name=task["agent"],
            task=task["instruction"],
            description=state["description"],
            context=(
                "Original user request: " + state["request"]
                + "\nResults from declared dependency tasks: " + str(state["prior"])
            ),
        )
        print(f"[Worker {tid}] DONE", flush=True)
        # Unique task IDs and the dict reducer allow concurrent updates.
        return {"results": {tid: result}}

    def approve(state: DynamicState) -> dict:
        answer = interrupt({
            "question": "Approve release of the generated review report?",
            "request": state["request"],
            "plan": state["plan"],
            "results": state["results"],
        })
        if type(answer) is not bool:
            raise ValueError("Approval must be a boolean (True/False)")
        return {"approved": answer}

    def approval_route(state: DynamicState) -> str:
        return "approved_report" if state["approved"] else "rejected_report"

    def approved_report(state: DynamicState) -> dict:
        lines = [
            "# APPROVED — Dynamic Multi-Agent Review",
            f"Request: {state['request']}",
            "Note: The demo reviews user-provided TEXT, not an actual drawing image.",
            "",
            "## LLM-generated execution plan",
        ]
        for task in state["plan"]["tasks"]:
            lines.append(
                f"- {task['id']} / {task['agent']} / depends_on={task['depends_on']}: "
                + task["instruction"]
            )
        lines.extend(["", "## Task results"])
        for task in state["plan"]["tasks"]:
            result = state["results"][task["id"]]
            lines.append(
                f"### {task['id']} — {task['agent']} (tool_used={result['tool_used']})"
            )
            lines.append(result["answer"])
            if result.get("tool_results"):
                lines.append("Tool evidence:")
                lines.extend(f"- {x}" for x in result["tool_results"])
        lines.append(
            "\nHuman approval confirms release only; it does not establish factual correctness."
        )
        return {"report": "\n".join(lines)}

    def rejected_report(state: DynamicState) -> dict:
        return {
            "report": "REJECTED — The human reviewer did not approve releasing this report."
        }

    builder.add_node("planner", plan_node)
    builder.add_node("scheduler", scheduler)
    builder.add_node("worker", worker)
    builder.add_node("approval", approve)
    builder.add_node("approved_report", approved_report)
    builder.add_node("rejected_report", rejected_report)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "scheduler")
    builder.add_conditional_edges("scheduler", dispatch)
    builder.add_edge("worker", "scheduler")
    builder.add_conditional_edges(
        "approval", approval_route,
        {
            "approved_report": "approved_report",
            "rejected_report": "rejected_report",
        },
    )
    builder.add_edge("approved_report", END)
    builder.add_edge("rejected_report", END)

    return builder.compile(checkpointer=checkpointer)


def initial_state(request: str, description: str) -> DynamicState:
    return {
        "request": request,
        "description": description,
        "plan": {},
        "results": {},
        "approved": False,
        "report": "",
    }
