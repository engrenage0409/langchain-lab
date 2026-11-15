"""LangGraph lab 01: parallel nodes, reducer, and fan-in.

No LLM or Ollama required. Run:
    python 02_langgraph/parallel_review.py
"""
from __future__ import annotations

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class ReviewState(TypedDict):
    drawing: str
    errors: Annotated[list[str], operator.add]
    report: str


def symbol_review(state: ReviewState) -> dict:
    print(f"[symbol] checking {state['drawing']}", flush=True)
    time.sleep(0.3)
    return {"errors": ["Missing welding symbol", "Incorrect symbol label"]}


def geometry_review(state: ReviewState) -> dict:
    print(f"[geometry] checking {state['drawing']}", flush=True)
    time.sleep(0.3)
    return {"errors": ["Incorrect dimension"]}


def report(state: ReviewState) -> dict:
    print(f"[report] sees {len(state['errors'])} merged errors", flush=True)
    return {
        "report": (
            f"Drawing: {state['drawing']}\n"
            f"Errors: {len(state['errors'])}\n"
            + "\n".join(f"- {item}" for item in state["errors"])
        )
    }


def build_graph():
    graph = StateGraph(ReviewState)
    graph.add_node("symbol", symbol_review)
    graph.add_node("geometry", geometry_review)
    graph.add_node("report", report)

    # Both nodes run in the same graph super-step.
    graph.add_edge(START, "symbol")
    graph.add_edge(START, "geometry")

    # A waiting edge: report starts after BOTH reviews finish.
    graph.add_edge(["symbol", "geometry"], "report")
    graph.add_edge("report", END)
    return graph.compile()


def main() -> None:
    graph = build_graph()
    initial_state: ReviewState = {
        "drawing": "DWG-001",
        "errors": [],
        "report": "",
    }

    final_state = initial_state
    started = time.perf_counter()
    for state in graph.stream(initial_state, stream_mode="values"):
        print("\n[STATE SNAPSHOT]", state, flush=True)
        final_state = state

    print("\n========== FINAL REPORT ==========")
    print(final_state["report"])
    assert len(final_state["errors"]) == 3, "Expected 3 merged errors"
    print(f"\n[PASS] reducer merged 3 errors in {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
