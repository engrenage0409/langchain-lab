"""LangGraph lab 02: run two Ollama LLM review nodes in parallel.

Both nodes analyze the same *text description*, not an actual drawing image.
Requires Ollama with qwen3-vl:2b-instruct running locally.

Run: python 02_langgraph/parallel_llm_review.py
"""
from __future__ import annotations

import operator
import time
from typing import Annotated, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


class ReviewState(TypedDict):
    drawing_description: str
    findings: Annotated[list[str], operator.add]
    report: str


# Ollama is local; two graph nodes can issue requests in the same super-step.
model = ChatOllama(
    model="qwen3-vl:2b-instruct",
    temperature=0,
    num_ctx=4096,
)


def ask_reviewer(role: str, task: str, description: str) -> str:
    print(f"[{role}] -> Qwen request", flush=True)
    t0 = time.perf_counter()
    response = model.invoke(
        [
            ("system", "You are a careful engineering drawing review assistant. "
             "The input is a TEXT description, not an image. "
             "Only report issues supported by the description. "
             "If evidence is missing, clearly say it is uncertain. "
             "Reply in one short English sentence."),
            ("human", f"Task: {task}\nDrawing description: {description}"),
        ]
    )
    text = response.content
    answer = text if isinstance(text, str) else str(text)
    print(f"[{role}] <- Qwen ({time.perf_counter()-t0:.2f}s): {answer}", flush=True)
    return answer


def symbol_review(state: ReviewState) -> dict:
    answer = ask_reviewer(
        "symbol",
        "Identify any supported symbol-annotation issue.",
        state["drawing_description"],
    )
    return {"findings": [f"[symbol] {answer}"]}


def geometry_review(state: ReviewState) -> dict:
    answer = ask_reviewer(
        "geometry",
        "Identify any supported geometry or dimension issue.",
        state["drawing_description"],
    )
    return {"findings": [f"[geometry] {answer}"]}


def make_report(state: ReviewState) -> dict:
    print(f"[report] reducer supplied {len(state['findings'])} findings", flush=True)
    # Deterministic Python report, no extra LLM call.
    return {"report": "Review results:\n" + "\n".join(state["findings"])}


def main() -> None:
    builder = StateGraph(ReviewState)
    builder.add_node("symbol", symbol_review)
    builder.add_node("geometry", geometry_review)
    builder.add_node("report", make_report)

    builder.add_edge(START, "symbol")
    builder.add_edge(START, "geometry")
    builder.add_edge(["symbol", "geometry"], "report")
    builder.add_edge("report", END)

    graph = builder.compile()
    initial: ReviewState = {
        "drawing_description": (
            "A mechanical drawing note states that a required welding symbol "
            "is missing at joint A. Dimension B is labeled 25 mm, "
            "while the stated design requirement is 30 mm. "
            "No actual drawing image is provided."
        ),
        "findings": [],
        "report": "",
    }

    print("[start] parallel LangGraph + real Ollama Qwen calls", flush=True)
    final = initial
    for snapshot in graph.stream(initial, stream_mode="values"):
        final = snapshot
        print("[state] findings =", len(snapshot["findings"]), flush=True)
    print("\n========== FINAL REPORT ==========")
    print(final["report"])
    assert len(final["findings"]) == 2, "Expected one finding per LLM node"
    print("\n[PASS] Two Qwen responses were merged by the LangGraph reducer")


if __name__ == "__main__":
    main()
