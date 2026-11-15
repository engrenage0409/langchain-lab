"""Run a Qwen-planned LangGraph DAG with Milvus tools and human approval.

New run:
  python 03_full_agent/run_dynamic.py --thread dag-001

Resume a paused run:
  python 03_full_agent/run_dynamic.py --thread dag-001 --resume

Use --fake for an offline demonstration without Ollama or Milvus.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from dynamic_workflow import initial_state, make_graph
from planner import QwenPlanner


DEFAULT_REQUEST = (
    "First independently review the welding-symbol annotation and dimensions; "
    "then retrieve relevant engineering rules; "
    "then verify the findings against evidence. "
    "Choose which specialist to use for an additional focused review if needed."
)
DEFAULT_DESCRIPTION = (
    "A drawing TEXT note says the required welding symbol is missing at joint A. "
    "Dimension B reads 25 mm, whereas the stated design requirement is 30 mm. "
    "No actual drawing image is available."
)


class FakePlanner:
    """Deterministic example plan that exercises concurrency and dependencies."""

    def plan(self, request: str, description: str) -> dict:
        return {
            "tasks": [
                {"id": "T1", "agent": "symbol", "instruction": "Inspect symbol notes", "depends_on": []},
                {"id": "T2", "agent": "geometry", "instruction": "Inspect dimensions", "depends_on": []},
                {"id": "T3", "agent": "research", "instruction": "Find relevant rules", "depends_on": ["T1", "T2"]},
                {"id": "T4", "agent": "validate", "instruction": "Check evidence coverage", "depends_on": ["T3"]},
                {"id": "T5", "agent": "geometry", "instruction": "Focused final check", "depends_on": ["T4"]},
            ]
        }


class FakeWorkers:
    def run(self, agent_name: str, task: str, description: str, context: str = "") -> dict:
        return {
            "agent": agent_name,
            "answer": f"[FAKE] {agent_name}: {task}",
            "tool_used": True,
            "tool_calls": [{"name": "fake_tool", "args": {}}],
            "tool_results": ["[FAKE] This is not real retrieved evidence"],
        }


def human_approval() -> bool:
    while True:
        answer = input("\n[APPROVAL] Publish report? yes/no: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please type yes or no.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default="dynamic-demo-001")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fake", action="store_true", help="Offline orchestration demo (no LLM/Milvus)")
    parser.add_argument("--task", default=DEFAULT_REQUEST)
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument("--checkpoint", default="./data/dynamic_checkpoints.sqlite")
    parser.add_argument("--milvus-uri", default=os.getenv("MILVUS_URI", "./data/agent_rules.db"))
    args = parser.parse_args()

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    if args.fake:
        planner = FakePlanner()
        workers = FakeWorkers()
        store = None
    else:
        # Imports only when real services are requested.
        from agents import AgentWorkers
        from knowledge import RuleStore

        if "://" not in args.milvus_uri:
            Path(args.milvus_uri).parent.mkdir(parents=True, exist_ok=True)
        store = RuleStore(uri=args.milvus_uri)
        try:
            store.seed()
            workers = AgentWorkers(store)
            planner = QwenPlanner()
        except BaseException:
            store.close()
            raise

    try:
        config = {"configurable": {"thread_id": args.thread}}
        with SqliteSaver.from_conn_string(args.checkpoint) as saver:
            graph = make_graph(workers, planner, saver)
            saved = graph.get_state(config)

            if args.resume:
                if "approval" not in saved.next:
                    raise RuntimeError(
                        "No saved approval interrupt for this thread. "
                        "Use --thread with the SAME ID as the paused run."
                    )
                result = graph.invoke(
                    Command(resume=human_approval()),
                    config={**config, "recursion_limit": 50},
                )
            else:
                if saved.values:
                    raise RuntimeError(
                        "Thread ID already exists. Use a new --thread for a new run, "
                        "or --resume to approve an interrupted run."
                    )
                result = graph.invoke(
                    initial_state(args.task, args.description),
                    config={**config, "recursion_limit": 50},
                )
                if result.get("__interrupt__"):
                    print("\n[PAUSED] Review the proposed DAG and results:")
                    for entry in result["__interrupt__"]:
                        print(entry.value)
                    print("\nTip: terminate the script and rerun with --resume to test checkpoint recovery.")
                    result = graph.invoke(
                        Command(resume=human_approval()),
                        config={**config, "recursion_limit": 50},
                    )

            if result.get("__interrupt__"):
                print("[PAUSED] Graph is waiting for approval.")
                return
            print("\n================ FINAL REPORT ================")
            print(result["report"])
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    main()
