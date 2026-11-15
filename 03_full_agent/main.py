"""Run the complete LangGraph + LangChain + Milvus + approval demo.

Fresh run:
    python 03_full_agent/main.py --thread demo-001
Resume after process restart:
    python 03_full_agent/main.py --thread demo-001 --resume
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agents import AgentWorkers
from knowledge import RuleStore
from workflow import build_graph, initial_state


DEFAULT_DESCRIPTION = (
    "A mechanical drawing note explicitly says a required welding symbol "
    "is missing at joint A. Dimension B is labeled 25 mm while its design "
    "requirement is 30 mm. No drawing image is provided."
)


def prompt_approval() -> bool:
    while True:
        answer = input("\n[HUMAN APPROVAL] Release report? [yes/no]: ").strip().lower()
        if answer in ("yes", "y"):
            return True
        if answer in ("no", "n"):
            return False
        print("Please type yes or no.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default="drawing-review-001", help="Stable checkpoint thread ID")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted saved thread")
    parser.add_argument("--task", default="Review welding symbol and dimension issues")
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument("--milvus-uri", default=os.environ.get("MILVUS_URI", "./data/agent_rules.db"))
    parser.add_argument("--checkpoint", default="./data/checkpoints.sqlite")
    args = parser.parse_args()

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    if "://" not in args.milvus_uri:
        Path(args.milvus_uri).parent.mkdir(parents=True, exist_ok=True)

    store = RuleStore(uri=args.milvus_uri)
    try:
        store.seed()
        workers = AgentWorkers(store)
        config = {"configurable": {"thread_id": args.thread}}

        # SQLite persists checkpoints beyond the lifetime of this Python process.
        with SqliteSaver.from_conn_string(args.checkpoint) as saver:
            graph = build_graph(workers, checkpointer=saver)

            if args.resume:
                saved = graph.get_state(config)
                if not saved.next or "approval" not in saved.next:
                    raise RuntimeError("No pending approval interrupt for this thread")
                print(f"[Resume] restoring thread {args.thread}")
                approved = prompt_approval()
                result = graph.invoke(Command(resume=approved), config=config)
            else:
                print(f"[Start] thread={args.thread}")
                result = graph.invoke(
                    initial_state(args.task, args.description),
                    config=config,
                )
                # interrupt() stops the graph, persisting a resumable checkpoint.
                if result.get("__interrupt__"):
                    print("\n[PAUSED] LangGraph needs human approval:")
                    for item in result["__interrupt__"]:
                        print(item.value)
                    approved = prompt_approval()
                    result = graph.invoke(Command(resume=approved), config=config)

            if result.get("__interrupt__"):
                print("[PAUSED] Still waiting for approval. Run with --resume.")
                return

            print("\n========== FINAL REPORT ==========")
            print(result["report"])
    finally:
        store.close()


if __name__ == "__main__":
    main()
