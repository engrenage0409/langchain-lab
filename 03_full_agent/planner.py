"""LLM-driven task DAG planning with strict schema and dependency validation."""
from __future__ import annotations

import json
import re
from typing import Any, Protocol

from langchain_ollama import ChatOllama


AGENTS = {"symbol", "geometry", "research", "validate"}
MAX_TASKS = 8
MAX_DEPENDENCIES = 7


class PlannerProtocol(Protocol):
    def plan(self, task: str, description: str) -> dict[str, Any]: ...


def validate_plan(plan: Any) -> dict[str, Any]:
    """Fail closed: allowlisted agents, unique IDs, valid dependency DAG, bounded size."""
    if not isinstance(plan, dict) or not isinstance(plan.get("tasks"), list):
        raise ValueError("Plan must be an object containing a tasks list")
    tasks = plan["tasks"]
    if not 1 <= len(tasks) <= MAX_TASKS:
        raise ValueError(f"Plan requires between 1 and {MAX_TASKS} tasks")

    ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for item in tasks:
        if not isinstance(item, dict):
            raise ValueError("Every task must be an object")
        task_id = item.get("id")
        agent = item.get("agent")
        instruction = item.get("instruction")
        deps = item.get("depends_on")
        if not isinstance(task_id, str) or not re.fullmatch(r"T[1-9][0-9]?", task_id):
            raise ValueError(f"Invalid task ID: {task_id!r} (expected T1, T2, ...)")
        if task_id in ids:
            raise ValueError(f"Duplicate task ID: {task_id}")
        if agent not in AGENTS:
            raise ValueError(f"Unknown agent {agent!r}")
        if not isinstance(instruction, str) or not 5 <= len(instruction.strip()) <= 500:
            raise ValueError(f"Task {task_id} has an invalid instruction")
        if (
            not isinstance(deps, list)
            or len(deps) > MAX_DEPENDENCIES
            or any(not isinstance(dep, str) for dep in deps)
            or len(set(deps)) != len(deps)
        ):
            raise ValueError(f"Task {task_id} has invalid dependencies")
        ids.add(task_id)
        validated.append({
            "id": task_id,
            "agent": agent,
            "instruction": instruction.strip(),
            "depends_on": deps,
        })

    graph = {item["id"]: item["depends_on"] for item in validated}
    for task_id, deps in graph.items():
        if any(dep not in graph for dep in deps):
            raise ValueError(f"Unknown dependency in {task_id}: {deps}")
        if task_id in deps:
            raise ValueError(f"Task {task_id} depends on itself")

    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError(f"Cyclic dependency detected at {task_id}")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in graph[task_id]:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in graph:
        visit(task_id)

    return {"tasks": validated}


def extract_json(text: str) -> dict[str, Any]:
    """Accept plain JSON or a single fenced JSON block, not arbitrary prose."""
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if not match:
            raise ValueError("Response is not a single JSON code block")
        text = match.group(1)
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("JSON must be an object")
    return result


class QwenPlanner:
    def __init__(self, model: ChatOllama | None = None):
        self.model = model or ChatOllama(
            model="qwen3-vl:2b-instruct", temperature=0, num_ctx=4096,
            format="json",
        )

    def plan(self, task: str, description: str) -> dict[str, Any]:
        system = (
            "You plan a DAG of tasks for an engineering drawing TEXT review. "
            "Available agents: symbol, geometry, research, validate. "
            "Choose ANY subset and ordering. For independent tasks, depends_on=[] "
            "so they can run in parallel. For dependent tasks, list IDs in depends_on. "
            "A single relevant task is valid. Use research if a rule lookup is requested, "
            "and validate after its supporting evidence exists. "
            "You may use the same agent multiple times with distinct task IDs. "
            "Do not create dependencies that form a cycle. "
            "Do not assume a visual drawing image exists. "
            "Produce exactly one JSON object, no markdown: "
            '{"tasks":[{"id":"T1","agent":"symbol",'
            '"instruction":"Inspect the welding symbol notes",'
            '"depends_on":[]}]}. '
            "IDs must be T1, T2, etc. Use at most 8 tasks."
        )
        last_error = ""
        for attempt in range(2):
            messages = [
                ("system", system),
                ("human", f"User task: {task}\nDrawing TEXT description: {description}"
                 + (f"\nYour previous plan was invalid: {last_error}. Correct the JSON." if last_error else "")),
            ]
            response = self.model.invoke(messages)
            raw = response.content if isinstance(response.content, str) else str(response.content)
            print(f"[Planner] attempt {attempt + 1}: {raw}", flush=True)
            try:
                return validate_plan(extract_json(raw))
            except (ValueError, json.JSONDecodeError) as error:
                last_error = str(error)
        raise ValueError(f"LLM could not generate a valid plan after 2 attempts: {last_error}")
