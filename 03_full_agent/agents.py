"""Real LangChain agents and executable tools used by LangGraph nodes."""
from __future__ import annotations

import json
import re
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

from knowledge import RuleStore


def make_model() -> ChatOllama:
    return ChatOllama(
        model="qwen3-vl:2b-instruct",
        temperature=0,
        num_ctx=8192,
    )


class AgentWorkers:
    def __init__(self, store: RuleStore, model: ChatOllama | None = None):
        self.store = store
        self.model = model or make_model()

        @tool
        def inspect_symbols(description: str) -> str:
            """Extract explicit welding-symbol facts from a drawing TEXT description. Not image recognition."""
            lines = re.split(r"(?<=[.!?])\s+", description)
            matched = [line for line in lines if any(w in line.lower() for w in ("symbol", "weld", "焊"))]
            return json.dumps({"facts": matched, "source": "user_provided_description"}, ensure_ascii=False)

        @tool
        def inspect_dimensions(description: str) -> str:
            """Extract explicit dimensions and requirement statements from a drawing TEXT description."""
            lines = re.split(r"(?<=[.!?])\s+", description)
            matched = [line for line in lines if any(w in line.lower() for w in ("dimension", " mm", "size", "尺寸"))]
            return json.dumps({"facts": matched, "source": "user_provided_description"}, ensure_ascii=False)

        @tool
        def search_policy(query: str) -> str:
            """Search Milvus for relevant engineering rules, including their rule IDs."""
            return json.dumps(self.store.search(query), ensure_ascii=False)

        @tool
        def validate_evidence(claim: str, evidence: str) -> str:
            """Check whether claim and cited evidence are nonempty. Does not certify factual correctness."""
            claim = claim.strip()
            evidence = evidence.strip()
            status = "incomplete" if not claim or not evidence else "requires_human_verification"
            return json.dumps({"status": status, "claim": claim, "evidence": evidence}, ensure_ascii=False)

        self.tools = {
            "symbol": [inspect_symbols],
            "geometry": [inspect_dimensions],
            "research": [search_policy],
            "validate": [validate_evidence],
        }
        self.prompts = {
            "symbol": "You are the symbol specialist. Use inspect_symbols on the provided description. Discuss ONLY symbol defects. Do not infer image contents.",
            "geometry": "You are the geometry specialist. Use inspect_dimensions on the provided description. Discuss ONLY dimensional defects. Do not infer image contents.",
            "research": "You are the RAG researcher. Use search_policy to retrieve relevant rules. Include exact returned rule IDs. Do not fabricate sources.",
            "validate": "You are the evidence validator. Use validate_evidence on a claim and its supporting evidence. State limitations and do not declare a finding verified without proof.",
        }
        self.agents = {
            name: create_agent(model=self.model, tools=tools, system_prompt=self.prompts[name])
            for name, tools in self.tools.items()
        }

    def run(self, agent_name: str, task: str, description: str, context: str = "") -> dict[str, Any]:
        agent = self.agents[agent_name]
        user_message = (
            f"Task: {task}\n"
            f"Drawing description (TEXT ONLY): {description}\n"
            f"Additional context:\n{context}\n"
            "Use your available tool before answering. Keep your answer short and grounded."
        )
        print(f"[Agent:{agent_name}] start", flush=True)
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={"recursion_limit": 8},
        )
        msgs = response["messages"]
        tool_messages = [m for m in msgs if m.type == "tool"]
        calls = [
            {"name": tc["name"], "args": tc["args"]}
            for m in msgs if m.type == "ai"
            for tc in (getattr(m, "tool_calls", None) or [])
        ]
        final = msgs[-1].content
        answer = final if isinstance(final, str) else str(final)
        print(f"[Agent:{agent_name}] tool_calls={len(calls)} tool_results={len(tool_messages)}", flush=True)
        if not tool_messages:
            print(f"[WARN] {agent_name}: Qwen did NOT execute a tool", flush=True)
        return {
            "agent": agent_name,
            "answer": answer,
            "tool_used": bool(tool_messages),
            "tool_calls": calls,
            "tool_results": [str(m.content) for m in tool_messages],
        }

    def choose(self, task: str, description: str, findings: str) -> str:
        """Supervisor LLM chooses exactly one stage-3 specialist."""
        response = self.model.invoke([
            (
                "system",
                "You are an agent supervisor. Choose ONE final review specialist. "
                "symbol = welding symbol or annotation uncertainty. "
                "geometry = dimensions, measurements, size uncertainty. "
                "Based on the task and intermediate findings, return exactly ONE word: symbol or geometry. "
                "No punctuation, no other text.",
            ),
            ("human", f"Task: {task}\nDescription: {description}\nFindings: {findings}"),
        ])
        answer = str(response.content).strip().lower()
        print(f"[Supervisor] raw={answer!r}", flush=True)
        if answer not in {"symbol", "geometry"}:
            raise ValueError(f"Supervisor did not return a valid route: {answer!r}")
        return answer
