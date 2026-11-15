"""Optional Qwen-assisted context summary and memory candidate proposal.

This does NOT automatically save personal memories. The LLM may hallucinate;
candidates are presented for human review, with source-turn references.
Run: python 05_memory/llm_memory.py
"""
from __future__ import annotations

import json
import re

from langchain_ollama import ChatOllama

from memory_core import compact_messages


def parse_qwen_json(raw: str) -> dict:
    """Read the first complete JSON object in a response.

    Small Qwen models sometimes append another object or prose after JSON.
    Never use greedy regex (which can eat multiple JSON objects). Reject
    malformed objects; accept the first complete object with a visible warning.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^\`\`\`(?:json)?\\s*", "", text, count=1, flags=re.I)
        text = re.sub(r"\\s*\`\`\`\\s*$", "", text, count=1)
    first = text.find("{")
    if first < 0:
        raise ValueError(f"Qwen returned no JSON object; output={raw[:400]!r}")
    try:
        obj, end = json.JSONDecoder().raw_decode(text[first:])
    except json.JSONDecodeError as exc:
        print(f"[Memory JSON error] Qwen output excerpt: {raw[:500]!r}")
        raise ValueError(f"Qwen returned invalid JSON: {exc}") from exc
    suffix = text[first + end:].strip()
    if suffix and suffix != "```":
        print(f"[Memory JSON warning] Ignoring trailing model text: {suffix[:160]!r}")
    if not isinstance(obj, dict):
        raise ValueError("Qwen JSON root must be an object")
    return obj


def propose_memories(model: ChatOllama, turns: list[dict]) -> dict:
    numbered = "\n".join(
        f"{i}. {m['role']}: {m['content']}"
        for i, m in enumerate(turns)
    )
    response = model.invoke([
        (
            "system",
            "Extract only explicitly supported NON-SENSITIVE, useful facts, "
            "events and procedures from the numbered conversation. "
            "Do not invent information. Do not store secrets. "
            "Return ONLY JSON: "
            '{"summary":"short factual summary","candidates":['
            '{"kind":"semantic|episodic|procedural","content":"...",'
            '"source_index":0}]}. '
            "Candidates must cite the exact turn index and be directly supported "
            "by that turn. If uncertain, omit the candidate.",
        ),
        ("human", numbered),
    ])
    raw = response.content if isinstance(response.content, str) else str(response.content)
    parsed = parse_qwen_json(raw)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("candidates"), list):
        raise ValueError("Invalid Qwen JSON response")
    safe_candidates = []
    for candidate in parsed["candidates"][:10]:
        if not isinstance(candidate, dict):
            continue
        ix = candidate.get("source_index")
        kind = candidate.get("kind")
        content = candidate.get("content")
        if (
            type(ix) is int and 0 <= ix < len(turns)
            and turns[ix]["role"] == "user"
            and kind in ("semantic", "episodic", "procedural")
            and isinstance(content, str) and 0 < len(content) < 500
        ):
            # This verifies source *existence*, not semantic entailment.
            safe_candidates.append({
                "kind": kind,
                "content": content,
                "source": f"user_turn_{ix}",
                "source_text": turns[ix]["content"],
            })
    return {
        "summary": str(parsed.get("summary", "")),
        "candidates_for_review": safe_candidates,
    }


def main() -> None:
    turns = [
        {"role": "user", "content": "I prefer Python for small demo projects."},
        {"role": "assistant", "content": "Understood, we can use Python."},
        {"role": "user", "content": "Yesterday I tested the LangGraph checkpoint feature."},
        {"role": "assistant", "content": "Great, the checkpoint is separate from long-term memory."},
        {"role": "user", "content": "Before releasing a generated report, always ask for approval."},
    ]
    model = ChatOllama(model="qwen3-vl:2b-instruct", temperature=0, num_ctx=4096, format="json")
    result = propose_memories(model, turns)
    print("\n[Qwen context summary]\n", result["summary"])
    print("\n[Memory candidates: NOT automatically saved]\n")
    for item in result["candidates_for_review"]:
        print(json.dumps(item, indent=2, ensure_ascii=False))
    print("\nHuman review is required before storing any candidate.")
    print("\n[Context window baseline]")
    print(compact_messages(turns, max_estimated_tokens=70))


if __name__ == "__main__":
    main()
