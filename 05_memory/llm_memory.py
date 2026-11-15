"""Optional Qwen-assisted context summary and memory candidate proposal.

This does NOT automatically save personal memories. The LLM may hallucinate;
candidates are presented for human review, with source-turn references.
Run: python 05_memory/llm_memory.py
"""
from __future__ import annotations

import json

from langchain_ollama import ChatOllama

from memory_core import compact_messages


def parse_qwen_json(raw: str) -> dict:
    """Read the first complete JSON object in a response.

    Small Qwen models sometimes append another object or prose after JSON.
    Never use greedy regex (which can eat multiple JSON objects). Reject
    malformed objects; accept the first complete object with a visible warning.
    """
    text = raw.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].lstrip()
    elif text.startswith("```"):
        text = text[len("```"):].lstrip()
    if text.endswith("```"):
        text = text[:-3].rstrip()
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


def validate_candidates(parsed: dict, turns: list[dict]) -> list[dict]:
    """Check that proposed memory has a real user-turn provenance."""
    items = parsed.get("candidates", [])
    if not isinstance(items, list):
        print("[Memory extraction] candidates was not a list:", type(items).__name__)
        return []
    print(f"[Memory extraction] Qwen proposed {len(items)} raw candidate(s)")
    accepted = []
    for candidate in items[:10]:
        if not isinstance(candidate, dict):
            print("[Memory rejected] Not a JSON object:", repr(candidate)[:150])
            continue
        ix = candidate.get("source_index")
        if isinstance(ix, str) and ix.strip().isdigit():
            ix = int(ix.strip())
        kind = str(candidate.get("kind", "")).strip().lower()
        content = candidate.get("content")
        if (
            type(ix) is int and 0 <= ix < len(turns)
            and turns[ix].get("role") == "user"
            and kind in ("semantic", "episodic", "procedural")
            and isinstance(content, str) and 0 < len(content.strip()) < 500
        ):
            accepted.append({
                "kind": kind, "content": content.strip(),
                "source": f"user_turn_{ix}", "source_text": turns[ix]["content"],
            })
        else:
            print("[Memory rejected] Invalid type/index/source:",
                  repr(candidate)[:300])
    return accepted


def propose_memories(model: ChatOllama, turns: list[dict]) -> dict:
    user_turns = [(i, m["content"]) for i, m in enumerate(turns)
                  if m.get("role") == "user" and m.get("content", "").strip()]
    numbered = "\\n".join(f"{i}. user: {content}" for i, content in user_turns)
    instruction = (
        "You extract long-term memory candidates from USER statements. "
        "Return a single JSON object with exactly two keys: summary and candidates. "
        "Each candidate MUST have kind (semantic, episodic, or procedural), "
        "content (a short faithful statement), and source_index (the original "
        "number of the USER turn). Extract explicit non-sensitive preferences, "
        "completed events, and standing workflows. Do NOT save passwords, "
        "sensitive information, or assistant-generated claims. "
        "You may extract multiple candidates from different user turns. "
        "Example input: '2. user: I prefer Python.' "
        "Example output: "
        '{"summary":"Prefers Python.","candidates":'
        '[{"kind":"semantic","content":"Prefers Python.","source_index":2}]}. '
        "Never cite a turn number that does not appear in the input."
    )
    if not user_turns:
        return {"summary": "", "candidates_for_review": []}

    def ask(extra: str) -> tuple[dict, list[dict]]:
        response = model.invoke([
            ("system", instruction + extra),
            ("human", "User turns with ORIGINAL indices:\\n" + numbered),
        ])
        raw = response.content if isinstance(response.content, str) else str(response.content)
        parsed = parse_qwen_json(raw)
        return parsed, validate_candidates(parsed, turns)

    parsed, candidates = ask("")
    if not candidates:
        # A small model may summarize correctly but omit or misformat the array.
        # Retry once; do not invent or silently save fallback memories.
        print("[Memory extraction] No valid candidates; retrying Qwen once.")
        parsed_retry, candidates = ask(
            " Focus on extracting at least one grounded candidate IF the user "
            "explicitly stated a preference, a completed event, or a rule. "
            "Use an empty candidates array only when no suitable statement exists."
        )
        if candidates:
            parsed = parsed_retry
    return {
        "summary": str(parsed.get("summary", "")),
        "candidates_for_review": candidates,
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
