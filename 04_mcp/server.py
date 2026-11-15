"""Small MCP Server exposing Tools, Resources and Prompts.

Uses the official MCP Python SDK v2 (MCPServer).
The server writes no data and has no access to personal files.
"""
from mcp.server import MCPServer

mcp = MCPServer("engineering-review-demo")

RULES = {
    "WELD-001": "Required welding symbols must appear at their specified joints.",
    "DIM-001": "A dimension label must match the explicitly stated design requirement.",
}


@mcp.tool()
def lookup_rule(rule_id: str) -> dict:
    """Look up a demo engineering rule by its rule ID."""
    if rule_id not in RULES:
        return {"found": False, "rule_id": rule_id, "text": ""}
    return {"found": True, "rule_id": rule_id, "text": RULES[rule_id]}


@mcp.tool()
def compare_dimension(measured_mm: float, required_mm: float) -> dict:
    """Compare the dimension shown on a drawing with the design requirement."""
    delta = round(measured_mm - required_mm, 4)
    return {"matches": abs(delta) < 1e-6, "delta_mm": delta}


@mcp.resource("rules://catalog")
def rule_catalog() -> str:
    """Return all example rule IDs and descriptions as static reference data."""
    return "\n".join(f"{key}: {value}" for key, value in RULES.items())


@mcp.prompt()
def review_instruction(topic: str) -> str:
    """Create a reusable review instruction for a given topic."""
    return (
        f"Review the following {topic} issue using only provided evidence. "
        "Use MCP tools when needed and do not invent engineering rules."
    )


if __name__ == "__main__":
    # For stdio, keep stdout reserved for MCP protocol messages.
    mcp.run()
