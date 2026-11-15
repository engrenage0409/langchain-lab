"""A REAL LangGraph node that calls an HTTP MCP server.

No Qwen or Milvus needed. Launch http_server.py in a separate terminal.
"""
import asyncio
import os
from typing import TypedDict

from langgraph.graph import START, END, StateGraph
from mcp import Client


class DimensionState(TypedDict):
    measured_mm: float
    required_mm: float
    rule_id: str
    dimension_result: str
    rule_result: str
    report: str


async def mcp_dimension(state: DimensionState):
    url = os.environ.get("MCP_URL", "http://127.0.0.1:8000/mcp")
    async with Client(url) as client:
        response = await client.call_tool(
            "compare_dimension",
            {"measured_mm": state["measured_mm"], "required_mm": state["required_mm"]},
        )
    return {"dimension_result": str(response.structured_content or response.content)}


async def mcp_rule(state: DimensionState):
    url = os.environ.get("MCP_URL", "http://127.0.0.1:8000/mcp")
    async with Client(url) as client:
        response = await client.call_tool("lookup_rule", {"rule_id": state["rule_id"]})
    return {"rule_result": str(response.structured_content or response.content)}


def report(state: DimensionState):
    return {"report": (
        "Dimension comparison from HTTP MCP: " + state["dimension_result"]
        + "\nRule from HTTP MCP: " + state["rule_result"]
    )}


async def main():
    builder = StateGraph(DimensionState)
    builder.add_node("dimension", mcp_dimension)
    builder.add_node("rule", mcp_rule)
    builder.add_node("report", report)
    builder.add_edge(START, "dimension")
    builder.add_edge(START, "rule")
    builder.add_edge(["dimension", "rule"], "report")
    builder.add_edge("report", END)
    graph = builder.compile()
    result = await graph.ainvoke({
        "measured_mm": 25.0,
        "required_mm": 30.0,
        "rule_id": "DIM-001",
        "dimension_result": "",
        "rule_result": "",
        "report": "",
    })
    print("[LangGraph -> HTTP MCP] Final report:")
    print(result["report"])
    assert "-5.0" in result["dimension_result"]
    assert "DIM-001" in result["rule_result"]
    print("[PASS] LangGraph parallel nodes reached the MCP Server through HTTP")


if __name__ == "__main__":
    asyncio.run(main())
