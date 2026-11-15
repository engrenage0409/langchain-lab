"""Connect to the HTTP MCP service (not stdio).

Requires another terminal to run: python 04_mcp/http_server.py
"""
import asyncio
import os
from mcp import Client


async def main():
    url = os.environ.get("MCP_URL", "http://127.0.0.1:8000/mcp")
    async with Client(url) as client:
        tools = await client.list_tools()
        print("[HTTP MCP] tools:", [t.name for t in tools.tools])
        rule = await client.call_tool("lookup_rule", {"rule_id": "DIM-001"})
        print("[HTTP MCP] lookup_rule:", rule.structured_content or rule.content)
        dimension = await client.call_tool(
            "compare_dimension",
            {"measured_mm": 25.0, "required_mm": 30.0},
        )
        print("[HTTP MCP] compare_dimension:", dimension.structured_content or dimension.content)


if __name__ == "__main__":
    asyncio.run(main())
