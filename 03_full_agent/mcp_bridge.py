"""Synchronous LangChain tool bridge to an HTTP MCP server.

Graph worker nodes run synchronously. Each tool invocation opens a short-lived
MCP HTTP session in its own event loop. Avoid exposing arbitrary MCP URLs to an
untrusted LLM: the URL is configured by the application operator.
"""
from __future__ import annotations

import asyncio
import json
from mcp import Client


async def _invoke(url: str, name: str, arguments: dict) -> str:
    async with Client(url) as client:
        tool_names = {tool.name for tool in (await client.list_tools()).tools}
        if name not in tool_names:
            raise ValueError(f"MCP tool {name!r} not offered by server: {sorted(tool_names)}")
        result = await client.call_tool(name, arguments)
        if result.is_error:
            raise RuntimeError(f"MCP tool {name!r} failed: {result.content}")
        if result.structured_content is not None:
            return json.dumps(result.structured_content, ensure_ascii=False)
        return "\n".join(
            getattr(item, "text", str(item)) for item in result.content
        )


def call_http_mcp(url: str, name: str, arguments: dict) -> str:
    """Use from normal sync LangChain tool functions, not from an event loop."""
    print(f"[MCP HTTP] -> {name} {arguments}", flush=True)
    result = asyncio.run(_invoke(url, name, arguments))
    print(f"[MCP HTTP] <- {result}", flush=True)
    return result
