"""MCP in-process test: validates the protocol-facing tool rather than a Python direct call."""
import asyncio
import importlib.util
from pathlib import Path

from mcp import Client


def load_server():
    path = Path(__file__).resolve().parents[1] / "04_mcp" / "server.py"
    spec = importlib.util.spec_from_file_location("lab_mcp_server", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.mcp


def test_mcp_tool_and_primitives():
    async def run():
        async with Client(load_server()) as client:
            assert {"lookup_rule", "compare_dimension"}.issubset(
                {tool.name for tool in (await client.list_tools()).tools}
            )
            assert "rules://catalog" in {
                str(resource.uri)
                for resource in (await client.list_resources()).resources
            }
            assert "review_instruction" in {
                prompt.name for prompt in (await client.list_prompts()).prompts
            }
            result = await client.call_tool(
                "compare_dimension",
                {"measured_mm": 25, "required_mm": 30},
            )
            assert result.structured_content["matches"] is False
            assert result.structured_content["delta_mm"] == -5.0

    asyncio.run(run())
