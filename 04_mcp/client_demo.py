"""Call the MCP server through an actual stdio subprocess."""
import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters


async def main() -> None:
    server_file = Path(__file__).with_name("server.py").resolve()
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_file)],
    )

    async with Client(params) as client:
        tools = await client.list_tools()
        print("Tools:", [tool.name for tool in tools.tools])

        resources = await client.list_resources()
        print("Resources:", [str(resource.uri) for resource in resources.resources])

        prompts = await client.list_prompts()
        print("Prompts:", [prompt.name for prompt in prompts.prompts])

        rule = await client.call_tool("lookup_rule", {"rule_id": "DIM-001"})
        print("lookup_rule:", rule.structured_content or rule.content)

        dimension = await client.call_tool(
            "compare_dimension",
            {"measured_mm": 25, "required_mm": 30},
        )
        print("compare_dimension:", dimension.structured_content or dimension.content)

        catalog = await client.read_resource("rules://catalog")
        print("Resource content:", catalog.contents[0].text)

        prompt = await client.get_prompt(
            "review_instruction",
            arguments={"topic": "dimension"},
        )
        print("Prompt content:", prompt.messages[0].content)


if __name__ == "__main__":
    asyncio.run(main())
