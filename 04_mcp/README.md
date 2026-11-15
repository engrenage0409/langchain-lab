# MCP Lab — Tools / Resources / Prompts

This is a **separate mini-experiment** inside `langchain-lab`. It uses the official Python MCP SDK **v2**, where the server class is called `MCPServer` (not the old v1 `FastMCP`).

## Run (Windows PowerShell)

```powershell
conda activate pytorch_env
cd D:\GH\langchain-lab
git pull
python -m pip install -r .\04_mcp\requirements.txt
python .\04_mcp\client_demo.py
python -m pytest -q tests\test_mcp_demo.py
```

If your Python 3.10 environment cannot install MCP SDK v2, use a newer Python environment. Check `python --version`.

The client spawns the server automatically via **stdio**. Do **not** separately start `server.py` for this test.

## Files

- `server.py`: defines two real MCP tools, one resource and one prompt.
- `client_demo.py`: launches the server in another process, discovers all capabilities, calls tools, reads a resource, and fetches a prompt.
- `tests/test_mcp_demo.py`: in-process MCP protocol test (no Ollama, Milvus, or LLM required).

## Why MCP is not just @tool

`@mcp.tool()` **publishes** a typed function through a standardized protocol. A Client (such as an Agent host) can discover its schema and call it across a process or network boundary. `@langchain.tool` is registered in one LangChain application runtime. A host adapter can expose MCP tools to LangChain, but **MCP itself does not choose tools**; an LLM Agent is needed for that behavior.

Here, `lookup_rule` uses a tiny built-in demonstration dictionary, **not Milvus**; this experiment teaches protocol interoperability first. We can later replace it with your Milvus search tool without changing the MCP client contract.

## Official documentation

- https://py.sdk.modelcontextprotocol.io/
- https://github.com/modelcontextprotocol/python-sdk
