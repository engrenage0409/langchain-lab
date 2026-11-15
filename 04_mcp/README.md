# HTTP MCP + LangGraph integration (new)

This lab now supports **Streamable HTTP** on `127.0.0.1:8000/mcp`, in addition to the existing stdio example.

## A. Start the HTTP MCP Server

In **PowerShell Terminal 1**:

```powershell
conda activate pytorch_env
cd D:\GH\langchain-lab
git pull
python -m pip install -r .\04_mcp\requirements.txt
python .\04_mcp\http_server.py
```

Leave Terminal 1 open. It binds to localhost only and registers the same two tools, one Resource, and one Prompt as the stdio server.

## B. Verify via a real HTTP client

In **PowerShell Terminal 2**:

```powershell
conda activate pytorch_env
cd D:\GH\langchain-lab
python .\04_mcp\http_client_demo.py
```

This prints the remotely discovered tool names and actual tool results. You can check the listening port with `Test-NetConnection 127.0.0.1 -Port 8000`, but an MCP client handshake is the authoritative protocol check. Do not test by opening `/mcp` in a normal browser tab; MCP expects protocol requests and headers.

## C. Call the HTTP Server from LangGraph (no LLM)

With Terminal 1 still running:

```powershell
python .\04_mcp\langgraph_http_demo.py
```

Graph layout: `START -> (dimension || rule) -> report -> END`. Both **async graph nodes** use `Client("http://127.0.0.1:8000/mcp")` and call actual remote MCP tools. The program asserts that the dimension delta is -5 and the response contains DIM-001.

## D. Full existing Qwen + LangGraph + Milvus + HTTP MCP

With Terminal 1 and Ollama running, in Terminal 2:

```powershell
cd D:\GH\langchain-lab
python -m pip install -r requirements.txt
ollama list
python .\03_full_agent\run_dynamic.py --thread mcp-real-001 --mcp-url http://127.0.0.1:8000/mcp
```

This **reuses** the existing LLM DAG Planner, Milvus RAG, dynamic scheduler, and human approval. It adds two remote tools to the relevant LangChain workers:

- Geometry: `mcp_compare_dimension(measured_mm, required_mm)` -> HTTP MCP `compare_dimension`
- Research: `mcp_lookup_rule(rule_id)` -> HTTP MCP `lookup_rule`

The other tools (`inspect_dimensions`, Milvus `search_policy`, etc.) stay available. Qwen decides whether to call the remote tool, so its usage is **not guaranteed** on every run, especially with a 2B model. Watch for `[MCP HTTP] ->` and `[MCP HTTP] <-` in the logs; `tool_used=True` alone proves only that *some* tool ran.

The graph executes only the workers chosen by its LLM Planner. If the plan has no geometry or research tasks, no MCP tools are used. To encourage a test (without hardcoding the DAG), run with:

```powershell
python .\03_full_agent\run_dynamic.py --thread mcp-real-002 --mcp-url http://127.0.0.1:8000/mcp --task "Check the stated dimension of 25 mm against the required 30 mm; inspect the DIM-001 rule."
```

**Security:** The server is bound to localhost, has no authentication, and contains demo rules only. Never bind it to `0.0.0.0` or expose it online without authentication, TLS, access controls, and transport protection.

---

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
