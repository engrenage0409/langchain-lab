# LangChain + LangGraph Lab

Practical experiments with LangChain, LangGraph, local Ollama Qwen, real executable tools, Milvus semantic retrieval, and human approval.

## Project map

- `01_langchain/mini_agent.py`: model → tool → model.
- `02_langgraph/parallel_review.py`: parallel pure-Python graph + reducer.
- `02_langgraph/parallel_llm_review.py`: parallel Qwen nodes.
- `03_full_agent/`: complete multi-agent lab: parallel → sequential → supervisor choice → interrupt → approval.
- `tests/test_full_graph.py`: offline orchestration tests using fake agents (no Ollama or Milvus).

## Full Agent architecture

```text
Task + drawing TEXT description
                |
        PARALLEL / Phase 1
     +----------+-----------+
     |                      |
  symbol specialist      geometry specialist
  Qwen + inspect_symbols Qwen + inspect_dimensions
     +----------+-----------+
                | (Reducer merges both outputs)
                v
        SEQUENTIAL / Phase 2
      RAG researcher: Qwen + search_policy() -> Milvus
                |
      Evidence validator: Qwen + validate_evidence()
                |
          LLM supervisor / Phase 3
             /           \
       symbol             geometry  (choose exactly one)
             \           /
                |
           HUMAN APPROVAL
        interrupt() + SQLite checkpoint
          /                  \
        YES                   NO
       report               rejected
```

**Honest boundary:** The graph structure fixes phase 1 as two parallel workers and phase 2 as two sequential workers, *as requested*. Qwen chooses its tools inside each specialist and autonomously chooses the phase-3 route. This is not an unconstrained autonomous workflow planner. Worker tools inspect a **text description**, not real DWG/PDF files. The evidence validation demo verifies that a claim and supporting text were supplied; it does **not** establish engineering correctness.

## Requirements

- Python 3.10+; Windows + conda `pytorch_env` supported subject to dependency wheels.
- Ollama running with `qwen3-vl:2b-instruct` **and** `nomic-embed-text` (separate semantic embedding model).
- Milvus Lite local storage or a running remote Milvus instance.
- Qwen3-VL 2B is small and can occasionally fail tool calling, route format, or instruction following. The code logs actual tool calls and rejects invalid supervisor routes; do not treat generated text as verified evidence.

## Setup on Windows PowerShell

```powershell
conda activate pytorch_env
cd D:\GH\langchain-lab
git pull
python -m pip install -r requirements.txt
ollama pull nomic-embed-text
ollama list
```

Then run the full system:

```powershell
python .\03_full_agent\main.py --thread review-001
```

The first run creates `data/agent_rules.db` (a local Milvus Lite store) with three **illustrative** engineering rules and `data/checkpoints.sqlite` for checkpoint persistence. Two review agents run concurrently. You will see `[Agent:...] tool_calls=N tool_results=N` for each worker. After the supervisor chooses the final specialist, LangGraph pauses at `interrupt()` and asks for **yes/no** approval. Rejection outputs only a rejected status, never a released report.

To resume an existing paused thread after restarting the application, use the **same** thread ID and checkpoint path:

```powershell
python .\03_full_agent\main.py --thread review-001 --resume
```

You can change the task/description:

```powershell
python .\03_full_agent\main.py --thread review-002 --task "Review dimension B" --description "Dimension B is 25 mm but design requires 30 mm. Welding symbol is present."
```

### Connecting remote Milvus

```powershell
$env:MILVUS_URI = "http://127.0.0.1:19530"
$env:MILVUS_TOKEN = "root:Milvus"
python .\03_full_agent\main.py --thread remote-001
```

Use your own token; never commit credentials. The bundled seed rules are for demonstration only, **not professional standards or authoritative engineering codes**. Milvus supports storing/retrieving semantic vectors; `nomic-embed-text` produces the vectors used here.

### Offline graph tests

```powershell
python -m pytest -q tests\test_full_graph.py
```

The tests inject fake workers so you can test orchestration without a model, verify parallel fan-in, strict serial steps, the dynamic routing seam, approval/rejection and resumption. Tests do not prove actual Ollama/Milvus integration.

## Learning objective

1. **Graph Node**: Python function that reads and updates LangGraph State.
2. **LangChain Agent**: `create_agent(model, tools=...)` runs the model/tool loop inside a node.
3. **Tool**: `@tool`-registered executable function, distinct from the node.
4. **Milvus**: semantic vector search behind the `search_policy` tool.
5. **Supervisor**: Qwen returns `symbol` or `geometry`; a conditional edge dispatches the selected node.
6. **Human-in-the-loop**: `interrupt()`, `Command(resume=bool)`, persistent SQLite `thread_id` checkpoints.

## Limits before production

Add authenticated approval (CLI input is for trusted local use), fixed audit IDs, claim-level evidence verification, execution retry budgets, production logging/observability, robust schema-constrained routing, actual image/CAD parsing, and safe handling of external tool side effects. Milvus Lite is for local experiments, not multi-user production deployments.

## Sources

- [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [Milvus Lite](https://milvus.io/docs/milvus_lite.md)
- [Milvus quickstart](https://milvus.io/docs/quickstart.md)
