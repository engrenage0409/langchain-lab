# NEW: Qwen-driven dynamic DAG planner (recommended entrypoint)

**This is the new fully task-dependent workflow.** Earlier `03_full_agent/main.py` and `workflow.py` remain as fixed-workflow comparison examples. Use **`run_dynamic.py`** for autonomous planning.

## The dynamic execution graph

```text
User request + drawing TEXT description
              |
        Qwen DAG Planner
              |
     strict plan validation
       (agent whitelist,
      dependencies, cycles,
       task budget <= 8)
              |
       LangGraph Scheduler <-------+
              |                   |
   all dependency-ready tasks      |
    via Send => concurrent         |
              |                   |
        LangChain Agent            |
     (Qwen tool-calling loop)      |
  symbol / geometry / research /   |
        validate                  |
              |                   |
       merge task results --------+
              |
         all tasks done?
              |
    LangGraph interrupt() -> human approval
              |
         approved / rejected
```

Planner returns `{ "tasks": [{"id":"T1","agent":"symbol","instruction":"...","depends_on":[]}, ...] }`. The scheduler runs **all** currently ready tasks in parallel. Downstream tasks execute **only after** their dependencies complete. Choosing a single agent means returning one task; chaining tasks means adding dependencies. Both choices are made by Qwen, not predefined in the Python graph. The planner may choose repeated agents. Tool execution is still done by the LangChain `create_agent()` worker. The workflow validates model-generated plans but cannot guarantee their semantic quality.

### Windows setup (recommended)

```powershell
conda activate pytorch_env
cd D:\GH\langchain-lab
git pull
python -m pip install -r requirements.txt
```

**Step 1: run offline scheduling tests (no Ollama or Milvus):**

```powershell
python -m pytest -q tests\test_dynamic_workflow.py
```

**Step 2: run a complete fake demonstration (no Ollama or Milvus):**

```powershell
python .\03_full_agent\run_dynamic.py --fake --thread fake-001
```

You will see `[Plan]`, `[Scheduler] ready: ['T1', 'T2']`, worker execution, and the `[APPROVAL]` prompt. The fake mode uses a **hardcoded example DAG** and fake tools; it proves orchestration, not LLM planning or Milvus retrieval.

**Step 3: connect real Ollama + Milvus:**

```powershell
ollama pull nomic-embed-text
ollama list
$env:MILVUS_URI = "http://127.0.0.1:19530"
# Set MILVUS_TOKEN only if your server requires it:
# $env:MILVUS_TOKEN = "<your-credential>"
python .\03_full_agent\run_dynamic.py --thread real-001
```

You must have a Milvus server running and reachable at `MILVUS_URI` (e.g. Milvus Standalone via Docker Desktop / WSL2 or remote server). See [official Milvus Docker setup](https://milvus.io/docs/install_standalone-docker-compose.md). **Native Windows Milvus Lite is not a reliable default; the requirements deliberately skip it on Windows.** On supported Linux setups, you can omit `MILVUS_URI` and use the local default `./data/agent_rules.db`.

If Qwen fails to generate valid JSON twice, the planner raises an error rather than silently switching to a hardcoded plan. With the 2B model you may need to retry using a new `--thread` or swap to a stronger Ollama model in `planner.py` and `agents.py`.

**Try three genuinely different requests** (each must use a new thread ID):

```powershell
python .\03_full_agent\run_dynamic.py --thread real-symbol --task "Only check the welding symbol annotations."
python .\03_full_agent\run_dynamic.py --thread real-both --task "Independently check symbols and dimensions, then compare findings."
python .\03_full_agent\run_dynamic.py --thread real-rag --task "First retrieve the relevant rules, then validate the reported dimension mismatch."
```

You will see the **actual** Qwen-produced plan printed and the dependency-ready task batches printed by the scheduler. Parallel nodes issue concurrent requests, but local Ollama may serialize GPU inference.

**Approval / persistent resume:**

At `[APPROVAL] Publish report? yes/no:`, type `yes` or `no`. To test resume across process restarts, stop the process while it is waiting at the approval prompt, and run:

```powershell
python .\03_full_agent\run_dynamic.py --thread real-001 --resume
```

Keep the same checkpoint file (`./data/dynamic_checkpoints.sqlite`) and the same `--thread`. Don't reuse a completed thread for a new run.

**Limitations:** This is a demonstration based on a drawing *text description*, not image/CAD analysis. The seed rules are illustrative, not official engineering standards. The evidence validator tool only checks presence of evidence, not truth. Human approval is local CLI input, not authenticated. There is no autonomous re-planning after execution failures; the planner makes one validated DAG per run.

---

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
