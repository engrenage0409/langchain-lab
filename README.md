# Agent / MCP / Memory 学习归档（2026-11）

> 本仓库为可运行的学习实验；不是可直接上线的生产系统。LLM / Agent 学习线在此阶段性收尾。

## 学习路线与源码

| 模块 | 已实现内容 | 入口 |
| --- | --- | --- |
| LangChain | Qwen Tool Calling、@tool | [01_langchain/mini_agent.py](01_langchain/mini_agent.py) |
| LangGraph | State / Reducer / 并行节点 | [02_langgraph/parallel_llm_review.py](02_langgraph/parallel_llm_review.py) |
| Dynamic Multi-Agent | Qwen DAG Planner、Send Scheduler、Milvus RAG、Human Approval、Checkpoint | [03_full_agent/run_dynamic.py](03_full_agent/run_dynamic.py) |
| MCP | stdio / Streamable HTTP、Tools/Resources/Prompts、LangGraph HTTP Client、Qwen 远程 Tool | [04_mcp/README.md](04_mcp/README.md) |
| Memory | Context、短期状态、三种长期记忆、Milvus 检索、Qwen 提炼和遗忘 | [05_memory/README.md](05_memory/README.md) |

此前学习的 Agent Harness / Pi Coding Agent 另有独立项目，不在这五个目录内。

## 实践验证与边界

- **动态 Agent 与 MCP**：Qwen 生成规划 DAG，LangGraph 调度，Research Agent 实际调用 Milvus 和远程 HTTP MCP 的 `lookup_rule`，走到人工审批。工具调用日志已验证；Qwen 的规划仍可能漏掉某些要求。
- **跨会话记忆**：在 `chat-a` 提炼和审核保存记忆，在新 `chat-b` 检索、召回；同一 thread 的消息可从 SQLite Checkpoint 恢复。用户已本地验证成功。
- **不属于已实现功能**：Python 直接管理推理 KV Cache、模型级精确 token 计数、生产级自动去重/冲突合并、无审核的自动记忆写入、多用户权限隔离，以及实际 DWG 图纸内容识别。

## 长短期记忆实际存在哪里？

下面的相对路径均以 `D:\GH\langchain-lab` 为当前工作目录：

| 数据 | 实际存储位置 | 关键代码 |
| --- | --- | --- |
| 短期对话、Graph State 与 Checkpoint | `data/real_chat_checkpoints.sqlite`；由 `thread_id` 隔离 | `real_memory_agent.py:ChatState`、`SqliteSaver`、`graph.get_state()` |
| 长期事实、事件与操作规则原文 | `data/real_agent_memories.sqlite` SQLite 表 `memories` | `memory_core.py:MemoryStore.remember()/list_active()/forget()` |
| 长期记忆向量索引 | `data/memory_vectors.db` 的 Milvus Lite collection `real_agent_memory_vectors`（如果设置了 `MILVUS_URI` 则以它为准） | `milvus_memory.py:MilvusMemoryIndex.rebuild()/search()` |

**写入长期记忆：** `/remember` → `llm_memory.py:propose_memories()`（Qwen 提取候选）→ 用户逐条确认 → `MemoryStore.remember()` 写 SQLite → `MilvusMemoryIndex.rebuild()` 用 `nomic-embed-text` 建立 Milvus 索引。

**读取长期记忆：** 用户提问 → `real_memory_agent.py:retrieve()` → Query Embedding → Milvus Top-K → `answer()` 将检索结果放到 Qwen Prompt → `model.invoke()` → 新回复写入 LangGraph State 和 Checkpoint。

**读取短期记忆：** 传入同一 `thread_id`，由 `SqliteSaver` 恢复对应 `ChatState.messages`；`show_history()` 可展示最后八条，`answer()` 使用裁剪后的消息。Context 压缩只改变本次模型输入，不会删除 Checkpoint 原始历史。

**遗忘：** `/forget ID` 调用 `MemoryStore.forget()`，再重建 Milvus 索引；不会同步删除旧会话 Checkpoint 中包含的同一文本。

## 复现实验

~~~powershell
conda activate pytorch_env
cd D:\GH\langchain-lab
git pull
python -m pip install -r requirements.txt
ollama list
python .\05_memory\real_memory_agent.py --thread chat-a
~~~

在 `chat-a` 对话，输入 `/remember` 审核并保存记忆，输入 `/exit` 后以新会话运行：

~~~powershell
python .\05_memory\real_memory_agent.py --thread chat-b
~~~

## 仍值得掌握的概念

- KV Cache 在 Ollama 推理运行时（Prefill / Decode）；不是持久化用户记忆。
- Context Window 与 `compact_messages()`：当前实现是基于字符预算的裁剪，**不是**语义摘要。
- Semantic / Episodic / Procedural：同一 SQLite 表的三种类型。
- Milvus 相似度不是事实真实性概率；Memory Consolidation 需要来源检查和人工审核。

---

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

For this project, leaving `MILVUS_URI` unset uses a local Milvus Lite database (`./data/agent_rules.db`). This has been verified in the user's Windows environment, although installation support varies across Windows setups; `requirements.txt` deliberately skips an automatic Milvus Lite install on Windows. Set `MILVUS_URI` only when using a separately running remote Milvus server.

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
