# Real integrated Qwen + Milvus + LangGraph chat (recommended)

This is the **first experiment to run** if you want one real end-to-end agent instead of separate toy primitives.

## Setup

Ensure local Ollama is running and has both models:

```powershell
conda activate pytorch_env
cd D:\GH\langchain-lab
git pull
python -m pip install -r requirements.txt
ollama list
ollama pull nomic-embed-text
ollama pull qwen3-vl:2b-instruct
python -m pytest -q tests\test_real_memory_agent.py
python .\05_memory\real_memory_agent.py --thread chat-a
```

Type the following **one at a time** at `You>`:

```text
我喜欢用 Python 做计算机视觉项目。
我今天完成了 MCP HTTP 和 LangGraph 的联调。
以后发布工程审查报告前，需要先人工审核。
/history
/remember
/list
/exit
```

When `/remember` is executed, Qwen proposes semantic/episodic/procedural memories with quoted source turns. **Review and enter `yes` for only the grounded candidates you wish to store**. No write occurs without approval. Saved records go into SQLite, and the vector index is rebuilt in Milvus.

Then open a **new conversation thread**:

```powershell
python .\05_memory\real_memory_agent.py --thread chat-b
```

Ask:

```text
我之前说过，我更喜欢用什么语言做视觉项目？发布报告前有什么要求？
```

The fresh `chat-b` thread has no history from `chat-a`, but the relevant memory is loaded from **the shared Milvus long-term index** and inserted into Qwen's context. The program prints `[LONG-TERM RETRIEVAL / Milvus]`, `[SHORT-TERM / LangGraph]`, and `[QWEN ANSWER]` to prove the execution path.

To demonstrate same-thread session recovery, exit and relaunch `--thread chat-a`, then type `/history` and compare that with `chat-b`.

To forget: type `/list`, copy an exact memory ID, then `/forget <ID>`. It deletes the canonical SQLite record **and rebuilds the Milvus index**.

## Which parts are real?

- **Qwen**: `ChatOllama("qwen3-vl:2b-instruct")` generates answers and proposes memories from real turns.
- **LangGraph**: `SqliteSaver` stores message history per `thread_id` across restarts.
- **Milvus**: `OllamaEmbeddings("nomic-embed-text")` embeds queries and memories; real similarity search returns relevant long-term records.
- **Context budget**: `compact_messages()` trims older messages to an approximate character-based budget (not model-native tokenization). Old omitted details are **not** truly summarized in this example.
- **Consolidation**: Qwen produces a proposed summary and candidate memories, with human confirmation before storage.
- **Forgetting**: deletes from SQLite and synchronously rebuilds this demo's Milvus index.
- **KV Cache**: lives in Ollama's inference engine and is NOT implemented by this Python Agent or stored in SQLite.

Limitations: single-user example, no user authentication, vector search returns relevance not truth probability, and the provided source-turn index is verified structurally but **not factually entailed**. A real deployment must add source validation, safety/privacy filtering, per-user scoping and exact tokenizer accounting. The demo does not automatically decide to consolidate after each message; call `/remember` deliberately.

---

# LLM + Agent Memory Lab

A compact, runnable lab covering **five kinds of memory mechanisms**. No external personal information is used; the sample turns are synthetic.

## Run (Windows PowerShell)

```powershell
conda activate pytorch_env
cd D:\GH\langchain-lab
git pull
python -m pip install -r requirements.txt

# No model and no Milvus needed:
python -m pytest -q tests\test_memory_lab.py
python .\05_memory\demo.py

# Real semantic vector retrieval (Ollama + nomic-embed-text + Milvus Lite):
ollama list
python .\05_memory\demo.py --milvus
```

If using a remote Milvus server, set `$env:MILVUS_URI = "http://127.0.0.1:19530"` first. On your environment Milvus Lite has already worked, so no server is needed if you leave `MILVUS_URI` unset. The Milvus demo uses a **new dedicated collection** `memory_lab_vectors` and does not alter the existing `agent_lab_rules` collection.

## Files

- `memory_core.py`: budgeted context compaction, three memory types, SQLite persistence, provenance, TTL, deletion, explicit memory candidate extraction.
- `milvus_memory.py`: semantic embeddings via Ollama `nomic-embed-text`, vector storage/retrieval in Milvus, index rebuilding after forget/expiry.
- `demo.py`: run all mechanisms, including SQLite-backed LangGraph Checkpoint restore across connections.
- `tests/test_memory_lab.py`: offline deterministic tests.

## Five important distinctions

| Concept | What it retains | Lifetime | Is it a semantic memory? |
|---|---|---|---|
| KV Cache | attention keys/values of previously processed tokens; avoids redundant calculations | usually model inference/session scoped | **No** |
| Context Window | tokens currently visible to the LLM, constrained by model limit | current inference turn | **No** |
| Context compaction | shortened or summarized messages to fit a token budget | until conversation context changes | Not necessarily |
| LangGraph Checkpoint | serialized graph State, execution progress, prior messages if included | depends on Checkpointer | **No, not by itself** |
| Agent long-term memory | persisted facts, experiences, procedures (with provenance) | cross-turn / cross-session | **Yes** |

### 1) Context Memory

`compact_messages()` demonstrates a context budget and keeps latest turns. It currently inserts an **honest omission marker**, not an LLM-generated summary: omitted facts are unavailable and must not be invented. `estimate_tokens()` is a character heuristic, **not a precise tokenizer**. True KV Cache lives inside inference runtimes (e.g. Ollama), not in this Python memory store.

In production use token-aware message trimming and/or explicit, source-grounded summarization before exceeding the model context budget, taking output tokens into account.

### 2) Short-term Memory

`demo.py` creates a LangGraph State and saves it using `SqliteSaver`. It then opens a new Checkpointer and reads the saved State with the **same thread_id**. This proves checkpoint persistence independently of long-term memory. A checkpoint is not a list of facts; it is execution state.

### 3) Long-term Memory

`MemoryStore.remember()` stores three *types*:
- **Semantic**: durable fact (e.g. a user's explicitly stated programming preference).
- **Episodic**: event and when/where it happened.
- **Procedural**: how to do something, a workflow or approved policy.

Each record contains a memory ID, source/provenance, creation time, optional TTL, and importance. SQLite provides durability. Provenance and consent matter; do not automatically store secrets or infer sensitive traits.

### 4) Retrieval and forgetting

`MilvusMemoryIndex.rebuild(store)` indexes active SQLite memories with `nomic-embed-text` embeddings; `search(query)` retrieves semantically related candidates. The score is **not a probability of truth**. The index is a *derived* copy of the canonical SQLite records; call `rebuild()` after `forget()` or expiry to remove stale entries. Deleting the SQLite record does not automatically erase data already copied to the vector index.

This example is single-user and re-creates a dedicated collection. A multi-user service must use scoped records, authentication, controlled deletion, and access filters before retrieval.

### 5) Consolidation

`consolidate_candidates()` extracts only explicit user-labeled `FACT:`, `EVENT:`, or `PROCEDURE:` statements. It is an educational, auditable baseline. A real automatic pipeline would perform:

```text
Conversation turns
    -> Candidate extraction (LLM, optionally)
    -> Verification against original turns
    -> Sensitivity/consent filter
    -> Deduplication + conflict resolution + provenance
    -> Save semantic / episodic / procedural memory
    -> Embed + index in Milvus
    -> Retrieve for future turns
    -> Expire / revise / explicit forget
```

Never equate `KV Cache` with remembered user facts, nor `Checkpoint` with semantic memory.

## Notes

Use `python .\05_memory\demo.py` first and inspect the logs. For a real vector search, run the `--milvus` version after verifying your local Ollama embedding model is present.
