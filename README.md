# LangChain Lab

Hands-on experiments with LangChain and LangGraph.

## 01 — LangChain mini agent

The first example connects to a local Ollama Qwen model and demonstrates the full **Model → Tool → Model** loop using LangChain `create_agent()`.

### Requirements

- Python 3.10+ (a recent Python version is recommended)
- Ollama running locally with model `qwen3-vl:2b-instruct`

### Windows PowerShell

```powershell
conda activate pytorch_env
git clone https://github.com/engrenage0409/langchain-lab.git
cd langchain-lab
python -m pip install -r requirements.txt
ollama list
python .\01_langchain\mini_agent.py
```

If the repo is already cloned, run `git pull` instead of cloning again.

### What to look for

The script prints `[TOOL EXECUTED] multiply(23, 47) = 1081` if the model invokes the tool, and then prints the LangChain message trace. A successful trace typically contains:

1. Human message
2. AI tool call (`multiply`)
3. Tool message (`1081`)
4. AI final answer

**Note:** The Qwen 2B model may not always invoke the tool correctly. A correct numerical answer alone does **not** prove the tool ran; check for `[TOOL EXECUTED]` and the tool message.

### Project layout

```text
01_langchain/
  mini_agent.py
requirements.txt
.gitignore
README.md
```
