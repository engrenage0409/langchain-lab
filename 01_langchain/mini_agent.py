"""Minimal LangChain agent using a local Ollama model.

Run:
    python 01_langchain/mini_agent.py
"""
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return their product."""
    result = a * b
    print(f"\n[TOOL EXECUTED] multiply({a}, {b}) = {result}", flush=True)
    return result


def main() -> None:
    model = ChatOllama(
        model="qwen3-vl:2b-instruct",
        temperature=0,
        num_ctx=4096,
    )

    agent = create_agent(
        model=model,
        tools=[multiply],
        system_prompt=(
            "You are a helpful assistant. "
            "For multiplication questions, call the multiply tool. "
            "Use the tool result to answer. Do not guess."
        ),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is 23 multiplied by 47? Use the multiply tool.",
                }
            ]
        },
        config={"recursion_limit": 6},
    )

    print("\n========== AGENT TRACE ==========")
    for message in result["messages"]:
        print(f"\nTYPE: {message.type}")
        if message.type == "ai" and message.tool_calls:
            print("TOOL CALLS:", message.tool_calls)
        else:
            print("CONTENT:", message.content)

    print("\n========== FINAL ANSWER ==========")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
