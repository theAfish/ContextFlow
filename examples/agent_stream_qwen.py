import asyncio
import os

from contextflow import Agent, ContextNode, MessageRole


MODEL = os.getenv("QWEN_MODEL", "openai/qwen-flash")
BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
API_KEY = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "dummy"
ENABLE_THINKING = False


def resolve_api_key(explicit_api_key: str | None) -> str:
    if explicit_api_key:
        return explicit_api_key
    return os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "dummy"


async def main() -> None:
    agent = Agent(
        model=MODEL,
        name="chat_agent",
        description="Multi-turn chat assistant.",
        instruction="You are a concise and helpful assistant for multi-turn conversation.",
        base_url=BASE_URL,
        api_key=resolve_api_key(API_KEY),
        enable_thinking=ENABLE_THINKING,
        tools=[],
    )

    history: list[ContextNode] = []

    print("Type 'exit' to stop the chat.\n")
    user_text = input("You: ")

    while True:
        if user_text.strip().lower() == "exit":
            break

        result = await agent.run_once(
            user_input=user_text,
            history=history,
        )

        print("\n=== Raw messages[] sent to LLM ===")
        print(result.messages)
        print("\nAssistant:")
        print(result.output_text)

        history.append(ContextNode(role=MessageRole.USER, content=user_text))
        history.append(ContextNode(role=MessageRole.ASSISTANT, content=result.output_text))

        user_text = input("\nYou: ")


if __name__ == "__main__":
    asyncio.run(main())
