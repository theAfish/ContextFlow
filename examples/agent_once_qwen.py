import asyncio
import os

from contextflow import Agent


MODEL = os.getenv("QWEN_MODEL", "openai/qwen-flash")
BASE_URL = os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
API_KEY = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "dummy"
ENABLE_THINKING = True
USER_INPUT = "What is LLM?"


def resolve_api_key(explicit_api_key: str | None) -> str:
    if explicit_api_key:
        return explicit_api_key
    return os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "dummy"


async def main() -> None:
    agent = Agent(
        model=MODEL,
        name="root_agent",
        description="General-purpose assistant.",
        instruction="You are a helpful assistant. Keep answers concise and factual.",
        base_url=BASE_URL,
        api_key=resolve_api_key(API_KEY),
        enable_thinking=ENABLE_THINKING,
        tools=[],
    )

    result = await agent.run_once(
        user_input=USER_INPUT,
    )
    print("\n=== Raw messages[] sent to LLM ===")
    print(result.messages)
    print("\n=== Model output ===")
    print(result.output_text)


if __name__ == "__main__":
    asyncio.run(main())
