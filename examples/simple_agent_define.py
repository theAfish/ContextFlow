import asyncio
import os

from contextflow import Agent


MODEL = os.getenv("QWEN_MODEL", "openai/qwen-flash")
BASE_URL = os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
API_KEY = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "dummy"
ENABLE_THINKING = True
USER_INPUT = "What is the current time in Tokyo?"


def resolve_api_key(explicit_api_key: str | None) -> str:
    if explicit_api_key:
        return explicit_api_key
    return os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "dummy"


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city."""
    return {"status": "success", "city": city, "time": "10:30 AM"}


async def main() -> None:
    root_agent = Agent(
        model=MODEL,
        name="root_agent",
        description="Tells the current time in a specified city.",
        instruction=(
            "You are a helpful assistant that tells the current time in cities. "
            "When needed, propose using the 'get_current_time' tool with JSON arguments."
        ),
        base_url=BASE_URL,
        api_key=resolve_api_key(API_KEY),
        enable_thinking=ENABLE_THINKING,
        tools=[get_current_time],
    )

    result = await root_agent.run_once(
        user_input=USER_INPUT,
    )

    print("\n=== Raw messages[] ===")
    print(result.messages)
    print("\n=== Agent output ===")
    print(result.output_text)


if __name__ == "__main__":
    asyncio.run(main())
