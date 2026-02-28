import asyncio
import os

from contextflow import Agent, resolve_api_key, resolve_base_url


MODEL = os.getenv("QWEN_MODEL", "openai/qwen-flash")
BASE_URL = resolve_base_url()
API_KEY = resolve_api_key()
ENABLE_THINKING = True
USER_INPUT = "What is the current time in Tokyo?"


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
        api_key=API_KEY,
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
