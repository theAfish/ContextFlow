import asyncio
import os

from contextflow import Agent, resolve_api_key, resolve_base_url


MODEL = os.getenv("QWEN_MODEL", "openai/qwen-flash")
BASE_URL = resolve_base_url()
API_KEY = resolve_api_key()
ENABLE_THINKING = True
USER_INPUT = "What is LLM?"


async def main() -> None:
    agent = Agent(
        model=MODEL,
        name="root_agent",
        description="General-purpose assistant.",
        instruction="You are a helpful assistant. Keep answers concise and factual.",
        base_url=BASE_URL,
        api_key=API_KEY,
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
