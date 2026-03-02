import asyncio
import os

from contextflow import Agent


MODEL = os.getenv("QWEN_MODEL", "openai/qwen-flash")
ENABLE_THINKING = True
USER_INPUT = "What is LLM?"  # ensure QWEN_API_KEY / QWEN_BASE_URL are set in env


async def main() -> None:
    agent = Agent(
        model=MODEL,
        name="root_agent",
        description="General-purpose assistant.",
        instruction="You are a helpful assistant. Keep answers concise and factual.",
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
