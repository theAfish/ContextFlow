import asyncio
import os

from contextflow import Agent, ChatSession


MODEL = os.getenv("QWEN_MODEL", "openai/qwen-flash")
ENABLE_THINKING = False
# Make sure QWEN_API_KEY / QWEN_BASE_URL are defined in your environment


async def main() -> None:
    agent = Agent(
        model=MODEL,
        name="root_agent",
        description="Multi-turn chat assistant.",
        instruction="You are a concise and helpful assistant for multi-turn conversation.",
        enable_thinking=ENABLE_THINKING,
        tools=[],
    )

    root_agent = agent

    session = ChatSession(root_agent)

    print("Type 'exit' to stop the chat.\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() == "exit":
            break

        answer = await session.send_with_tools(user_text)

        print(f"\nAssistant:\n{answer}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
