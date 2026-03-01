import asyncio
import os

from contextflow import Agent, ChatSession, resolve_api_key, resolve_base_url


MODEL = os.getenv("QWEN_MODEL", "openai/qwen-flash")
BASE_URL = resolve_base_url()
API_KEY = resolve_api_key()
ENABLE_THINKING = False


async def main() -> None:
    agent = Agent(
        model=MODEL,
        name="chat_agent",
        description="Multi-turn chat assistant.",
        instruction="You are a concise and helpful assistant for multi-turn conversation.",
        base_url=BASE_URL,
        api_key=API_KEY,
        enable_thinking=ENABLE_THINKING,
        tools=[],
    )

    session = ChatSession(agent)

    print("Type 'exit' to stop the chat.\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() == "exit":
            break

        answer = await session.send(user_text)

        print(f"\nAssistant:\n{answer}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
