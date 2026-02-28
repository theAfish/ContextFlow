import asyncio
import os

from contextflow import Agent, ContextNode, MessageRole, resolve_api_key, resolve_base_url


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
