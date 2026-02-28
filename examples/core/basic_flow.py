from contextflow.core import Composer, ContextNode, MessageRole


def main() -> None:
    composer = Composer()
    history = [ContextNode(role=MessageRole.USER, content="Summarize this design in 3 bullets.")]
    stack = composer.compose(
        system_prompt="You are a concise assistant. Current time: {{current_time}}",
        history=history,
        rag_snippets=["ContextFlow exposes transparent message arrays."],
        slots={"current_time": "2026-02-23T00:00:00Z"},
    )

    messages = composer.render(stack)
    print(messages)


if __name__ == "__main__":
    main()
