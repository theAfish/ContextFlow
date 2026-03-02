import asyncio
import json
import os
from typing import Any

from contextflow import Agent, AgentSandbox, ContextNode, MessageRole, ResponseParser, ParseError
# credentials will be pulled from environment automatically


MODEL = os.getenv("QWEN_MODEL", "openai/qwen3-max")
ENABLE_THINKING = True
MAX_TOOL_STEPS = 6
# set your QWEN_API_KEY / QWEN_BASE_URL (or OPENAI_ variants) in the env


async def main() -> None:
    sandbox = AgentSandbox.create("examples/_sandbox_smoke")

    base_agent = Agent(
        model=MODEL,
        name="sandbox_code_agent",
        description="Writes and edits code in an isolated sandbox based on user chat.",
        enable_thinking=ENABLE_THINKING,
        instruction=(
            "You are a coding assistant with sandbox tools. "
            "For each turn, return ONLY valid JSON with one of these schemas:\n"
            "1) Tool call:\n"
            '{"action":"tool","tool":"sandbox_write_text","args":{"path":"main.py","content":"print(1)"}}\n'
            "2) Final answer:\n"
            '{"action":"final","message":"Done. I created main.py."}\n\n'
            "Rules:\n"
            "- Prefer tool calls when user asks to create/update/run files.\n"
            "- If user asks a normal chat question, respond with action=final and do not call tools.\n"
            "- For interactive commands that require input, call sandbox_run with stdin.\n"
            "- Example: {\"action\":\"tool\",\"tool\":\"sandbox_run\",\"args\":{\"command\":\"python app.py\",\"stdin\":\"Alice\\n\"}}\n"
            "- Use relative paths only.\n"
            "- After receiving tool results, either continue with another tool call or finalize."
        ),
    )
    agent = sandbox.attach_agent(base_agent)

    history: list[ContextNode] = []

    print("Sandbox chat-codegen started. Type 'exit' to quit.")
    print("Example: create app.py that asks name and prints hello")

    try:
        while True:
            user_text = input("\nYou: ").strip()
            if user_text.lower() == "exit":
                break
            if not user_text:
                continue

            pending_user_input = user_text

            for step in range(MAX_TOOL_STEPS):
                try:
                    result = await agent.run_once(
                        user_input=pending_user_input,
                        history=history,
                    )
                except Exception as exc:
                    print("\nModel request failed:")
                    print(str(exc))
                    print(
                        "Check API key/base URL, or retry with streaming-enabled behavior if provider requires it. "
                        "You can also set QWEN_BASE_URL/QWEN_API_KEY (or OPENAI_BASE_URL/OPENAI_API_KEY)."
                    )
                    break
                assistant_text = result.output_text.strip()
                print("\nAssistant:")
                print(assistant_text)

                history.append(ContextNode(role=MessageRole.USER, content=pending_user_input))
                history.append(ContextNode(role=MessageRole.ASSISTANT, content=assistant_text))

                try:
                    payload = ResponseParser.parse_json(assistant_text)
                except ParseError:
                    print("\nAssistant:")
                    print(assistant_text)
                    break

                print(payload)

                action = payload.get("action")
                if action == "final":
                    print("\nAssistant:")
                    print(payload.get("message", "Done."))
                    break

                if action != "tool":
                    print("\nAssistant returned invalid action:")
                    print(payload)
                    break

                tool_name = str(payload.get("tool", ""))
                args = payload.get("args")
                if not isinstance(args, dict):
                    args = {}

                tool_result = agent.execute_tool(tool_name, args)
                tool_feedback = {
                    "tool": tool_name,
                    "args": args,
                    "tool_result": tool_result,
                }
                print(f"\n[tool] {tool_name}({args}) -> {tool_result}")

                if (
                    tool_name == "sandbox_run"
                    and not tool_result.get("ok", False)
                    and "timed out" in str(tool_result.get("error", "")).lower()
                ):
                    print("\nAssistant:")
                    print(
                        "That command timed out (likely waiting for stdin/input). "
                        "I can run it by passing stdin to sandbox_run, for example stdin='Alice\\n'."
                    )

                    pending_user_input = (
                        "Tool result JSON:\n"
                        f"{json.dumps(tool_feedback, ensure_ascii=False)}\n"
                        "The command likely needs stdin. If appropriate, retry with action=tool "
                        "using sandbox_run and args including a stdin string."
                    )
                    continue

                pending_user_input = (
                    "Tool result JSON:\n"
                    f"{json.dumps(tool_feedback, ensure_ascii=False)}\n"
                    "If the user request is complete, return action=final. "
                    "Otherwise return next action=tool JSON."
                )

                if step == MAX_TOOL_STEPS - 1:
                    print("\nAssistant: stopped after max tool steps. Please ask to continue.")
    except KeyboardInterrupt:
        print("\n\nInterrupted. Exiting sandbox chat-codegen.")


if __name__ == "__main__":
    asyncio.run(main())
