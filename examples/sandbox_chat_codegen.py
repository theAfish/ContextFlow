import asyncio
import json
import os
from typing import Any

from contextflow import Agent, AgentSandbox, ContextNode, MessageRole, ResponseParser
from contextflow.core.parser import ParseError
from contextflow.providers import ProviderConfig, create_client


BACKEND = "openai"
MODEL = os.getenv("QWEN_MODEL", "qwen-flash")
BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
API_KEY = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "dummy"
ENABLE_THINKING = True
MAX_TOOL_STEPS = 6


def resolve_api_key(explicit_api_key: str | None) -> str:
    if explicit_api_key:
        return explicit_api_key
    return os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "dummy"


def execute_tool(agent: Agent, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    tool_map = {tool.__name__: tool for tool in agent.tools}
    if tool_name not in tool_map:
        return {"ok": False, "error": f"Unknown tool: {tool_name}"}

    try:
        output = tool_map[tool_name](**args)
        return {"ok": True, "result": output}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def main() -> None:
    config = ProviderConfig(
        backend=BACKEND,
        model=MODEL,
        base_url=BASE_URL,
        api_key=resolve_api_key(API_KEY),
        enable_thinking=ENABLE_THINKING,
    )
    llm_client = create_client(config)

    sandbox = AgentSandbox.create("examples/_sandbox_smoke")

    base_agent = Agent(
        model=config.model,
        name="sandbox_code_agent",
        description="Writes and edits code in an isolated sandbox based on user chat.",
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
                        llm_client=llm_client,
                        history=history,
                    )
                except Exception as exc:
                    print("\nModel request failed:")
                    print(str(exc))
                    print(
                        "Set OPENAI_BASE_URL and OPENAI_API_KEY (or DASHSCOPE_API_KEY), then retry."
                    )
                    break
                assistant_text = result.output_text.strip()
                

                history.append(ContextNode(role=MessageRole.USER, content=pending_user_input))
                history.append(ContextNode(role=MessageRole.ASSISTANT, content=assistant_text))

                try:
                    payload = ResponseParser.parse_json(assistant_text)
                except ParseError:
                    print("\nAssistant:")
                    print(assistant_text)
                    break

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

                tool_result = execute_tool(agent, tool_name, args)
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
