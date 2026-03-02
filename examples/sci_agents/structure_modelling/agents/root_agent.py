# the root agent here is responsible for orchestrating the entire structure modelling process, including:
# 1. managing the overall workflow and coordinating between different sub-agents
# 2. maintaining the global state and context of the modelling process
# 3. making high-level decisions and delegating tasks to specialized sub-agents

import asyncio
import sys
from pathlib import Path

from contextflow import Agent, ChatSession

STRUCTURE_MODELLING_ROOT = Path(__file__).resolve().parent.parent
if str(STRUCTURE_MODELLING_ROOT) not in sys.path:
    sys.path.insert(0, str(STRUCTURE_MODELLING_ROOT))

from tools.modelling_tools import generate_simple_structure


MODEL = "openai/qwen3-max"

INSTRUCTION = """You are the root agent responsible for orchestrating the structure modelling process. Your tasks include:
1. Managing the overall workflow and coordinating between different sub-agents.
2. Maintaining the global state and context of the modelling process.
3. Making high-level decisions and delegating tasks to specialized sub-agents.
Your goal is to ensure the structure modelling process runs smoothly and efficiently, while effectively utilizing the capabilities
of the sub-agents under your control.

When you need to call a tool, respond with EXACTLY this JSON format and nothing else:
{"tool_call": {"name": "<tool_name>", "args": {<arguments>}}}

If no tool is needed, respond in natural language.
After receiving a tool result, summarize the result and provide a direct answer in natural language.
"""




root_agent = Agent(
    model=MODEL,
    name="root_agent",
    description="Root agent for orchestrating the structure modelling process.",
    instruction=INSTRUCTION,
    tools=[generate_simple_structure],
)

async def main() -> None:
    session = ChatSession(root_agent)

    def _on_tool_call(tool_name: str, tool_args: dict) -> None:
        print(f"\n[tool_call] {tool_name}({tool_args})")

    def _on_tool_result(tool_name: str, result: dict) -> None:
        print(f"[tool_result] {tool_name}: {result}\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        response = await session.send_with_tools(
            user_input,
            # on_tool_call=_on_tool_call,
            # on_tool_result=_on_tool_result,
        )

        print(f"\nRoot Agent:\n{response}\n")

if __name__ == "__main__":
    asyncio.run(main())