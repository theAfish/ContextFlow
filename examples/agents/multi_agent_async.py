"""Example: Async multi-agent orchestration with multiple specialists.

A more advanced multi-agent setup with:

    ┌─────────────────┐
    │  Orchestrator    │
    │  (main agent)    │
    └──┬──────────┬────┘
       │          │
  tool call    tool call
       │          │
       ▼          ▼
  ┌──────────┐  ┌──────────────┐
  │ Weather  │  │  Knowledge   │
  │  Agent   │  │    Agent     │
  └──────────┘  └──────────────┘

Instead of blocking on ``run_until_complete`` inside a sync tool, this
version keeps everything async:

  - The orchestrator's LLM output is parsed for tool calls via
    ``ResponseParser.parse_tool_call``.
  - Tool calls are dispatched to the correct sub-agent asynchronously
    using ``Agent.run_with_tools``.
  - Results are fed back to the orchestrator.

This approach is cleaner and avoids nested event-loop issues.

Run:
    python examples/agents/multi_agent_async.py
"""

import asyncio
import json
import os
from datetime import datetime

from contextflow import Agent, ChatSession, ContextNode, MessageRole, ResponseParser, resolve_api_key, resolve_base_url

# ── Config ──────────────────────────────────────────────────────────────────
MODEL    = os.getenv("QWEN_MODEL", "openai/qwen-flash")
BASE_URL = resolve_base_url()
API_KEY  = resolve_api_key()

MAX_TOOL_ROUNDS = 6


# ═══════════════════════════════════════════════════════════════════════════
#  Tools (plain functions used by specialist agents)
# ═══════════════════════════════════════════════════════════════════════════

def get_weather(city: str) -> dict:
    """Returns the current weather for a given city."""
    weather_db = {
        "beijing":     {"temp_c": 5,  "condition": "Cloudy",       "humidity": 45, "wind_kph": 12},
        "tokyo":       {"temp_c": 12, "condition": "Partly Sunny", "humidity": 60, "wind_kph": 8},
        "new york":    {"temp_c": -2, "condition": "Snow",         "humidity": 80, "wind_kph": 20},
        "london":      {"temp_c": 8,  "condition": "Rainy",        "humidity": 90, "wind_kph": 15},
        "paris":       {"temp_c": 10, "condition": "Overcast",     "humidity": 70, "wind_kph": 10},
        "sydney":      {"temp_c": 28, "condition": "Sunny",        "humidity": 55, "wind_kph": 6},
        "los angeles": {"temp_c": 22, "condition": "Sunny",        "humidity": 35, "wind_kph": 5},
        "shanghai":    {"temp_c": 7,  "condition": "Foggy",        "humidity": 75, "wind_kph": 9},
    }
    lookup = city.strip().lower()
    if lookup in weather_db:
        return {"status": "success", "city": city, "date": datetime.now().strftime("%Y-%m-%d"), **weather_db[lookup]}
    return {"status": "not_found", "city": city,
            "message": f"No data for '{city}'. Known cities: {', '.join(weather_db)}"}


def lookup_knowledge(topic: str) -> dict:
    """Look up a short factual snippet about a topic."""
    kb = {
        "python":     "Python is a high-level, interpreted programming language known for its readability.",
        "rust":       "Rust is a systems programming language focused on safety, speed, and concurrency.",
        "contextflow":"ContextFlow is a lightweight Python framework for building LLM-powered agents.",
        "llm":        "A Large Language Model (LLM) is a neural network trained on vast text data to understand and generate language.",
    }
    key = topic.strip().lower()
    for k, v in kb.items():
        if k in key:
            return {"status": "success", "topic": topic, "summary": v}
    return {"status": "not_found", "topic": topic, "message": "No knowledge entry found."}


# ═══════════════════════════════════════════════════════════════════════════
#  Specialist Agents
# ═══════════════════════════════════════════════════════════════════════════

weather_agent = Agent(
    model=MODEL,
    name="weather_agent",
    description="Specialist for weather queries.",
    instruction=(
        "You are a weather specialist. Use available tools to fetch weather data, "
        "then present the information clearly.\n\n"
        "To call a tool, reply with ONLY this JSON:\n"
        '{"tool_call": {"name": "<tool>", "args": {<arguments>}}}\n\n'
        "Once you have the data, respond in plain natural language."
    ),
    base_url=BASE_URL,
    api_key=API_KEY,
    enable_thinking=True,
    tools=[get_weather],
)

knowledge_agent = Agent(
    model=MODEL,
    name="knowledge_agent",
    description="Specialist for general knowledge queries.",
    instruction=(
        "You are a knowledge specialist. Use the lookup_knowledge tool to find "
        "factual information, then elaborate on it for the user.\n\n"
        "To call a tool, reply with ONLY this JSON:\n"
        '{"tool_call": {"name": "<tool>", "args": {<arguments>}}}\n\n'
        "Once you have the data, respond in plain natural language."
    ),
    base_url=BASE_URL,
    api_key=API_KEY,
    enable_thinking=True,
    tools=[lookup_knowledge],
)


# ═══════════════════════════════════════════════════════════════════════════
#  Orchestrator Agent — routes to sub-agents
# ═══════════════════════════════════════════════════════════════════════════

# Registry of sub-agents keyed by the tool name the orchestrator uses
SUB_AGENTS: dict[str, Agent] = {
    "ask_weather_agent":   weather_agent,
    "ask_knowledge_agent": knowledge_agent,
}


def ask_weather_agent(question: str) -> str:
    """Delegate a weather question to the weather specialist agent."""
    return ""  # placeholder — we intercept this in the async dispatch below


def ask_knowledge_agent(question: str) -> str:
    """Delegate a knowledge/factual question to the knowledge specialist agent."""
    return ""  # placeholder


orchestrator = Agent(
    model=MODEL,
    name="orchestrator",
    description="Main orchestrator that delegates to specialist agents.",
    instruction=(
        "You are the main orchestrator assistant. You do NOT answer questions "
        "directly when a specialist is available. Instead, delegate:\n\n"
        "  • ask_weather_agent(question)   — for weather enquiries\n"
        "  • ask_knowledge_agent(question) — for factual / knowledge queries\n\n"
        "To call a tool, reply with ONLY this JSON:\n"
        '{"tool_call": {"name": "<tool>", "args": {"question": "<question>"}}}\n\n'
        "After receiving the specialist's answer, present it to the user in a "
        "friendly way. You may combine answers from multiple specialists.\n\n"
        "If the question is casual (greetings, chitchat), answer directly."
    ),
    base_url=BASE_URL,
    api_key=API_KEY,
    enable_thinking=True,
    tools=[ask_weather_agent, ask_knowledge_agent],
)

# Preferred root-agent export for CLI/debug tooling.
root_agent = orchestrator

# ═══════════════════════════════════════════════════════════════════════════
#  Agent Sessions — Each agent maintains its own independent history
# ═══════════════════════════════════════════════════════════════════════════

# Create separate chat sessions for each agent
orchestrator_session = ChatSession(orchestrator)
weather_agent_session = ChatSession(weather_agent)
knowledge_agent_session = ChatSession(knowledge_agent)

# Registry mapping sub-agent names to their sessions
SUB_AGENT_SESSIONS: dict[str, ChatSession] = {
    "ask_weather_agent":   weather_agent_session,
    "ask_knowledge_agent": knowledge_agent_session,
}


async def dispatch_tool(tool_name: str, tool_args: dict) -> dict:
    """Route a tool call to the appropriate sub-agent and return the result.
    
    Sub-agents work independently with their own history. Only the final
    answer is returned to the orchestrator — internal details are private.
    """
    sub_agent = SUB_AGENTS.get(tool_name)
    sub_session = SUB_AGENT_SESSIONS.get(tool_name)
    
    if sub_agent is None or sub_session is None:
        return {"ok": False, "error": f"Unknown tool: {tool_name}"}

    question = tool_args.get("question", str(tool_args))
    print(f"  🚀 Dispatching to [{sub_agent.name}]: {question}")

    # Use the framework's built-in agentic loop with the agent's own session
    answer = await sub_agent.run_with_tools(
        question,
        max_rounds=MAX_TOOL_ROUNDS,
        on_tool_call=lambda name, args: print(f"    🔧 [{sub_agent.name}] tool: {name}({json.dumps(args, ensure_ascii=False)})"),
        on_tool_result=lambda name, res: print(f"    📦 [{sub_agent.name}] result: {json.dumps(res, ensure_ascii=False)[:200]}"),
    )

    # Add this interaction to the sub-agent's own session history
    sub_session.history.append(ContextNode(role=MessageRole.USER, content=question))
    sub_session.history.append(ContextNode(role=MessageRole.ASSISTANT, content=answer))

    print(f"  ✅ [{sub_agent.name}] responded")
    return {"ok": True, "agent": sub_agent.name, "answer": answer}


async def run_orchestrator(user_input: str) -> str:
    """Orchestrator loop: LLM → parse → dispatch sub-agent → feed back → repeat.
    
    The orchestrator maintains its own independent history. Sub-agents are treated
    as black boxes: the orchestrator only sees their final answers, not their
    internal reasoning or tool calls.
    """
    current_input = user_input

    for _ in range(MAX_TOOL_ROUNDS):
        # Use orchestrator's own session history
        result = await orchestrator.run_once(
            user_input=current_input,
            history=orchestrator_session.history
        )
        raw = result.output_text
        tc = ResponseParser.parse_tool_call(raw)

        if tc is None:
            # Final answer from orchestrator — update our session and return
            orchestrator_session.history.append(ContextNode(role=MessageRole.USER, content=current_input))
            orchestrator_session.history.append(ContextNode(role=MessageRole.ASSISTANT, content=raw))
            return raw

        name = tc.get("name", "")
        args = tc.get("args", {})
        print(f"  🔧 [orchestrator] wants to call: {name}")

        # Async dispatch to sub-agent (sub-agent manages its own history)
        sub_result = await dispatch_tool(name, args)

        # Update orchestrator's history with its own thinking and the tool response
        orchestrator_session.history.append(ContextNode(role=MessageRole.USER, content=current_input))
        orchestrator_session.history.append(ContextNode(role=MessageRole.ASSISTANT, content=raw))
        
        # Only pass the summary result back to orchestrator, not sub-agent internals
        current_input = (
            f"The specialist agent responded:\n"
            f"{json.dumps(sub_result, ensure_ascii=False, indent=2)}\n\n"
            "Please present this information to the user in a clear, friendly way. "
            "Respond in natural language (no JSON)."
        )

    return "[Max orchestration rounds reached]"


# Debug CLI integration: keep the same conversation flow as main().
_debug_orchestrator_session = orchestrator_session


async def debug_chat(user_input: str) -> str:
    """Entry used by `contextflow web` to mirror this script's orchestration flow."""
    answer = await run_orchestrator(user_input)
    return answer


# ═══════════════════════════════════════════════════════════════════════════
#  Interactive Loop
# ═══════════════════════════════════════════════════════════════════════════

async def main() -> None:
    print("=" * 68)
    print("  Multi-Agent Async Demo: Orchestrator → Weather / Knowledge")
    print("  Try: 'What's the weather in Tokyo?'")
    print("       'Tell me about Python and the weather in London'")
    print("       'What is ContextFlow?'")
    print("  Type 'exit' to quit.")
    print("=" * 68)

    while True:
        user_input = input("\n📝 You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        answer = await run_orchestrator(user_input)

        print(f"\n🤖 Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
