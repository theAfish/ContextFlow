"""Example: Multi-agent orchestration — main agent delegates to a weather agent.

Demonstrates how to build a **multi-agent** system with ContextFlow where:

    ┌─────────────┐   tool call    ┌────────────────┐
    │  Main Agent  │ ────────────► │  Weather Agent  │
    │ (orchestrator)│ ◄──────────── │  (specialist)   │
    └─────────────┘   tool result  └────────────────┘

Architecture
────────────
1.  A **Weather Agent** is a specialist that knows about weather data.
    It has tools like ``get_weather`` that return structured weather info.

2.  The **Main Agent** (orchestrator) has a tool ``ask_weather_agent``
    that internally delegates to the Weather Agent.  From the main agent's
    perspective, talking to another agent is just calling a tool.

3.  The framework's built-in ``Agent.run_with_tools`` handles the agentic
    loop (LLM → parse → execute tool → feed back → repeat) automatically.

This pattern generalises easily:
    - Add more specialist agents (search, calculator, code-runner, etc.)
    - Each is exposed as a tool on the orchestrator.
    - The orchestrator decides *when* and *which* specialist to call.

Run:
    python examples/agents/multi_agent_weather.py
"""

import asyncio
import json
import os
from datetime import datetime

from contextflow import Agent, ChatSession, resolve_api_key, resolve_base_url

# ── Config ──────────────────────────────────────────────────────────────────
MODEL    = os.getenv("QWEN_MODEL", "openai/qwen-flash")

MAX_TOOL_ROUNDS = 5  # safety cap to prevent infinite tool-call loops


# ═══════════════════════════════════════════════════════════════════════════
#  1.  Weather Agent  (specialist)
# ═══════════════════════════════════════════════════════════════════════════

def get_weather(city: str) -> dict:
    """Returns the current weather for a given city.

    In a real application this would call a weather API.
    Here we return mock data for demonstration purposes.
    """
    weather_db = {
        "beijing":    {"temp_c": 5,  "condition": "Cloudy",       "humidity": 45, "wind_kph": 12},
        "tokyo":      {"temp_c": 12, "condition": "Partly Sunny", "humidity": 60, "wind_kph": 8},
        "new york":   {"temp_c": -2, "condition": "Snow",         "humidity": 80, "wind_kph": 20},
        "london":     {"temp_c": 8,  "condition": "Rainy",        "humidity": 90, "wind_kph": 15},
        "paris":      {"temp_c": 10, "condition": "Overcast",     "humidity": 70, "wind_kph": 10},
        "sydney":     {"temp_c": 28, "condition": "Sunny",        "humidity": 55, "wind_kph": 6},
        "los angeles":{"temp_c": 22, "condition": "Sunny",        "humidity": 35, "wind_kph": 5},
    }

    lookup = city.strip().lower()
    if lookup in weather_db:
        data = weather_db[lookup]
        return {
            "status": "success",
            "city": city,
            "date": datetime.now().strftime("%Y-%m-%d"),
            **data,
        }
    return {
        "status": "not_found",
        "city": city,
        "message": f"No weather data available for '{city}'. Try: {', '.join(weather_db.keys())}",
    }


def get_forecast(city: str, days: int) -> dict:
    """Returns a simple weather forecast for the next N days.

    Mock implementation for demonstration.
    """
    conditions = ["Sunny", "Cloudy", "Rainy", "Partly Sunny", "Windy"]
    import random
    random.seed(hash(city + str(days)))  # deterministic for demo

    forecast = []
    base_temp = get_weather(city).get("temp_c", 15)
    for i in range(min(days, 5)):
        forecast.append({
            "day": i + 1,
            "temp_c": base_temp + random.randint(-3, 3),
            "condition": random.choice(conditions),
        })

    return {"status": "success", "city": city, "days": len(forecast), "forecast": forecast}


# The weather agent — a specialist that can answer weather questions
weather_agent = Agent(
    model=MODEL,
    name="weather_agent",
    description="Specialist agent for weather information.",
    instruction=(
        "You are a professional weather reporter. You have access to weather "
        "tools. When asked about weather, use the appropriate tool to fetch "
        "data and then present it in a clear, friendly format.\n\n"
        "IMPORTANT — when you need to call a tool, reply with EXACTLY this "
        "JSON and nothing else:\n"
        '{"tool_call": {"name": "<tool_name>", "args": {<arguments>}}}\n\n'
        "When you have gathered all the data you need, respond with a "
        "natural-language summary (no JSON)."
    ),
    enable_thinking=True,
    tools=[get_weather, get_forecast],
)


# ═══════════════════════════════════════════════════════════════════════════
#  2.  Main Agent  (orchestrator) — delegates to sub-agents via tools
# ═══════════════════════════════════════════════════════════════════════════

def ask_weather_agent(question: str) -> dict:
    """Delegate a weather-related question to the specialist Weather Agent.

    Send any weather question and the weather agent will look up the data
    and return an answer.
    """
    answer = asyncio.get_event_loop().run_until_complete(
        weather_agent.run_with_tools(
            question,
            max_rounds=MAX_TOOL_ROUNDS,
            on_tool_call=lambda name, args: print(f"    🔧 [weather_agent] calling: {name}({args})"),
            on_tool_result=lambda name, res: print(f"    📦 [weather_agent] result: {json.dumps(res, ensure_ascii=False)[:200]}"),
        )
    )
    return {"status": "success", "agent": "weather_agent", "answer": answer}


# The main orchestrator agent
main_agent = Agent(
    model=MODEL,
    name="main_agent",
    description="Orchestrator agent that delegates tasks to specialist sub-agents.",
    instruction=(
        "You are a helpful general-purpose assistant. You can delegate tasks "
        "to specialist sub-agents by calling the tools available to you.\n\n"
        "Available specialist agents:\n"
        "  • ask_weather_agent — delegates weather questions to a weather specialist\n\n"
        "IMPORTANT — when you need to call a tool, reply with EXACTLY this "
        "JSON and nothing else:\n"
        '{"tool_call": {"name": "<tool_name>", "args": {<arguments>}}}\n\n'
        "When you have all the information, respond in natural language to the user."
    ),
    enable_thinking=True,
    tools=[ask_weather_agent],
)


# ═══════════════════════════════════════════════════════════════════════════
#  3.  Interactive demo
# ═══════════════════════════════════════════════════════════════════════════

async def main() -> None:
    print("=" * 64)
    print("  Multi-Agent Demo: Main Agent ↔ Weather Agent")
    print("  Ask about weather in any city. Type 'exit' to quit.")
    print("  Supported cities: Beijing, Tokyo, New York, London,")
    print("                    Paris, Sydney, Los Angeles")
    print("=" * 64)

    session = ChatSession(main_agent)

    while True:
        user_input = input("\n📝 You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        print()
        answer = await session.send_with_tools(
            user_input,
            max_rounds=MAX_TOOL_ROUNDS,
            on_tool_call=lambda name, args: print(f"  🔧 [main_agent] calling: {name}({args})"),
            on_tool_result=lambda name, res: print(f"  📦 [main_agent] result: {json.dumps(res, ensure_ascii=False)[:200]}"),
        )

        print(f"\n🤖 Main Agent: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
