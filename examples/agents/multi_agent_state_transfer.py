"""Example: State-based multi-agent — Main Agent ↔ Weather Agent via transfer_to.

Demonstrates the **state-based agent transfer** pattern where agents hand off
control to each other (or back to the human) using a built-in ``transfer_to``
tool — no explicit orchestrator routing needed.

Flow
────
    Human  ──(question)──►  main_agent
                               │
                     transfer_to("weather_agent")
                               │
                               ▼
                          weather_agent
                      (calls get_weather tool)
                               │
                     transfer_to("main_agent")
                               │
                               ▼
                          main_agent
                     (presents final answer)
                               │
                     transfer_to("human")
                               │
                               ▼
                            Human

Key ideas:
  • Every agent has a built-in ``transfer_to`` tool injected automatically.
  • After each LLM call, the system **enforces** that ``transfer_to`` was
    called.  If not, the agent is re-prompted until it complies.
  • ``"human"`` is a special target that returns control to the user.
  • Agents see the full roster and each other's descriptions so they know
    who can handle what.

Run:
    python examples/agents/multi_agent_state_transfer.py
"""

import asyncio
import os
from datetime import datetime

from contextflow import (
    Agent,
    MultiAgentSession,
    SessionEvent,
    resolve_api_key,
    resolve_base_url,
)

# ── Config ──────────────────────────────────────────────────────────────────
MODEL    = os.getenv("QWEN_MODEL", "openai/qwen-flash")
BASE_URL = resolve_base_url()
API_KEY  = resolve_api_key()


# ═══════════════════════════════════════════════════════════════════════════
#  Tools for the Weather Agent
# ═══════════════════════════════════════════════════════════════════════════

def get_weather(city: str) -> dict:
    """Returns the current weather for a given city.

    In real use this would call a weather API.  Here we use mock data.
    """
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
        return {
            "status": "success",
            "city": city,
            "date": datetime.now().strftime("%Y-%m-%d"),
            **weather_db[lookup],
        }
    return {
        "status": "not_found",
        "city": city,
        "message": f"No data for '{city}'. Known: {', '.join(weather_db)}",
    }


def get_forecast(city: str, days: int) -> dict:
    """Returns a simple weather forecast for the next N days (mock data)."""
    import random
    random.seed(hash(city + str(days)))
    conditions = ["Sunny", "Cloudy", "Rainy", "Partly Sunny", "Windy"]
    base = get_weather(city)
    base_temp = base.get("temp_c", 15)
    forecast = [
        {"day": i + 1, "temp_c": base_temp + random.randint(-3, 3),
         "condition": random.choice(conditions)}
        for i in range(min(days, 5))
    ]
    return {"status": "success", "city": city, "forecast": forecast}


# ═══════════════════════════════════════════════════════════════════════════
#  Agent definitions
# ═══════════════════════════════════════════════════════════════════════════

main_agent = Agent(
    model=MODEL,
    name="main_agent",
    description="General-purpose assistant that coordinates other agents.",
    instruction=(
        "You are the main assistant. For weather-related questions, delegate "
        "to weather_agent by calling transfer_to. For general questions, "
        "answer directly and then transfer to human.\n"
        "When you receive a result back from weather_agent, present it "
        "nicely to the user and transfer to human."
    ),
    base_url=BASE_URL,
    api_key=API_KEY,
    enable_thinking=True,
    tools=[],  # no domain tools — just the built-in transfer_to
)

weather_agent = Agent(
    model=MODEL,
    name="weather_agent",
    description="Specialist for weather queries. Has get_weather and get_forecast tools.",
    instruction=(
        "You are a weather specialist. Use your tools (get_weather, get_forecast) "
        "to fetch weather data. After gathering the data, summarise it and "
        "transfer back to main_agent with the summary.\n"
        "Do NOT transfer to human directly — always go back to main_agent."
    ),
    base_url=BASE_URL,
    api_key=API_KEY,
    enable_thinking=True,
    tools=[get_weather, get_forecast],
)


# ═══════════════════════════════════════════════════════════════════════════
#  Session & interactive loop
# ═══════════════════════════════════════════════════════════════════════════

async def main() -> None:
    session = MultiAgentSession(
        agents=[main_agent, weather_agent],
        initial_agent="main_agent",
    )

    print("=" * 64)
    print("  State-Based Multi-Agent: Main Agent ↔ Weather Agent")
    print("  Agents transfer control via the built-in transfer_to tool.")
    print()
    print("  Try: 'What is the weather in Tokyo?'")
    print("       'Compare weather in London and Sydney'")
    print("       'Give me a 3-day forecast for Paris'")
    print("  Type 'exit' to quit.")
    print("=" * 64)

    while True:
        user_input = input("\n📝 You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        print()
        async for event in session.run(user_input):
            event.pretty_print()

        print()


if __name__ == "__main__":
    asyncio.run(main())
