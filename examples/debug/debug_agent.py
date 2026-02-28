"""Example: Launch an agent with the ContextFlow Debug Frontend.

This starts a local web server at http://127.0.0.1:8790 that provides
a chat interface and real-time debug inspector showing state, session
history, and every LLM call.

Run:
    python examples/debug_agent.py
"""

import os

from contextflow import (
    Agent,
    AgentStateMachine,
    DebugSession,
    launch_debug,
    resolve_api_key,
    resolve_base_url,
)

# ── Config ─────────────────────────────────────────────────────────────
MODEL    = os.getenv("QWEN_MODEL", "openai/qwen-flash")
BASE_URL = resolve_base_url()
API_KEY  = resolve_api_key()

# ── (Optional) State Machine ──────────────────────────────────────────
sm = AgentStateMachine(
    initial="chatting",
    transitions={
        "chatting":  ["reflecting", "done"],
        "reflecting": ["chatting", "done"],
    },
)
sm.allow_run_when("chatting", "reflecting")


@sm.on_change
def log_transition(old, new, ctx):
    print(f"  [state] {old} → {new}")


# ── Agent ──────────────────────────────────────────────────────────────
agent = Agent(
    model=MODEL,
    name="debug_assistant",
    description="A helpful assistant wired to the debug frontend.",
    instruction=(
        "You are a helpful, concise assistant. "
        "Answer the user's questions clearly in 2-4 sentences."
    ),
    base_url=BASE_URL,
    api_key=API_KEY,
    enable_thinking=True,
    tools=[],
    state_machine=sm,
)

# ── Launch ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    session = DebugSession(agent)
    launch_debug(session, port=8790, open_browser=True)
