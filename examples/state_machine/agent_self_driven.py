"""Example: Self-driven agent that decides its own state transitions.

The agent autonomously manages its workflow states.  After **each** LLM call
the model returns a structured JSON directive that tells the harness:

    1. What it wants to say / do (``"response"``).
    2. Which state to transition to next (``"next_state"``).

If the agent transitions to a *working* state (``"planning"`` or
``"executing"``) it is **not** returned to the user — instead the harness
loops it back immediately so it can keep working until it decides to go to
``"reporting"`` (present results) or ``"idle"`` (wait for more input).

State diagram
─────────────

    idle  →  planning  →  executing  →  reporting  →  idle
               ↑              │
               └──────────────┘   (agent loops back if more steps needed)

Now uses ``instruction=dict[str,str]`` for per-state prompts and
``ResponseParser.parse_directive`` to extract the JSON directive.

Run:
    python examples/state_machine/agent_self_driven.py
"""

import asyncio
import os
from contextflow import Agent, AgentStateMachine, ContextNode, MessageRole, ResponseParser, resolve_api_key, resolve_base_url

# ── Config ──────────────────────────────────────────────────────────────────
MODEL    = os.getenv("QWEN_MODEL", "openai/qwen-flash")
BASE_URL = resolve_base_url()
API_KEY  = resolve_api_key()

MAX_AUTONOMOUS_STEPS = 10  # safety cap to prevent runaway loops

RESPONSE_FORMAT_INSTRUCTION = """
IMPORTANT — every reply you give MUST be **exactly** a JSON object with two
keys and nothing else (no markdown fences, no commentary outside the JSON):

{
  "response": "<your message to the user or your internal working notes>",
  "next_state": "<one of: idle, planning, executing, reporting>"
}

Rules for choosing ``next_state``:
• After receiving a brand-new task → "planning"
• While still laying out the plan  → "planning"  (or "executing" when done)
• While working through steps      → "executing" (or "reporting" when done)
• When presenting the final answer → "reporting" (or "idle" when done)
• When you are waiting for user input → "idle"
"""

# ── State Machine ───────────────────────────────────────────────────────────
sm = AgentStateMachine(
    initial="idle",
    transitions={
        "idle":      ["planning"],
        "planning":  ["executing", "planning"],
        "executing": ["executing", "reporting", "planning"],  # can loop or finish
        "reporting": ["idle"],
    },
)

sm.allow_run_when("idle", "planning", "executing", "reporting")


@sm.on_change
def log_transition(old, new, ctx):
    print(f"  ⚙ state: {old} → {new}")


# ── Agent ───────────────────────────────────────────────────────────────────
# Dict-based instruction — the framework resolves the right one per state.
# The RESPONSE_FORMAT_INSTRUCTION is appended to each state's instruction.
agent = Agent(
    model=MODEL,
    name="self_driven_agent",
    description="An agent that decides its own state transitions and keeps "
                "working until the task is complete.",
    instruction={
        "idle": (
            "You are a helpful task assistant. The user will give you a task. "
            "Acknowledge the task and decide what to do next.\n"
            + RESPONSE_FORMAT_INSTRUCTION
        ),
        "planning": (
            "You are in PLANNING mode. Break the current task down into concrete "
            "sub-steps. List them clearly. When the plan is ready, move to "
            "'executing' to start working through it.\n"
            + RESPONSE_FORMAT_INSTRUCTION
        ),
        "executing": (
            "You are in EXECUTING mode. Work through the plan step by step. "
            "Each time you are called, complete the next step (or several small "
            "ones) and report what you did. If there are remaining steps, stay "
            "in 'executing'. If all steps are done, move to 'reporting'.\n"
            + RESPONSE_FORMAT_INSTRUCTION
        ),
        "reporting": (
            "You are in REPORTING mode. Summarise everything that was done and "
            "present the final result to the user. Then transition to 'idle' so "
            "the user can give another task (or type 'exit').\n"
            + RESPONSE_FORMAT_INSTRUCTION
        ),
    },
    base_url=BASE_URL,
    api_key=API_KEY,
    enable_thinking=True,
    tools=[],
    state_machine=sm,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
async def agent_turn(user_text: str, history: list[ContextNode]) -> tuple[str, list[ContextNode]]:
    """Run a single LLM call and return (response_text, updated_history)."""
    result = await agent.run_once(user_input=user_text, history=history)

    # Add the exchange to the running history
    history.append(ContextNode(role=MessageRole.USER, content=user_text))
    history.append(ContextNode(role=MessageRole.ASSISTANT, content=result.output_text))

    return result.output_text, history


# ── Main Loop ───────────────────────────────────────────────────────────────
async def main() -> None:
    print("=" * 60)
    print("  Self-Driven Agent — autonomous state transitions")
    print("  Type a task, then watch the agent plan + execute it.")
    print("  Type 'exit' to quit.")
    print("=" * 60)

    history: list[ContextNode] = []

    while True:
        user_input = input("\n📝 You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # ── Kick off from idle ──────────────────────────────────────────
        raw, history = await agent_turn(user_input, history)
        directive = ResponseParser.parse_directive(raw, fallback_state=agent.state)
        print(f"\n🤖 Agent ({agent.state}): {directive.get('response', raw)}")

        next_state = directive.get("next_state", agent.state)

        # ── Autonomous loop ─────────────────────────────────────────────
        steps = 0
        while next_state != agent.state or next_state in ("planning", "executing"):
            # Transition
            if next_state != agent.state:
                try:
                    await agent.transition_to(next_state)
                except Exception as exc:
                    print(f"  ⚠ Transition failed: {exc}")
                    break

            # Safety cap
            steps += 1
            if steps >= MAX_AUTONOMOUS_STEPS:
                print(f"\n  ⚠ Reached {MAX_AUTONOMOUS_STEPS}-step safety cap. "
                      "Forcing report.")
                await agent.force_transition("reporting")

            # Let the agent continue working
            internal_prompt = (
                f"[Current state: {agent.state}] Continue working on the task. "
                f"Here is what you said last:\n{directive.get('response', '')}"
            )
            raw, history = await agent_turn(internal_prompt, history)
            directive = ResponseParser.parse_directive(raw, fallback_state=agent.state)

            response_text = directive.get("response", raw)
            next_state = directive.get("next_state", agent.state)

            # Show the agent's output
            label = agent.state
            if agent.state in ("planning", "executing"):
                print(f"\n🤖 Agent ({label} — working): {response_text}")
            else:
                print(f"\n🤖 Agent ({label}): {response_text}")

            # If agent says go to idle, we're done with this task
            if next_state == "idle":
                if agent.state != "idle":
                    try:
                        await agent.transition_to(next_state)
                    except Exception:
                        await agent.force_transition("idle")
                break

        # Make sure we're back in idle for the next user turn
        if agent.state != "idle":
            await agent.force_transition("idle")

    # ── History ─────────────────────────────────────────────────────────
    if agent.state_machine.history:
        print("\n" + "=" * 60)
        print("  Transition History")
        print("=" * 60)
        for i, entry in enumerate(agent.state_machine.history, 1):
            print(f"  {i}. {entry.from_state:12s} → {entry.to_state:12s}  "
                  f"({entry.timestamp:%H:%M:%S})")


if __name__ == "__main__":
    asyncio.run(main())
