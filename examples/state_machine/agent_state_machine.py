"""Example: Agent with a state machine.

Demonstrates how to attach an ``AgentStateMachine`` to an ``Agent`` so that:

* The agent can only call the LLM when it is in the ``"running"`` state.
* Transition hooks log state changes.
* A guard prevents moving to ``"done"`` unless the turn produced output.
* User code drives transitions explicitly.
"""

import asyncio
import os

from contextflow import Agent, AgentStateMachine, resolve_api_key, resolve_base_url


# ── 1. Configure the LLM ──────────────────────────────────────────────────
MODEL      = os.getenv("QWEN_MODEL", "openai/qwen-flash")
BASE_URL   = resolve_base_url()
API_KEY    = resolve_api_key()

USER_INPUT = "Explain what a finite state machine is in 2 sentences."


# ── 2. Build the state machine ────────────────────────────────────────────
sm = AgentStateMachine(
    initial="idle",
    transitions={
        "idle":    ["running"],
        "running": ["paused", "done", "error"],
        "paused":  ["running", "done"],
        "error":   ["idle"],
    },
)

# Only allow the LLM call when the agent is in "running" state
sm.allow_run_when("running")


# Hook: log every state change
@sm.on_change
def log_transition(old_state, new_state, ctx):
    print(f"  [state] {old_state!r}  →  {new_state!r}")


# Hook: fire when entering "error"
@sm.on_enter("error")
def on_error(old_state, new_state, ctx):
    reason = ctx.get("reason", "unknown")
    print(f"  [error hook] Entered error state. Reason: {reason}")


# Guard: block "running" → "done" unless context has output
@sm.guard("running", "done")
def require_output(from_st, to_st, ctx):
    has_output = bool(ctx.get("output"))
    if not has_output:
        print("  [guard] Blocked running→done: no output yet!")
    return has_output


# ── 3. Build the agent with the state machine attached ────────────────────
agent = Agent(
    model=MODEL,
    name="stateful_agent",
    description="A stateful assistant to demonstrate AgentStateMachine.",
    instruction="You are a concise, helpful assistant.",
    base_url=BASE_URL,
    api_key=API_KEY,
    enable_thinking=True,
    tools=[],
    state_machine=sm,
)


# ── 4. Run ────────────────────────────────────────────────────────────────
async def main() -> None:
    print(f"Agent state: {agent.state}")            # → "idle"

    # Transition idle → running
    await agent.transition_to("running")
    print(f"Agent state: {agent.state}")            # → "running"

    # Now run_once is allowed
    result = await agent.run_once(user_input=USER_INPUT)
    print(f"\n=== Model output ===\n{result.output_text}\n")

    # Try to go to "done" without providing output context → guard blocks it
    try:
        await agent.transition_to("done")
    except Exception as exc:
        print(f"  Caught: {exc}")

    # Provide output in context → guard passes
    await agent.transition_to("done", context={"output": result.output_text})
    print(f"Agent state: {agent.state}")            # → "done"

    # Demonstrate force_transition (ignores transition table & guards)
    await agent.force_transition("idle")
    print(f"Agent state after force reset: {agent.state}")  # → "idle"

    # Show history
    print("\n=== Transition history ===")
    for entry in agent.state_machine.history:
        print(f"  {entry.from_state} → {entry.to_state}  at {entry.timestamp:%H:%M:%S}")

    # Demonstrate that run_once is blocked in non-running state
    print(f"\nAgent state: {agent.state}")           # → "idle"
    try:
        await agent.run_once(user_input="This should fail")
    except Exception as exc:
        print(f"  Caught (run blocked): {exc}")


if __name__ == "__main__":
    asyncio.run(main())
