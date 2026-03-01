"""Example: Multi-state LLM assistant that behaves differently per state.

The agent walks through a stateful workflow:

    greeting  →  gathering  →  answering  →  follow_up  →  done
                                   ↓
                                 error  →  gathering  (retry)

In each state the agent uses a **different system instruction** when calling
the LLM, so the model's behaviour changes depending on where we are in the
workflow.  Hooks and guards enforce the rules automatically.

Now uses ``instruction=dict[str, str]`` so the framework automatically
selects the right instruction per state — no manual swapping needed.

Run:
    python examples/agent_stateful_assistant.py
"""

import asyncio
import os

from contextflow import Agent, AgentStateMachine, resolve_api_key, resolve_base_url
from contextflow.agents.state_machine import RunBlockedByState


# ── Config ─────────────────────────────────────────────────────────────────
MODEL    = os.getenv("QWEN_MODEL", "openai/qwen-flash")
BASE_URL = resolve_base_url()
API_KEY  = resolve_api_key()


# ── State Machine ──────────────────────────────────────────────────────────
sm = AgentStateMachine(
    initial="greeting",
    transitions={
        "greeting":  ["gathering"],
        "gathering": ["answering"],
        "answering": ["follow_up", "error"],
        "follow_up": ["done", "gathering"],   # user can ask more → loop back
        "error":     ["gathering"],            # retry path
    },
)

# The agent can only call the LLM in these states (not "done" or "error")
sm.allow_run_when("greeting", "gathering", "answering", "follow_up")


# Track conversation pieces in a shared dict so hooks/guards can see them
workflow_data: dict[str, str] = {}


@sm.on_change
def log_transition(old, new, ctx):
    print(f"\n  ✦ state: {old} → {new}")


@sm.on_enter("error")
def handle_error(old, new, ctx):
    reason = ctx.get("reason", "unknown")
    print(f"  ⚠ Error occurred ({reason}). Will retry from 'gathering'.")


@sm.guard("answering", "follow_up")
def need_answer(from_st, to_st, ctx):
    """Only allow moving to follow_up if we actually produced an answer."""
    ok = bool(workflow_data.get("answer"))
    if not ok:
        print("  [guard] No answer generated yet — cannot proceed to follow_up.")
    return ok


# ── Agent ──────────────────────────────────────────────────────────────────
# Dict-based instruction: the framework resolves the right one per state.
agent = Agent(
    model=MODEL,
    name="stateful_tutor",
    description="A multi-phase assistant that changes behaviour per state.",
    instruction={
        "greeting": (
            "You are a friendly receptionist. Greet the user warmly and ask "
            "what topic they would like to learn about. Keep it to 1-2 sentences."
        ),
        "gathering": (
            "You are an expert interviewer. The user told you a topic. "
            "Ask 1-2 short clarifying questions so you can give a better answer later. "
            "Do NOT answer the topic yet."
        ),
        "answering": (
            "You are a knowledgeable tutor. The user provided a topic and some "
            "clarifications. Give a clear, detailed explanation in 3-5 sentences."
        ),
        "follow_up": (
            "You are a helpful assistant wrapping up. Ask the user if they have any "
            "follow-up questions. If not, say goodbye. Keep it to 1-2 sentences."
        ),
    },
    base_url=BASE_URL,
    api_key=API_KEY,
    enable_thinking=True,
    tools=[],
    state_machine=sm,
)


# ── Helpers ────────────────────────────────────────────────────────────────
async def call_llm(user_text: str) -> str:
    """Call the LLM — instruction is resolved automatically from current state."""
    result = await agent.run_once(user_input=user_text)
    return result.output_text


# ── Main workflow ──────────────────────────────────────────────────────────
async def main() -> None:
    print("=" * 60)
    print("  Multi-state LLM Assistant Demo")
    print("=" * 60)

    # ── Phase 1: Greeting ──────────────────────────────────────────────
    print(f"\nAgent state: {agent.state}")           # greeting
    greeting = await call_llm("Hi there!")
    print(f"\n🤖 Agent (greeting):\n{greeting}")

    # ── Phase 2: Gathering ─────────────────────────────────────────────
    await agent.transition_to("gathering")

    topic = input("\n📝 You (enter a topic): ") or "quantum computing"
    workflow_data["topic"] = topic

    clarification_qs = await call_llm(
        f"I want to learn about: {topic}"
    )
    print(f"\n🤖 Agent (gathering):\n{clarification_qs}")

    user_details = input("\n📝 You (answer the questions): ") or "Just give me the basics."
    workflow_data["details"] = user_details

    # ── Phase 3: Answering ─────────────────────────────────────────────
    await agent.transition_to("answering")

    answer = await call_llm(
        f"Topic: {workflow_data['topic']}. "
        f"Extra details: {user_details}"
    )
    print(f"\n🤖 Agent (answering):\n{answer}")
    workflow_data["answer"] = answer

    # ── Phase 4: Follow-up ─────────────────────────────────────────────
    await agent.transition_to("follow_up")

    follow = await call_llm(
        f"I just explained: {workflow_data['topic']}. "
        "Check if the user has follow-up questions."
    )
    print(f"\n🤖 Agent (follow_up):\n{follow}")

    more = input("\n📝 You (follow-up or 'no'): ").strip().lower()

    if more and more != "no":
        # Loop back to gathering for another round
        await agent.transition_to("gathering")
        extra_qs = await call_llm(more)
        print(f"\n🤖 Agent (gathering round 2):\n{extra_qs}")

        extra_details = input("\n📝 You: ") or "Keep it simple."

        await agent.transition_to("answering")
        extra_answer = await call_llm(
            f"Topic: {workflow_data['topic']}. Follow-up: {more}. "
            f"Details: {extra_details}"
        )
        print(f"\n🤖 Agent (answering round 2):\n{extra_answer}")
        workflow_data["answer"] = extra_answer

        await agent.transition_to("follow_up")
        goodbye = await call_llm("Wrap up the conversation.")
        print(f"\n🤖 Agent (follow_up):\n{goodbye}")

    # ── Phase 5: Done ──────────────────────────────────────────────────
    await agent.transition_to("done")
    print(f"\nAgent state: {agent.state}")           # done

    # LLM calls are blocked in "done" state
    try:
        await agent.run_once(user_input="Are you still there?")
    except RunBlockedByState as exc:
        print(f"\n  🚫 Blocked: {exc}")

    # ── History ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Transition History")
    print("=" * 60)
    for i, entry in enumerate(agent.state_machine.history, 1):
        print(f"  {i}. {entry.from_state:12s} → {entry.to_state:12s}  "
              f"({entry.timestamp:%H:%M:%S})")


if __name__ == "__main__":
    asyncio.run(main())
