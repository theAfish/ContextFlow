"""Multi-agent orchestration via state-based agent transfer.

Instead of explicit ``transfer_to_agent`` tool calls routed by an
orchestrator, each agent carries a **built-in ``transfer_to`` tool** that
sets the *next active agent* (or ``"human"``).  The session loop enforces
that every LLM turn **must** end with a ``transfer_to`` call — if the model
forgets, it is re-prompted automatically.

Architecture
────────────

    ┌──────────────────────────────────────┐
    │          MultiAgentSession           │
    │  ┌────────┐ ┌────────┐ ┌────────┐   │
    │  │ Agent A │ │ Agent B │ │ Agent C│   │
    │  └───┬────┘ └───┬────┘ └───┬────┘   │
    │      │          │          │         │
    │      └──── state (active agent) ────┘│
    │                                      │
    │  Built-in tool on every agent:       │
    │    transfer_to(agent, summary)       │
    │                                      │
    │  Special target: "human"             │
    └──────────────────────────────────────┘

Usage::

    session = MultiAgentSession(
        agents=[main_agent, weather_agent],
        initial_agent="main_agent",
    )

    async for event in session.run("What is the weather in Tokyo?"):
        print(event)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from contextflow.agents.agent import Agent, AgentRunResult, ToolSpec
from contextflow.core.models import ContextNode, MessageRole

# ---------------------------------------------------------------------------
# Transfer record
# ---------------------------------------------------------------------------

HUMAN_TARGET = "human"

# Maximum times the loop will re-prompt a single agent to force a
# ``transfer_to`` call before giving up.
_MAX_ENFORCE_RETRIES = 3

# Maximum total agent turns per ``session.run()`` invocation.
_MAX_SESSION_TURNS = 20


@dataclass(slots=True, frozen=True)
class TransferRecord:
    """Immutable log entry for an agent-to-agent (or agent-to-human) transfer."""

    from_agent: str
    to_agent: str
    summary: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Parsed LLM turn
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _TurnResult:
    """Internal result of a single LLM call with transfer parsing."""

    output_text: str
    transfer_target: str | None = None
    transfer_summary: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SessionEvent — yielded by the public ``run`` method
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SessionEvent:
    """A single event yielded by :meth:`MultiAgentSession.run`.

    ``kind`` is one of:

    * ``"agent_message"`` — an agent produced a response.
    * ``"transfer"``      — control transferred to another agent.
    * ``"tool_call"``     — an agent called one of its own tools.
    * ``"tool_result"``   — a tool returned a result.
    * ``"human"``         — control returned to the human.
    * ``"error"``         — something went wrong.
    """

    kind: str
    agent: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def pretty_print(self, *, max_content_len: int = 200) -> None:
        """Print a human-friendly representation of this event to stdout."""
        if self.kind == "agent_message":
            retry = self.metadata.get("enforce_retry")
            suffix = f" (retry #{retry})" if retry else ""
            print(f"  [agent] [{self.agent}]{suffix}: {self.content}")

        elif self.kind == "transfer":
            summary = self.metadata.get("summary", "")
            print(f"  [transfer] {self.content}")
            if summary:
                print(f"     Summary: {summary}")

        elif self.kind == "tool_call":
            print(f"  [tool_call] [{self.agent}] {self.content}")

        elif self.kind == "tool_result":
            display = (
                self.content[:max_content_len] + "..."
                if len(self.content) > max_content_len
                else self.content
            )
            print(f"  [tool_result] [{self.agent}] {display}")

        elif self.kind == "human":
            print(f"  [human] Returned to human. Summary: {self.content}")

        elif self.kind == "error":
            print(f"  [error] [{self.agent}] {self.content}")

        else:
            print(f"  [{self.kind}] [{self.agent}] {self.content}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tool_calls(raw: str) -> list[dict[str, Any]]:
    """Extract ``{"tool_call": {"name": ..., "args": ...}}`` objects from
    the LLM output.  Supports both bare JSON and markdown-fenced JSON."""
    results: list[dict[str, Any]] = []
    text = raw.strip()

    # Try the whole output as a single JSON object first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "tool_call" in obj:
            results.append(obj["tool_call"])
            return results
    except json.JSONDecodeError:
        pass

    # Fallback: find all JSON blocks
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict) and "tool_call" in obj:
                results.append(obj["tool_call"])
        except json.JSONDecodeError:
            continue

    return results


def _build_transfer_tool_spec(valid_targets: list[str]) -> dict[str, Any]:
    """Build the JSON spec block for the ``transfer_to`` built-in tool."""
    return {
        "name": "transfer_to",
        "description": (
            "REQUIRED — you MUST call this tool at the end of every response "
            "to indicate who should act next.  Targets: "
            + ", ".join(f'"{t}"' for t in valid_targets)
            + '.  Use "human" to return control to the user.'
        ),
        "parameters": [
            {"name": "agent", "type": "str",
             "description": f"Target agent name. One of: {valid_targets}"},
            {"name": "summary", "type": "str",
             "description": "Brief summary / message to pass to the next agent or human."},
        ],
    }


# ---------------------------------------------------------------------------
# MultiAgentSession
# ---------------------------------------------------------------------------

class MultiAgentSession:
    """State-based multi-agent session.

    Every registered agent automatically receives a ``transfer_to`` built-in
    tool.  After each LLM call the session checks whether ``transfer_to``
    was invoked.  If not, the agent is re-prompted until it complies.

    **Single-agent optimisation:** when the roster contains only one agent,
    the ``transfer_to`` tool is *not* injected and the enforcement loop is
    skipped — the agent simply responds and control returns to the human
    automatically.

    Parameters
    ----------
    agents:
        The roster of :class:`Agent` instances participating in the session.
    initial_agent:
        Name of the agent that starts the conversation.
    max_turns:
        Safety cap on total agent turns per ``run()`` invocation.
    max_enforce_retries:
        How many times to re-prompt an agent that forgot ``transfer_to``.
    """

    def __init__(
        self,
        *,
        agents: list[Agent],
        initial_agent: str,
        max_turns: int = _MAX_SESSION_TURNS,
        max_enforce_retries: int = _MAX_ENFORCE_RETRIES,
    ) -> None:
        self._agents: dict[str, Agent] = {a.name: a for a in agents}
        if initial_agent not in self._agents:
            raise ValueError(
                f"initial_agent {initial_agent!r} not found in roster: "
                f"{list(self._agents.keys())}"
            )
        self._active_agent: str = initial_agent
        self._max_turns = max_turns
        self._max_enforce_retries = max_enforce_retries

        # Shared conversation history across all agents
        self._history: list[ContextNode] = []

        # Transfer log
        self._transfers: list[TransferRecord] = []

        # Valid transfer targets = all agent names + "human"
        self._valid_targets: list[str] = list(self._agents.keys()) + [HUMAN_TARGET]

        # Single-agent mode: no transfer_to needed
        self._is_single_agent: bool = len(self._agents) == 1

        # Pre-compute the transfer tool spec (same for every agent)
        self._transfer_spec = _build_transfer_tool_spec(self._valid_targets)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_single_agent(self) -> bool:
        """``True`` when the roster has exactly one agent (no transfer needed)."""
        return self._is_single_agent

    @property
    def active_agent(self) -> str:
        return self._active_agent

    @property
    def transfers(self) -> list[TransferRecord]:
        return list(self._transfers)

    @property
    def history(self) -> list[ContextNode]:
        return list(self._history)

    # ------------------------------------------------------------------
    # System prompt augmentation
    # ------------------------------------------------------------------

    def _augmented_system_prompt(self, agent: Agent) -> str:
        """Build the system prompt.

        In **single-agent mode** the prompt is the agent's own prompt with
        its own tools (no ``transfer_to``).

        In **multi-agent mode** the ``transfer_to`` tool and roster info
        are injected.
        """
        # Collect the agent's own tools
        own_specs = [
            {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
            }
            for s in agent.tool_specs()
        ]

        # ── Single-agent mode ─────────────────────────────────────────
        if self._is_single_agent:
            tool_block = (
                json.dumps(own_specs, ensure_ascii=False, indent=2)
                if own_specs
                else "No tools available."
            )
            prompt = (
                f"Your name is {agent.name}\n"
                f"Here is some instructions for you: {agent.instruction}\n\n"
                "Tools:\n"
                f"{tool_block}"
            )
            if own_specs:
                prompt += (
                    "\n\nTo call a tool, output EXACTLY a JSON object like:\n"
                    '  {"tool_call": {"name": "<tool_name>", "args": {<arguments>}}}\n'
                    "After receiving tool results, incorporate them into your answer."
                )
            return prompt

        # ── Multi-agent mode ──────────────────────────────────────────
        # Inject the built-in transfer_to tool
        all_specs = own_specs + [self._transfer_spec]
        tool_block = json.dumps(all_specs, ensure_ascii=False, indent=2)

        # Build roster description so the agent knows who else exists
        roster_lines = []
        for name, a in self._agents.items():
            marker = " (you)" if name == agent.name else ""
            roster_lines.append(f"  • {name}{marker}: {a.description}")
        roster_block = "\n".join(roster_lines)

        return (
            f"Your name is {agent.name}\n"
            f"Here is some instructions for you: {agent.instruction}\n\n"
            f"You are part of a multi-agent system.  Other agents:\n"
            f"{roster_block}\n\n"
            "Tools:\n"
            f"{tool_block}\n\n"
            "CRITICAL RULES:\n"
            "1. You MUST end every response by calling the transfer_to tool.\n"
            "2. To call a tool, output EXACTLY a JSON object like:\n"
            '   {"tool_call": {"name": "<tool_name>", "args": {<arguments>}}}\n'
            "3. You may call your own tools first, then MUST call transfer_to last.\n"
            "4. If you are done helping and the user should respond, transfer to \"human\".\n"
            "5. If another agent is better suited, transfer to that agent's name.\n"
        )

    # ------------------------------------------------------------------
    # Single LLM turn
    # ------------------------------------------------------------------

    async def _run_turn(
        self,
        agent: Agent,
        user_input: str,
    ) -> _TurnResult:
        """Execute one LLM call and parse tool calls from the output."""
        # Override the system prompt to include multi-agent context
        from contextflow.core.composer import Composer

        # Build a temporary composer with our augmented system prompt
        merged_history = list(self._history)
        merged_history.append(ContextNode(role=MessageRole.USER, content=user_input))

        stack = agent.composer.compose(
            system_prompt=self._augmented_system_prompt(agent),
            history=merged_history,
        )

        llm_client = agent.resolve_llm_client()
        from contextflow.runners import AsyncAgentRunner
        runner = AsyncAgentRunner(llm_client=llm_client)
        raw_output = await runner.run(stack)

        # Parse all tool calls from the output
        tool_calls = _parse_tool_calls(raw_output)

        result = _TurnResult(output_text=raw_output)

        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})

            if name == "transfer_to":
                target = args.get("agent", "")
                summary = args.get("summary", "")
                result.transfer_target = target
                result.transfer_summary = summary
            else:
                # Execute the agent's own tool
                result.tool_calls.append(tc)
                tool_result = agent.execute_tool(name, args)
                result.tool_results.append({"tool": name, "result": tool_result})

        return result

    # ------------------------------------------------------------------
    # Enforce transfer_to
    # ------------------------------------------------------------------

    async def _enforce_transfer(
        self,
        agent: Agent,
        previous_output: str,
    ) -> _TurnResult:
        """Re-prompt the agent because it forgot to call transfer_to."""
        enforce_prompt = (
            "You did NOT call the transfer_to tool in your last response. "
            "This is required. You MUST call transfer_to to indicate who "
            "should act next.\n\n"
            f"Your previous response was:\n{previous_output}\n\n"
            "Now, call transfer_to with the appropriate target. "
            "Valid targets: " + ", ".join(f'"{t}"' for t in self._valid_targets) +
            "\n\nRespond with ONLY the tool call JSON:\n"
            '{"tool_call": {"name": "transfer_to", "args": {"agent": "<target>", "summary": "<brief summary>"}}}'
        )
        return await self._run_turn(agent, enforce_prompt)

    # ------------------------------------------------------------------
    # Public async generator — the main loop
    # ------------------------------------------------------------------

    async def run(self, user_input: str):
        """Run the session starting from the current active agent.

        Yields :class:`SessionEvent` objects as agents respond and transfer
        control.  Stops when an agent transfers to ``"human"`` or the turn
        cap is reached.

        Parameters
        ----------
        user_input:
            The human's message that kicks off (or continues) the session.
        """
        # Record the human message
        self._history.append(
            ContextNode(role=MessageRole.USER, content=user_input)
        )

        current_input = user_input
        turns = 0

        while turns < self._max_turns:
            turns += 1
            agent = self._agents[self._active_agent]

            # --- Run one LLM turn ---
            turn_result = await self._run_turn(agent, current_input)

            # Yield tool calls / results if any
            for tc in turn_result.tool_calls:
                yield SessionEvent(
                    kind="tool_call",
                    agent=agent.name,
                    content=json.dumps(tc, ensure_ascii=False),
                )
            for tr in turn_result.tool_results:
                yield SessionEvent(
                    kind="tool_result",
                    agent=agent.name,
                    content=json.dumps(tr, ensure_ascii=False),
                )

            # If the agent used its own tools, feed results back and let it
            # continue (it still needs to produce a final response)
            if turn_result.tool_results and turn_result.transfer_target is None:
                # Build a follow-up with tool results
                tool_feedback = "\n".join(
                    f"Tool '{tr['tool']}' returned: {json.dumps(tr['result'], ensure_ascii=False)}"
                    for tr in turn_result.tool_results
                )
                if self._is_single_agent:
                    current_input = (
                        f"Here are the tool results:\n{tool_feedback}\n\n"
                        "Now use these results to answer the user."
                    )
                else:
                    current_input = (
                        f"Here are the tool results:\n{tool_feedback}\n\n"
                        "Now use these results to respond, and then MUST call transfer_to."
                    )
                # Record the agent's tool-calling turn in history
                self._history.append(
                    ContextNode(role=MessageRole.ASSISTANT, content=turn_result.output_text,
                                metadata={"agent": agent.name})
                )
                continue

            # ── Single-agent fast path ────────────────────────────────
            # No transfer_to needed — just yield the response and return
            # to the human automatically.
            if self._is_single_agent:
                display_text = _strip_tool_call_json(turn_result.output_text)
                if display_text.strip():
                    yield SessionEvent(
                        kind="agent_message",
                        agent=agent.name,
                        content=display_text.strip(),
                    )
                self._history.append(
                    ContextNode(role=MessageRole.ASSISTANT, content=turn_result.output_text,
                                metadata={"agent": agent.name})
                )
                yield SessionEvent(
                    kind="human",
                    agent=agent.name,
                    content=turn_result.output_text,
                )
                return

            # ── Multi-agent: enforce transfer_to if missing ───────────
            retries = 0
            while turn_result.transfer_target is None and retries < self._max_enforce_retries:
                retries += 1
                yield SessionEvent(
                    kind="agent_message",
                    agent=agent.name,
                    content=turn_result.output_text,
                    metadata={"enforce_retry": retries},
                )
                # Record intermediate output
                self._history.append(
                    ContextNode(role=MessageRole.ASSISTANT, content=turn_result.output_text,
                                metadata={"agent": agent.name})
                )
                turn_result = await self._enforce_transfer(agent, turn_result.output_text)

            # If still no transfer after retries, force transfer to human
            if turn_result.transfer_target is None:
                turn_result.transfer_target = HUMAN_TARGET
                turn_result.transfer_summary = "(Agent failed to call transfer_to; defaulting to human)"
                yield SessionEvent(
                    kind="error",
                    agent=agent.name,
                    content="Agent did not call transfer_to after max retries. Returning to human.",
                )

            # Validate target
            target = turn_result.transfer_target
            if target not in self._valid_targets:
                yield SessionEvent(
                    kind="error",
                    agent=agent.name,
                    content=f"Invalid transfer target '{target}'. Returning to human.",
                )
                target = HUMAN_TARGET

            # --- Yield the agent's response ---
            # Strip the tool_call JSON from the display text for cleanliness
            display_text = _strip_tool_call_json(turn_result.output_text)
            if display_text.strip():
                yield SessionEvent(
                    kind="agent_message",
                    agent=agent.name,
                    content=display_text.strip(),
                )

            # Record in history
            self._history.append(
                ContextNode(role=MessageRole.ASSISTANT, content=turn_result.output_text,
                            metadata={"agent": agent.name})
            )

            # --- Record transfer ---
            self._transfers.append(
                TransferRecord(
                    from_agent=agent.name,
                    to_agent=target,
                    summary=turn_result.transfer_summary,
                )
            )

            yield SessionEvent(
                kind="transfer",
                agent=agent.name,
                content=f"{agent.name} → {target}",
                metadata={"summary": turn_result.transfer_summary},
            )

            # --- Transfer to human → stop ---
            if target == HUMAN_TARGET:
                self._active_agent = list(self._agents.keys())[0]  # reset to first
                yield SessionEvent(
                    kind="human",
                    agent=agent.name,
                    content=turn_result.transfer_summary,
                )
                return

            # --- Transfer to another agent ---
            self._active_agent = target
            # The next agent gets the summary as its input
            current_input = (
                f"[Transfer from {agent.name}]: {turn_result.transfer_summary}\n\n"
                f"Original user request: {user_input}"
            )

        # Turn cap reached
        yield SessionEvent(
            kind="error",
            agent=self._active_agent,
            content=f"Session turn cap ({self._max_turns}) reached.",
        )

    # ------------------------------------------------------------------
    # Convenience: non-generator single-shot
    # ------------------------------------------------------------------

    async def run_to_completion(self, user_input: str) -> list[SessionEvent]:
        """Collect all events from ``run()`` into a list."""
        events: list[SessionEvent] = []
        async for event in self.run(user_input):
            events.append(event)
        return events


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _strip_tool_call_json(text: str) -> str:
    """Remove ``{"tool_call": ...}`` JSON blocks from display text."""
    # Remove markdown-fenced JSON blocks containing tool_call
    cleaned = re.sub(
        r"```(?:json)?\s*\{[^`]*\"tool_call\"[^`]*\}\s*```",
        "",
        text,
        flags=re.DOTALL,
    )
    # Remove bare JSON tool_call blocks
    cleaned = re.sub(
        r'\{\s*"tool_call"\s*:\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\}\s*\}',
        "",
        cleaned,
    )
    return cleaned.strip()
