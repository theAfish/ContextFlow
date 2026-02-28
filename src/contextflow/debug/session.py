"""Debug session – instruments an Agent to capture every LLM call and state change."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from contextflow.agents.agent import Agent, AgentRunResult
from contextflow.core.models import ContextNode, MessageRole
from contextflow.providers.client import AsyncLLMClient
from contextflow.streaming import StreamEvent


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LLMCallRecord:
    """One captured LLM request → response pair."""

    call_id: str
    timestamp: str
    agent_state: str | None
    request_messages: list[dict[str, Any]]
    response_text: str
    duration_ms: float
    token_estimate_in: int
    token_estimate_out: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "timestamp": self.timestamp,
            "agent_state": self.agent_state,
            "request_messages": self.request_messages,
            "response_text": self.response_text,
            "duration_ms": round(self.duration_ms, 1),
            "token_estimate_in": self.token_estimate_in,
            "token_estimate_out": self.token_estimate_out,
        }


@dataclass(slots=True)
class StateChangeRecord:
    """One captured state transition."""

    from_state: str
    to_state: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Instrumented LLM client wrapper
# ---------------------------------------------------------------------------

class _InstrumentedClient:
    """Wraps a real LLM client, recording every call."""

    def __init__(self, real_client: AsyncLLMClient, session: DebugSession) -> None:
        self._real = real_client
        self._session = session

    async def complete(self, messages: list[dict[str, Any]]) -> str:
        call_id = str(uuid4())[:8]
        state = self._session.agent.state
        ts = datetime.now(timezone.utc).isoformat()
        token_est_in = sum(len(m.get("content", "")) // 4 for m in messages)

        t0 = time.perf_counter()
        response = await self._real.complete(messages)
        duration = (time.perf_counter() - t0) * 1000

        token_est_out = max(1, len(response) // 4)
        record = LLMCallRecord(
            call_id=call_id,
            timestamp=ts,
            agent_state=state,
            request_messages=messages,
            response_text=response,
            duration_ms=duration,
            token_estimate_in=token_est_in,
            token_estimate_out=token_est_out,
        )
        self._session._llm_calls.append(record)

        # Notify listeners
        for cb in self._session._on_llm_call:
            cb(record)

        return response

    async def stream(self, messages: list[dict[str, Any]]):
        """Stream-through, recording the full response after completion."""
        call_id = str(uuid4())[:8]
        state = self._session.agent.state
        ts = datetime.now(timezone.utc).isoformat()
        token_est_in = sum(len(m.get("content", "")) // 4 for m in messages)

        parts: list[str] = []
        t0 = time.perf_counter()

        async for event in self._real.stream(messages):
            if event.kind == "content" and event.text:
                parts.append(event.text)
            # Forward events through to the caller
            yield event

        duration = (time.perf_counter() - t0) * 1000
        full_response = "".join(parts)
        token_est_out = max(1, len(full_response) // 4)

        record = LLMCallRecord(
            call_id=call_id,
            timestamp=ts,
            agent_state=state,
            request_messages=messages,
            response_text=full_response,
            duration_ms=duration,
            token_estimate_in=token_est_in,
            token_estimate_out=token_est_out,
        )
        self._session._llm_calls.append(record)
        for cb in self._session._on_llm_call:
            cb(record)


# ---------------------------------------------------------------------------
# DebugSession
# ---------------------------------------------------------------------------

class DebugSession:
    """Wraps an Agent for debugging.

    * Instruments LLM calls (captures request/response/timing).
    * Listens to state-machine transitions.
    * Maintains conversation history (user + assistant messages).
    * Provides serialisable snapshots for the debug frontend.

    Usage::

        from contextflow.debug import DebugSession, launch_debug
        session = DebugSession(agent)
        launch_debug(session, port=8790)
    """

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self._conversation: list[dict[str, Any]] = []
        self._llm_calls: list[LLMCallRecord] = []
        self._state_changes: list[StateChangeRecord] = []

        # Callbacks that the server can hook into for real-time push
        self._on_llm_call: list[Callable[[LLMCallRecord], Any]] = []
        self._on_state_change: list[Callable[[StateChangeRecord], Any]] = []
        self._on_conversation_update: list[Callable[[dict[str, Any]], Any]] = []

        # Instrument the agent's LLM client
        self._instrument_agent()

        # Hook into state machine if present
        if agent.state_machine is not None:
            agent.state_machine._on_change.append(self._capture_state_change)

    # ------------------------------------------------------------------
    # Instrumentation
    # ------------------------------------------------------------------

    def _instrument_agent(self) -> None:
        """Replace the agent's LLM client with an instrumented wrapper."""
        real_client = self.agent.resolve_llm_client()
        instrumented = _InstrumentedClient(real_client, self)
        object.__setattr__(self.agent, "llm_client", instrumented)

    def _capture_state_change(self, old_state: str, new_state: str, ctx: dict) -> None:
        record = StateChangeRecord(
            from_state=old_state,
            to_state=new_state,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._state_changes.append(record)
        for cb in self._on_state_change:
            cb(record)

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def chat(self, user_input: str) -> str:
        """Send a user message, get the agent response, record everything."""
        self._conversation.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        for cb in self._on_conversation_update:
            cb(self._conversation[-1])

        # Build history from conversation
        history = [
            ContextNode(
                role=MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT,
                content=m["content"],
            )
            for m in self._conversation[:-1]  # exclude the one we just added – run_once adds it
        ]

        result: AgentRunResult = await self.agent.run_once(
            user_input=user_input,
            history=history if history else None,
        )

        self._conversation.append({
            "role": "assistant",
            "content": result.output_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        for cb in self._on_conversation_update:
            cb(self._conversation[-1])

        return result.output_text

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    async def transition_to(self, new_state: str) -> str:
        """Proxy to agent.transition_to for the frontend."""
        return await self.agent.transition_to(new_state)

    # ------------------------------------------------------------------
    # Snapshot API
    # ------------------------------------------------------------------

    def snapshot_status(self) -> dict[str, Any]:
        sm = self.agent.state_machine
        return {
            "agent_name": self.agent.name,
            "agent_description": self.agent.description,
            "model": f"{self.agent.backend}/{self.agent.model}",
            "current_state": self.agent.state,
            "available_states": sorted(sm.all_states) if sm else [],
            "allowed_transitions": (
                sm._transitions.get(sm.current, []) if sm and sm._transitions else []
            ),
            "can_run": sm.can_run() if sm else True,
            "total_llm_calls": len(self._llm_calls),
            "total_messages": len(self._conversation),
        }

    def snapshot_conversation(self) -> list[dict[str, Any]]:
        return list(self._conversation)

    def snapshot_llm_calls(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self._llm_calls]

    def snapshot_llm_call(self, call_id: str) -> dict[str, Any] | None:
        for c in self._llm_calls:
            if c.call_id == call_id:
                return c.to_dict()
        return None

    def snapshot_state_changes(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._state_changes]

    def snapshot_state_machine(self) -> dict[str, Any] | None:
        sm = self.agent.state_machine
        if sm is None:
            return None
        return {
            "current": sm.current,
            "all_states": sorted(sm.all_states),
            "transitions": sm._transitions,
            "run_states": sorted(sm._run_states) if sm._run_states else None,
            "history": [
                {
                    "from_state": h.from_state,
                    "to_state": h.to_state,
                    "timestamp": h.timestamp.isoformat(),
                }
                for h in sm.history
            ],
        }
