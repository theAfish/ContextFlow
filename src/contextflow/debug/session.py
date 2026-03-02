"""Debug session – instruments one or many agents for the debug frontend."""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from uuid import uuid4

from contextflow.agents.agent import Agent
from contextflow.core.models import ContextNode, MessageRole
from contextflow.providers.client import AsyncLLMClient


@dataclass(slots=True)
class LLMCallRecord:
    """One captured LLM request → response pair."""

    call_id: str
    timestamp: str
    agent_name: str
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
            "agent_name": self.agent_name,
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

    agent_name: str
    from_state: str
    to_state: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "timestamp": self.timestamp,
        }


class _InstrumentedClient:
    """Wraps a real LLM client, recording every call."""

    def __init__(
        self,
        real_client: AsyncLLMClient,
        session: "DebugSession",
        *,
        agent_name: str,
    ) -> None:
        self._real = real_client
        self._session = session
        self._agent_name = agent_name

    async def complete(self, messages: list[dict[str, Any]]) -> str:
        call_id = str(uuid4())[:8]
        ts = datetime.now(timezone.utc).isoformat()
        token_est_in = sum(len(m.get("content", "")) // 4 for m in messages)

        t0 = time.perf_counter()
        response = await self._real.complete(messages)
        duration = (time.perf_counter() - t0) * 1000

        token_est_out = max(1, len(response) // 4)
        record = LLMCallRecord(
            call_id=call_id,
            timestamp=ts,
            agent_name=self._agent_name,
            agent_state=self._session.agent_state(self._agent_name),
            request_messages=messages,
            response_text=response,
            duration_ms=duration,
            token_estimate_in=token_est_in,
            token_estimate_out=token_est_out,
        )
        self._session._llm_calls.append(record)
        self._session._append_conversation_message(
            role="assistant",
            content=response,
            kind="agent_trace",
            agent_name=self._agent_name,
            call_ids=[call_id],
        )

        for cb in self._session._on_llm_call:
            cb(record)

        return response

    async def stream(self, messages: list[dict[str, Any]]):
        """Stream-through, recording the full response after completion."""
        call_id = str(uuid4())[:8]
        ts = datetime.now(timezone.utc).isoformat()
        token_est_in = sum(len(m.get("content", "")) // 4 for m in messages)

        parts: list[str] = []
        t0 = time.perf_counter()

        async for event in self._real.stream(messages):
            if event.kind == "content" and event.text:
                parts.append(event.text)
            yield event

        duration = (time.perf_counter() - t0) * 1000
        full_response = "".join(parts)
        token_est_out = max(1, len(full_response) // 4)

        record = LLMCallRecord(
            call_id=call_id,
            timestamp=ts,
            agent_name=self._agent_name,
            agent_state=self._session.agent_state(self._agent_name),
            request_messages=messages,
            response_text=full_response,
            duration_ms=duration,
            token_estimate_in=token_est_in,
            token_estimate_out=token_est_out,
        )
        self._session._llm_calls.append(record)
        self._session._append_conversation_message(
            role="assistant",
            content=full_response,
            kind="agent_trace",
            agent_name=self._agent_name,
            call_ids=[call_id],
        )

        for cb in self._session._on_llm_call:
            cb(record)


class DebugSession:
    """Wrap one root agent (and optional peer agents) for web debugging."""

    def __init__(
        self,
        agent: Agent,
        *,
        agents: list[Agent] | None = None,
        chat_handler: Callable[[str], str | Awaitable[str]] | None = None,
    ) -> None:
        self.agent = agent
        self._chat_handler = chat_handler
        self._chat_lock = asyncio.Lock()
        self._conversation: list[dict[str, Any]] = []
        self._llm_calls: list[LLMCallRecord] = []
        self._state_changes: list[StateChangeRecord] = []
        self._last_turn_call_ids: list[str] = []

        tracked_agents: list[Agent] = [agent]
        for candidate in agents or []:
            if all(existing is not candidate for existing in tracked_agents):
                tracked_agents.append(candidate)
        self._agents: dict[str, Agent] = {tracked.name: tracked for tracked in tracked_agents}

        self._on_llm_call: list[Callable[[LLMCallRecord], Any]] = []
        self._on_state_change: list[Callable[[StateChangeRecord], Any]] = []
        self._on_conversation_update: list[Callable[[dict[str, Any]], Any]] = []

        self._instrument_agents()
        self._register_state_hooks()

    def _instrument_agents(self) -> None:
        for name, tracked_agent in self._agents.items():
            real_client = tracked_agent.resolve_llm_client()
            instrumented = _InstrumentedClient(real_client, self, agent_name=name)
            object.__setattr__(tracked_agent, "llm_client", instrumented)

    def _register_state_hooks(self) -> None:
        for name, tracked_agent in self._agents.items():
            if tracked_agent.state_machine is None:
                continue

            def _capture(old_state: str, new_state: str, ctx: dict, *, agent_name: str = name) -> None:
                record = StateChangeRecord(
                    agent_name=agent_name,
                    from_state=old_state,
                    to_state=new_state,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self._state_changes.append(record)
                for cb in self._on_state_change:
                    cb(record)

            tracked_agent.state_machine._on_change.append(_capture)

    def agent_state(self, agent_name: str) -> str | None:
        tracked_agent = self._agents.get(agent_name)
        if tracked_agent is None:
            return None
        return tracked_agent.state

    def _append_conversation_message(
        self,
        *,
        role: str,
        content: str,
        kind: str,
        agent_name: str | None = None,
        call_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "agent_name": agent_name,
            "call_ids": list(call_ids or []),
        }
        self._conversation.append(msg)
        for cb in self._on_conversation_update:
            cb(msg)
        return msg

    async def chat(self, user_input: str) -> str:
        """Send a user message through the configured debug conversation path."""
        user_msg = self._append_conversation_message(
            role="user",
            content=user_input,
            kind="chat",
            agent_name="human",
        )

        history = [
            ContextNode(
                role=MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT,
                content=m["content"],
            )
            for m in self._conversation
            if m.get("kind") == "chat"
        ]
        if history:
            history = history[:-1]

        async with self._chat_lock:
            calls_before = len(self._llm_calls)

            if self._chat_handler is not None:
                maybe_result = self._chat_handler(user_input)
                response_text = (
                    await maybe_result
                    if inspect.isawaitable(maybe_result)
                    else str(maybe_result)
                )
            else:
                def _on_tool_call(tool_name: str, tool_args: dict[str, Any]) -> None:
                    self._append_conversation_message(
                        role="tool",
                        content=f"[tool_call] {tool_name}({tool_args})",
                        kind="tool_call",
                        agent_name=self.agent.name,
                    )

                def _on_tool_result(tool_name: str, result: dict[str, Any]) -> None:
                    self._append_conversation_message(
                        role="tool",
                        content=f"[tool_result] {tool_name}: {result}",
                        kind="tool_result",
                        agent_name=self.agent.name,
                    )

                response_text = await self.agent.run_with_tools(
                    user_input=user_input,
                    history=history if history else None,
                    on_tool_call=_on_tool_call,
                    on_tool_result=_on_tool_result,
                )

            turn_call_ids = [record.call_id for record in self._llm_calls[calls_before:]]
            self._last_turn_call_ids = list(turn_call_ids)

            user_msg["call_ids"] = list(turn_call_ids)
            for cb in self._on_conversation_update:
                cb(user_msg)

            self._append_conversation_message(
                role="assistant",
                content=response_text,
                kind="chat",
                agent_name=self.agent.name,
                call_ids=turn_call_ids,
            )

        return response_text

    async def transition_to(self, new_state: str) -> str:
        return await self.agent.transition_to(new_state)

    def snapshot_status(self) -> dict[str, Any]:
        sm = self.agent.state_machine
        return {
            "agent_name": self.agent.name,
            "agent_description": self.agent.description,
            "model": f"{self.agent.backend}/{self.agent.model}",
            "tracked_agents": sorted(self._agents.keys()),
            "current_state": self.agent.state,
            "available_states": sorted(sm.all_states) if sm else [],
            "allowed_transitions": (
                sm._transitions.get(sm.current, []) if sm and sm._transitions else []
            ),
            "can_run": sm.can_run() if sm else True,
            "total_llm_calls": len(self._llm_calls),
            "total_messages": len(self._conversation),
            "is_busy": self._chat_lock.locked(),
        }

    def snapshot_conversation(self) -> list[dict[str, Any]]:
        return list(self._conversation)

    def snapshot_llm_calls(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._llm_calls]

    def snapshot_llm_call(self, call_id: str) -> dict[str, Any] | None:
        for record in self._llm_calls:
            if record.call_id == call_id:
                return record.to_dict()
        return None

    def snapshot_state_changes(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._state_changes]

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
                    "from_state": entry.from_state,
                    "to_state": entry.to_state,
                    "timestamp": entry.timestamp.isoformat(),
                }
                for entry in sm.history
            ],
        }

    def last_turn_call_ids(self) -> list[str]:
        return list(self._last_turn_call_ids)

    def reset_session(self) -> None:
        """Clear all conversation, LLM calls, and state changes to start fresh."""
        self._conversation.clear()
        self._llm_calls.clear()
        self._state_changes.clear()
        self._last_turn_call_ids.clear()
        
        # Reset state machines to initial state if they exist
        for tracked_agent in self._agents.values():
            if tracked_agent.state_machine is not None:
                tracked_agent.state_machine.history.clear()
                # Reset to initial state if defined
                if hasattr(tracked_agent.state_machine, '_initial') and tracked_agent.state_machine._initial:
                    tracked_agent.state_machine.current = tracked_agent.state_machine._initial
