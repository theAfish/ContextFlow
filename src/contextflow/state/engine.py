from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from contextflow.core.models import ContextNode, MessageRole
from contextflow.state.session import SessionManager

TurnHandler = Callable[[list[dict]], Awaitable[str]]


@dataclass(slots=True)
class TurnResult:
    session_id: str
    output_text: str
    messages: list[dict]


class StateEngine:
    """Orchestrates async turns and protects per-session state with locks."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._sessions = session_manager

    async def run_turn(self, session_id: str, user_input: str, handler: TurnHandler) -> TurnResult:
        lock = self._sessions.lock_for(session_id)
        async with lock:
            self._sessions.add_node(
                session_id,
                ContextNode(role=MessageRole.USER, content=user_input),
            )
            messages = self._sessions.render_messages(session_id)
            output = await handler(messages)
            self._sessions.add_node(
                session_id,
                ContextNode(role=MessageRole.ASSISTANT, content=output),
            )

            return TurnResult(
                session_id=session_id,
                output_text=output,
                messages=self._sessions.render_messages(session_id),
            )
