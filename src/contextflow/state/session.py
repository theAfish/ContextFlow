from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from uuid import uuid4

from contextflow.core.models import ContextNode, ContextStack


@dataclass(slots=True)
class SessionState:
    session_id: str
    stack: ContextStack = field(default_factory=ContextStack)


class SessionManager:
    """In-memory session state store. Replace with DB-backed repository later."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def create(self) -> SessionState:
        session = SessionState(session_id=str(uuid4()))
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> SessionState:
        existing = self.get(session_id)
        if existing is not None:
            return existing

        session = SessionState(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def add_node(self, session_id: str, node: ContextNode) -> SessionState:
        session = self.get_or_create(session_id)
        session.stack.add(node)
        return session

    def render_messages(self, session_id: str) -> list[dict]:
        session = self.get_or_create(session_id)
        return session.stack.render_messages()

    def lock_for(self, session_id: str) -> asyncio.Lock:
        return self._locks[session_id]
