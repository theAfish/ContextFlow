from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from contextflow.core.models import ContextStack
from contextflow.persistence.base import SessionRepository, SnapshotRepository


class InMemorySessionRepository(SessionRepository):
    def __init__(self) -> None:
        self._store: dict[str, ContextStack] = {}

    async def load(self, session_id: str) -> ContextStack | None:
        stack = self._store.get(session_id)
        return deepcopy(stack) if stack is not None else None

    async def save(self, session_id: str, stack: ContextStack) -> None:
        self._store[session_id] = deepcopy(stack)


class InMemorySnapshotRepository(SnapshotRepository):
    def __init__(self) -> None:
        self._snapshots: dict[str, dict[str, ContextStack]] = {}

    async def create_snapshot(self, session_id: str, stack: ContextStack) -> str:
        snapshot_id = str(uuid4())
        session_snapshots = self._snapshots.setdefault(session_id, {})
        session_snapshots[snapshot_id] = deepcopy(stack)
        return snapshot_id

    async def list_snapshots(self, session_id: str) -> list[str]:
        return sorted(self._snapshots.get(session_id, {}).keys())

    async def restore_snapshot(self, session_id: str, snapshot_id: str) -> ContextStack | None:
        snapshot = self._snapshots.get(session_id, {}).get(snapshot_id)
        return deepcopy(snapshot) if snapshot is not None else None
