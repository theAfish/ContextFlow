from __future__ import annotations

from abc import ABC, abstractmethod

from contextflow.core.models import ContextStack


class SessionRepository(ABC):
    @abstractmethod
    async def load(self, session_id: str) -> ContextStack | None:
        raise NotImplementedError

    @abstractmethod
    async def save(self, session_id: str, stack: ContextStack) -> None:
        raise NotImplementedError


class SnapshotRepository(ABC):
    @abstractmethod
    async def create_snapshot(self, session_id: str, stack: ContextStack) -> str:
        raise NotImplementedError

    @abstractmethod
    async def list_snapshots(self, session_id: str) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    async def restore_snapshot(self, session_id: str, snapshot_id: str) -> ContextStack | None:
        raise NotImplementedError
