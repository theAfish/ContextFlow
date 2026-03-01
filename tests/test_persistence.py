"""Tests for contextflow.persistence — InMemorySessionRepository, InMemorySnapshotRepository."""

from __future__ import annotations

import pytest

from contextflow.core.models import ContextNode, ContextStack, MessageRole
from contextflow.persistence.memory import (
    InMemorySessionRepository,
    InMemorySnapshotRepository,
)
from contextflow.persistence.base import SessionRepository, SnapshotRepository


# ═══════════════════════════════════════════════════════════════════════════
#  InMemorySessionRepository
# ═══════════════════════════════════════════════════════════════════════════


class TestInMemorySessionRepository:
    @pytest.mark.asyncio
    async def test_save_and_load(self):
        repo = InMemorySessionRepository()
        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.USER, content="Hello"))

        await repo.save("s1", stack)
        loaded = await repo.load("s1")
        assert loaded is not None
        assert len(loaded.nodes) == 1
        assert loaded.nodes[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_load_nonexistent(self):
        repo = InMemorySessionRepository()
        assert await repo.load("missing") is None

    @pytest.mark.asyncio
    async def test_save_returns_deep_copy(self):
        repo = InMemorySessionRepository()
        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.USER, content="Original"))

        await repo.save("s1", stack)
        loaded = await repo.load("s1")
        # Mutating the loaded copy should not affect the stored one
        loaded.nodes[0] = ContextNode(role=MessageRole.ASSISTANT, content="Modified")
        loaded2 = await repo.load("s1")
        assert loaded2.nodes[0].content == "Original"

    @pytest.mark.asyncio
    async def test_overwrite(self):
        repo = InMemorySessionRepository()
        stack1 = ContextStack()
        stack1.add(ContextNode(role=MessageRole.USER, content="V1"))
        await repo.save("s1", stack1)

        stack2 = ContextStack()
        stack2.add(ContextNode(role=MessageRole.USER, content="V2"))
        await repo.save("s1", stack2)

        loaded = await repo.load("s1")
        assert loaded.nodes[0].content == "V2"


# ═══════════════════════════════════════════════════════════════════════════
#  InMemorySnapshotRepository
# ═══════════════════════════════════════════════════════════════════════════


class TestInMemorySnapshotRepository:
    @pytest.mark.asyncio
    async def test_create_and_restore(self):
        repo = InMemorySnapshotRepository()
        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.USER, content="Snap"))

        snapshot_id = await repo.create_snapshot("s1", stack)
        assert isinstance(snapshot_id, str)

        restored = await repo.restore_snapshot("s1", snapshot_id)
        assert restored is not None
        assert len(restored.nodes) == 1
        assert restored.nodes[0].content == "Snap"

    @pytest.mark.asyncio
    async def test_list_snapshots(self):
        repo = InMemorySnapshotRepository()
        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.USER, content="A"))

        id1 = await repo.create_snapshot("s1", stack)
        id2 = await repo.create_snapshot("s1", stack)

        snaps = await repo.list_snapshots("s1")
        assert len(snaps) == 2
        assert id1 in snaps
        assert id2 in snaps

    @pytest.mark.asyncio
    async def test_list_snapshots_different_sessions(self):
        repo = InMemorySnapshotRepository()
        stack = ContextStack()

        await repo.create_snapshot("s1", stack)
        await repo.create_snapshot("s2", stack)

        assert len(await repo.list_snapshots("s1")) == 1
        assert len(await repo.list_snapshots("s2")) == 1

    @pytest.mark.asyncio
    async def test_list_snapshots_empty(self):
        repo = InMemorySnapshotRepository()
        assert await repo.list_snapshots("nonexistent") == []

    @pytest.mark.asyncio
    async def test_restore_nonexistent(self):
        repo = InMemorySnapshotRepository()
        assert await repo.restore_snapshot("s1", "fake-id") is None

    @pytest.mark.asyncio
    async def test_snapshot_is_deep_copy(self):
        repo = InMemorySnapshotRepository()
        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.USER, content="Original"))
        snap_id = await repo.create_snapshot("s1", stack)

        # Mutate the original
        stack.nodes[0] = ContextNode(role=MessageRole.USER, content="Mutated")

        restored = await repo.restore_snapshot("s1", snap_id)
        assert restored.nodes[0].content == "Original"


# ═══════════════════════════════════════════════════════════════════════════
#  ABC enforcement
# ═══════════════════════════════════════════════════════════════════════════


class TestABCEnforcement:
    def test_session_repository_abc(self):
        with pytest.raises(TypeError):
            SessionRepository()

    def test_snapshot_repository_abc(self):
        with pytest.raises(TypeError):
            SnapshotRepository()
