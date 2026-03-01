"""Tests for contextflow.state — SessionManager and StateEngine."""

from __future__ import annotations

import asyncio
import pytest

from contextflow.core.models import ContextNode, ContextStack, MessageRole
from contextflow.state.session import SessionManager, SessionState
from contextflow.state.engine import StateEngine, TurnResult


# ═══════════════════════════════════════════════════════════════════════════
#  SessionState
# ═══════════════════════════════════════════════════════════════════════════


class TestSessionState:
    def test_default_empty_stack(self):
        s = SessionState(session_id="abc")
        assert s.session_id == "abc"
        assert isinstance(s.stack, ContextStack)
        assert s.stack.nodes == []


# ═══════════════════════════════════════════════════════════════════════════
#  SessionManager
# ═══════════════════════════════════════════════════════════════════════════


class TestSessionManager:
    def test_create(self):
        mgr = SessionManager()
        session = mgr.create()
        assert session.session_id
        assert isinstance(session.stack, ContextStack)

    def test_get_existing(self):
        mgr = SessionManager()
        session = mgr.create()
        found = mgr.get(session.session_id)
        assert found is session

    def test_get_nonexistent(self):
        mgr = SessionManager()
        assert mgr.get("nonexistent") is None

    def test_get_or_create_existing(self):
        mgr = SessionManager()
        session = mgr.create()
        found = mgr.get_or_create(session.session_id)
        assert found is session

    def test_get_or_create_new(self):
        mgr = SessionManager()
        session = mgr.get_or_create("custom-id")
        assert session.session_id == "custom-id"

    def test_add_node(self):
        mgr = SessionManager()
        session = mgr.create()
        node = ContextNode(role=MessageRole.USER, content="Hello")
        mgr.add_node(session.session_id, node)
        assert len(session.stack.nodes) == 1

    def test_add_node_creates_session_if_needed(self):
        mgr = SessionManager()
        node = ContextNode(role=MessageRole.USER, content="Hello")
        session = mgr.add_node("new-id", node)
        assert session.session_id == "new-id"
        assert len(session.stack.nodes) == 1

    def test_render_messages(self):
        mgr = SessionManager()
        session = mgr.create()
        mgr.add_node(session.session_id, ContextNode(role=MessageRole.USER, content="Hi"))
        msgs = mgr.render_messages(session.session_id)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_lock_for(self):
        mgr = SessionManager()
        lock = mgr.lock_for("test-session")
        assert isinstance(lock, asyncio.Lock)
        # Same session → same lock
        assert mgr.lock_for("test-session") is lock

    def test_different_sessions_different_locks(self):
        mgr = SessionManager()
        lock1 = mgr.lock_for("session-1")
        lock2 = mgr.lock_for("session-2")
        assert lock1 is not lock2


# ═══════════════════════════════════════════════════════════════════════════
#  StateEngine
# ═══════════════════════════════════════════════════════════════════════════


class TestStateEngine:
    @pytest.mark.asyncio
    async def test_run_turn(self):
        mgr = SessionManager()
        session = mgr.create()
        engine = StateEngine(session_manager=mgr)

        async def handler(messages):
            return "Reply from LLM"

        result = await engine.run_turn(session.session_id, "Hello", handler)
        assert isinstance(result, TurnResult)
        assert result.output_text == "Reply from LLM"
        assert result.session_id == session.session_id
        # Should have both user + assistant messages
        assert len(result.messages) == 2

    @pytest.mark.asyncio
    async def test_run_turn_preserves_history(self):
        mgr = SessionManager()
        session = mgr.create()
        engine = StateEngine(session_manager=mgr)

        async def handler(messages):
            return f"Saw {len(messages)} messages"

        await engine.run_turn(session.session_id, "first", handler)
        result = await engine.run_turn(session.session_id, "second", handler)
        # After 2 turns: user1, assistant1, user2, assistant2 = 4 messages
        assert len(result.messages) == 4

    @pytest.mark.asyncio
    async def test_run_turn_handler_receives_messages(self):
        mgr = SessionManager()
        session = mgr.create()
        engine = StateEngine(session_manager=mgr)
        received_messages = []

        async def handler(messages):
            received_messages.extend(messages)
            return "ok"

        await engine.run_turn(session.session_id, "Hello", handler)
        # Handler should receive 1 message (the user message)
        assert len(received_messages) == 1
        assert received_messages[0]["role"] == "user"
