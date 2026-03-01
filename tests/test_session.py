"""Tests for contextflow.agents.session — MultiAgentSession helpers."""

from __future__ import annotations

import pytest

from contextflow.agents.session import (
    TransferRecord,
    SessionEvent,
    _parse_tool_calls,
    _build_transfer_tool_spec,
    _strip_tool_call_json,
    MultiAgentSession,
    HUMAN_TARGET,
)
from contextflow.agents.agent import Agent

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conftest import FakeLLMClient


# ═══════════════════════════════════════════════════════════════════════════
#  TransferRecord
# ═══════════════════════════════════════════════════════════════════════════


class TestTransferRecord:
    def test_creation(self):
        rec = TransferRecord(from_agent="A", to_agent="B", summary="handoff")
        assert rec.from_agent == "A"
        assert rec.to_agent == "B"
        assert rec.summary == "handoff"
        assert rec.timestamp is not None

    def test_frozen(self):
        rec = TransferRecord(from_agent="A", to_agent="B", summary="s")
        with pytest.raises(AttributeError):
            rec.from_agent = "C"


# ═══════════════════════════════════════════════════════════════════════════
#  SessionEvent
# ═══════════════════════════════════════════════════════════════════════════


class TestSessionEvent:
    def test_creation(self):
        evt = SessionEvent(kind="agent_message", agent="Bot", content="Hello")
        assert evt.kind == "agent_message"
        assert evt.agent == "Bot"
        assert evt.content == "Hello"
        assert evt.metadata == {}

    def test_pretty_print_agent_message(self, capsys):
        evt = SessionEvent(kind="agent_message", agent="Bot", content="Hi")
        evt.pretty_print()
        captured = capsys.readouterr()
        assert "Bot" in captured.out
        assert "Hi" in captured.out

    def test_pretty_print_transfer(self, capsys):
        evt = SessionEvent(
            kind="transfer",
            agent="A",
            content="A → B",
            metadata={"summary": "handoff"},
        )
        evt.pretty_print()
        captured = capsys.readouterr()
        assert "transfer" in captured.out.lower()

    def test_pretty_print_human(self, capsys):
        evt = SessionEvent(kind="human", agent="Bot", content="done")
        evt.pretty_print()
        captured = capsys.readouterr()
        assert "human" in captured.out.lower()

    def test_pretty_print_error(self, capsys):
        evt = SessionEvent(kind="error", agent="Bot", content="fail")
        evt.pretty_print()
        captured = capsys.readouterr()
        assert "error" in captured.out.lower()


# ═══════════════════════════════════════════════════════════════════════════
#  _parse_tool_calls
# ═══════════════════════════════════════════════════════════════════════════


class TestParseToolCalls:
    def test_single_tool_call(self):
        raw = '{"tool_call": {"name": "search", "args": {"q": "hello"}}}'
        calls = _parse_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"

    def test_no_tool_call(self):
        raw = "Just a normal response."
        calls = _parse_tool_calls(raw)
        assert calls == []

    def test_json_without_tool_call_key(self):
        raw = '{"key": "value"}'
        calls = _parse_tool_calls(raw)
        assert calls == []

    def test_embedded_tool_call_regex_limitation(self):
        # The fallback regex only supports one level of brace nesting.
        # Tool calls inherently have 2+ levels, so non-JSON-parseable
        # embedded text won't find them via regex. Only the direct
        # json.loads() path works.
        raw = 'Some text. {"tool_call": {"name": "calc", "args": {"n": 1}}}'
        calls = _parse_tool_calls(raw)
        assert calls == []


# ═══════════════════════════════════════════════════════════════════════════
#  _build_transfer_tool_spec
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildTransferToolSpec:
    def test_spec_structure(self):
        spec = _build_transfer_tool_spec(["agentA", "agentB", "human"])
        assert spec["name"] == "transfer_to"
        assert "agentA" in spec["description"]
        assert "human" in spec["description"]
        assert len(spec["parameters"]) == 2


# ═══════════════════════════════════════════════════════════════════════════
#  _strip_tool_call_json
# ═══════════════════════════════════════════════════════════════════════════


class TestStripToolCallJson:
    def test_strip_bare_json(self):
        text = 'Here is my answer. {"tool_call": {"name": "transfer_to", "args": {"agent": "human", "summary": "done"}}}'
        cleaned = _strip_tool_call_json(text)
        assert "Here is my answer." in cleaned
        assert "tool_call" not in cleaned

    def test_strip_fenced_json(self):
        text = 'Response:\n```json\n{"tool_call": {"name": "transfer_to", "args": {"agent": "human"}}}\n```'
        cleaned = _strip_tool_call_json(text)
        assert "tool_call" not in cleaned

    def test_no_tool_call_unchanged(self):
        text = "Just a plain response."
        assert _strip_tool_call_json(text) == text


# ═══════════════════════════════════════════════════════════════════════════
#  MultiAgentSession — construction
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiAgentSessionConstruction:
    def test_invalid_initial_agent_raises(self):
        agent = Agent(
            model="test", name="A", description="d", instruction="i",
            llm_client=FakeLLMClient(),
        )
        with pytest.raises(ValueError, match="not found"):
            MultiAgentSession(agents=[agent], initial_agent="nonexistent")

    def test_single_agent_mode(self):
        agent = Agent(
            model="test", name="A", description="d", instruction="i",
            llm_client=FakeLLMClient(),
        )
        session = MultiAgentSession(agents=[agent], initial_agent="A")
        assert session.is_single_agent is True
        assert session.active_agent == "A"

    def test_multi_agent_mode(self):
        a = Agent(model="t", name="A", description="d", instruction="i", llm_client=FakeLLMClient())
        b = Agent(model="t", name="B", description="d", instruction="i", llm_client=FakeLLMClient())
        session = MultiAgentSession(agents=[a, b], initial_agent="A")
        assert session.is_single_agent is False

    def test_properties(self):
        a = Agent(model="t", name="A", description="d", instruction="i", llm_client=FakeLLMClient())
        session = MultiAgentSession(agents=[a], initial_agent="A")
        assert session.transfers == []
        assert session.history == []


# ═══════════════════════════════════════════════════════════════════════════
#  MultiAgentSession — single-agent run
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiAgentSessionSingleRun:
    @pytest.mark.asyncio
    async def test_single_agent_run(self):
        fake = FakeLLMClient(response="Hi there!")
        agent = Agent(
            model="test", name="Bot", description="d", instruction="i",
            llm_client=fake,
        )
        session = MultiAgentSession(agents=[agent], initial_agent="Bot")
        events = await session.run_to_completion("Hello")

        kinds = [e.kind for e in events]
        assert "agent_message" in kinds
        assert "human" in kinds

    @pytest.mark.asyncio
    async def test_single_agent_with_tools(self):
        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        # First response: tool call, second response: final answer
        fake = FakeLLMClient(responses=[
            '{"tool_call": {"name": "add", "args": {"a": 1, "b": 2}}}',
            "The result is 3!",
        ])
        agent = Agent(
            model="test", name="Bot", description="d", instruction="i",
            llm_client=fake,
            tools=[add],
        )
        session = MultiAgentSession(agents=[agent], initial_agent="Bot")
        events = await session.run_to_completion("What is 1+2?")

        kinds = [e.kind for e in events]
        assert "tool_call" in kinds
        assert "tool_result" in kinds
