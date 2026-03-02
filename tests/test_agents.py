"""Tests for contextflow.agents.agent — Agent, ToolSpec, ChatSession."""

from __future__ import annotations

import pytest
import asyncio
from typing import Any

from contextflow.agents.agent import Agent, AgentRunResult, ToolSpec, ChatSession
from contextflow.agents.state_machine import AgentStateMachine, RunBlockedByState
from contextflow.core.models import ContextNode, MessageRole

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conftest import FakeLLMClient


# ═══════════════════════════════════════════════════════════════════════════
#  ToolSpec
# ═══════════════════════════════════════════════════════════════════════════


def sample_tool(city: str, unit: str = "celsius") -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 22{unit}"


def no_docstring_tool(x: int) -> int:
    return x + 1


class TestToolSpec:
    def test_from_callable(self):
        spec = ToolSpec.from_callable(sample_tool)
        assert spec.name == "sample_tool"
        assert "weather" in spec.description.lower()
        assert len(spec.parameters) == 2
        assert spec.parameters[0]["name"] == "city"
        assert spec.parameters[0]["type"] == "str"

    def test_from_callable_no_docstring(self):
        spec = ToolSpec.from_callable(no_docstring_tool)
        assert spec.description == "No description provided."

    def test_from_callable_parameter_types(self):
        spec = ToolSpec.from_callable(no_docstring_tool)
        assert spec.parameters[0]["type"] == "int"


# ═══════════════════════════════════════════════════════════════════════════
#  Agent — construction
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentConstruction:
    def test_basic_creation(self):
        agent = Agent(
            model="test-model",
            name="TestAgent",
            description="A test agent",
            instruction="Be helpful",
        )
        assert agent.model == "test-model"
        assert agent.name == "TestAgent"
        assert agent.backend == "openai"

    def test_model_identifier_splitting(self):
        agent = Agent(
            model="openai/gpt-4",
            name="GPT",
            description="desc",
            instruction="inst",
        )
        assert agent.backend == "openai"
        assert agent.model == "gpt-4"

    def test_litellm_backend_splitting(self):
        agent = Agent(
            model="litellm/claude-3",
            name="Claude",
            description="desc",
            instruction="inst",
        )
        assert agent.backend == "litellm"
        assert agent.model == "claude-3"

    def test_unsupported_prefix_not_split(self):
        agent = Agent(
            model="custom/mymodel",
            name="A",
            description="d",
            instruction="i",
        )
        assert agent.model == "custom/mymodel"
        assert agent.backend == "openai"

    def test_credentials_auto_resolve_from_env(self, monkeypatch):
        # environment variables should be consulted when no explicit values
        monkeypatch.setenv("QWEN_API_KEY", "env-key")
        monkeypatch.setenv("QWEN_BASE_URL", "http://envurl")
        agent = Agent(
            model="m",
            name="N",
            description="D",
            instruction="I",
        )
        assert agent.api_key == "env-key"
        assert agent.base_url == "http://envurl"

    def test_explicit_credentials_override_env(self, monkeypatch):
        monkeypatch.setenv("QWEN_API_KEY", "env-key")
        monkeypatch.setenv("QWEN_BASE_URL", "http://envurl")
        agent = Agent(
            model="m",
            name="N",
            description="D",
            instruction="I",
            api_key="explicit",
            base_url="http://explicit",
        )
        assert agent.api_key == "explicit"
        assert agent.base_url == "http://explicit"


# ═══════════════════════════════════════════════════════════════════════════
#  Agent — tool execution
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentToolExecution:
    def _make_agent_with_tools(self):
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        def fail_tool() -> None:
            """A tool that always fails."""
            raise ValueError("boom")

        return Agent(
            model="test",
            name="ToolAgent",
            description="Agent with tools",
            instruction="Use tools",
            tools=[add, fail_tool],
        )

    def test_tool_specs(self):
        agent = self._make_agent_with_tools()
        specs = agent.tool_specs()
        assert len(specs) == 2
        names = {s.name for s in specs}
        assert "add" in names
        assert "fail_tool" in names

    def test_tool_map(self):
        agent = self._make_agent_with_tools()
        m = agent.tool_map()
        assert "add" in m
        assert callable(m["add"])

    def test_execute_tool_success(self):
        agent = self._make_agent_with_tools()
        result = agent.execute_tool("add", {"a": 2, "b": 3})
        assert result == {"ok": True, "result": 5}

    def test_execute_tool_failure(self):
        agent = self._make_agent_with_tools()
        result = agent.execute_tool("fail_tool", {})
        assert result["ok"] is False
        assert "boom" in result["error"]

    def test_execute_unknown_tool(self):
        agent = self._make_agent_with_tools()
        result = agent.execute_tool("nonexistent", {})
        assert result["ok"] is False
        assert "Unknown tool" in result["error"]


# ═══════════════════════════════════════════════════════════════════════════
#  Agent — message construction
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentMessages:
    def test_system_prompt_includes_name_and_instruction(self):
        agent = Agent(
            model="test",
            name="TestBot",
            description="desc",
            instruction="Be accurate",
        )
        prompt = agent.system_prompt()
        assert "TestBot" in prompt
        assert "Be accurate" in prompt

    def test_system_prompt_includes_tool_info(self):
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}"

        agent = Agent(
            model="test",
            name="Bot",
            description="desc",
            instruction="inst",
            tools=[greet],
        )
        prompt = agent.system_prompt()
        assert "greet" in prompt
        assert "Greet someone" in prompt

    def test_system_prompt_no_tools(self):
        agent = Agent(
            model="test",
            name="Bot",
            description="desc",
            instruction="inst",
        )
        prompt = agent.system_prompt()
        assert "No tools available" in prompt

    def test_build_messages(self):
        agent = Agent(
            model="test",
            name="Bot",
            description="desc",
            instruction="inst",
        )
        msgs = agent.build_messages("Hello!")
        assert len(msgs) >= 2  # system + user
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "Hello!"

    def test_build_messages_with_history(self):
        agent = Agent(
            model="test",
            name="Bot",
            description="desc",
            instruction="inst",
        )
        history = [ContextNode(role=MessageRole.USER, content="prev")]
        msgs = agent.build_messages("Hello!", history=history)
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 2

    def test_dict_instruction_selects_by_state(self):
        sm = AgentStateMachine(initial="greeting", transitions=None)
        agent = Agent(
            model="test",
            name="Bot",
            description="desc",
            instruction={"greeting": "Say hello", "farewell": "Say goodbye", "*": "Default"},
            state_machine=sm,
        )
        prompt = agent.system_prompt()
        assert "Say hello" in prompt

    def test_dict_instruction_wildcard_fallback(self):
        sm = AgentStateMachine(initial="unknown_state", transitions=None)
        agent = Agent(
            model="test",
            name="Bot",
            description="desc",
            instruction={"greeting": "Say hello", "*": "Default instruction"},
            state_machine=sm,
        )
        prompt = agent.system_prompt()
        assert "Default instruction" in prompt


# ═══════════════════════════════════════════════════════════════════════════
#  Agent — run_once
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentRunOnce:
    @pytest.mark.asyncio
    async def test_run_once_basic(self):
        fake_client = FakeLLMClient(response="Test response")
        agent = Agent(
            model="test",
            name="Bot",
            description="desc",
            instruction="inst",
            llm_client=fake_client,
        )
        result = await agent.run_once(user_input="Hello")
        assert isinstance(result, AgentRunResult)
        assert result.output_text == "Test response"
        assert len(result.messages) >= 2

    @pytest.mark.asyncio
    async def test_run_once_blocked_by_state(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        sm.allow_run_when("running")

        agent = Agent(
            model="test",
            name="Bot",
            description="desc",
            instruction="inst",
            llm_client=FakeLLMClient(),
            state_machine=sm,
        )
        with pytest.raises(RunBlockedByState):
            await agent.run_once(user_input="Hello")


# ═══════════════════════════════════════════════════════════════════════════
#  Agent — state machine helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentStateMachineHelpers:
    def test_state_property_no_machine(self):
        agent = Agent(model="t", name="A", description="d", instruction="i")
        assert agent.state is None

    def test_state_property_with_machine(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        agent = Agent(model="t", name="A", description="d", instruction="i", state_machine=sm)
        assert agent.state == "idle"

    @pytest.mark.asyncio
    async def test_transition_to(self):
        sm = AgentStateMachine(
            initial="idle",
            transitions={"idle": ["running"], "running": ["done"]},
        )
        agent = Agent(model="t", name="A", description="d", instruction="i", state_machine=sm)
        result = await agent.transition_to("running")
        assert result == "running"
        assert agent.state == "running"

    @pytest.mark.asyncio
    async def test_transition_to_no_machine_raises(self):
        agent = Agent(model="t", name="A", description="d", instruction="i")
        with pytest.raises(RuntimeError, match="No state machine"):
            await agent.transition_to("running")

    @pytest.mark.asyncio
    async def test_force_transition(self):
        sm = AgentStateMachine(
            initial="idle",
            transitions={"idle": ["running"]},
        )
        agent = Agent(model="t", name="A", description="d", instruction="i", state_machine=sm)
        # Force to a state that's not in the transition table
        result = await agent.force_transition("error")
        assert result == "error"

    @pytest.mark.asyncio
    async def test_force_transition_no_machine_raises(self):
        agent = Agent(model="t", name="A", description="d", instruction="i")
        with pytest.raises(RuntimeError, match="No state machine"):
            await agent.force_transition("error")


# ═══════════════════════════════════════════════════════════════════════════
#  ChatSession
# ═══════════════════════════════════════════════════════════════════════════


class TestChatSession:
    @pytest.mark.asyncio
    async def test_send(self):
        fake = FakeLLMClient(response="Reply 1")
        agent = Agent(
            model="test", name="Bot", description="d", instruction="i",
            llm_client=fake,
        )
        session = ChatSession(agent)
        answer = await session.send("Hello")
        assert answer == "Reply 1"
        assert len(session.history) == 2
        assert session.history[0].role == MessageRole.USER
        assert session.history[1].role == MessageRole.ASSISTANT

    @pytest.mark.asyncio
    async def test_send_multiple(self):
        fake = FakeLLMClient(responses=["Reply 1", "Reply 2"])
        agent = Agent(
            model="test", name="Bot", description="d", instruction="i",
            llm_client=fake,
        )
        session = ChatSession(agent)
        await session.send("First")
        answer = await session.send("Second")
        assert answer == "Reply 2"
        assert len(session.history) == 4

    @pytest.mark.asyncio
    async def test_clear(self):
        fake = FakeLLMClient(response="Reply")
        agent = Agent(
            model="test", name="Bot", description="d", instruction="i",
            llm_client=fake,
        )
        session = ChatSession(agent)
        await session.send("msg")
        assert len(session.history) == 2
        session.clear()
        assert len(session.history) == 0
