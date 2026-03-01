"""Tests for contextflow.runners — AsyncAgentRunner."""

from __future__ import annotations

import pytest

from contextflow.core.models import ContextNode, ContextStack, MessageRole
from contextflow.runners.async_agent import AsyncAgentRunner

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from conftest import FakeLLMClient


# ═══════════════════════════════════════════════════════════════════════════
#  AsyncAgentRunner
# ═══════════════════════════════════════════════════════════════════════════


class TestAsyncAgentRunner:
    @pytest.mark.asyncio
    async def test_basic_run(self):
        client = FakeLLMClient(response="LLM says hello")
        runner = AsyncAgentRunner(llm_client=client)

        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.SYSTEM, content="System"))
        stack.add(ContextNode(role=MessageRole.USER, content="Hi"))

        output = await runner.run(stack)
        assert output == "LLM says hello"

    @pytest.mark.asyncio
    async def test_messages_passed_to_client(self):
        client = FakeLLMClient(response="ok")
        runner = AsyncAgentRunner(llm_client=client)

        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.USER, content="Test"))

        await runner.run(stack)
        assert client.last_messages is not None
        assert len(client.last_messages) == 1
        assert client.last_messages[0]["content"] == "Test"

    @pytest.mark.asyncio
    async def test_pre_interceptor(self):
        client = FakeLLMClient(response="ok")

        async def add_system(stack: ContextStack) -> ContextStack:
            stack.add(ContextNode(role=MessageRole.SYSTEM, content="Injected", priority=-200))
            return stack

        runner = AsyncAgentRunner(llm_client=client, pre_interceptors=[add_system])

        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.USER, content="Hi"))

        await runner.run(stack)
        # The client should have received 2 messages: injected system + user
        assert len(client.last_messages) == 2

    @pytest.mark.asyncio
    async def test_post_interceptor(self):
        client = FakeLLMClient(response="raw output")

        async def uppercase(output: str) -> str:
            return output.upper()

        runner = AsyncAgentRunner(llm_client=client, post_interceptors=[uppercase])

        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.USER, content="Hi"))

        output = await runner.run(stack)
        assert output == "RAW OUTPUT"

    @pytest.mark.asyncio
    async def test_chained_interceptors(self):
        client = FakeLLMClient(response="hello")

        async def exclaim(output: str) -> str:
            return output + "!"

        async def wrap(output: str) -> str:
            return f"[{output}]"

        runner = AsyncAgentRunner(
            llm_client=client,
            post_interceptors=[exclaim, wrap],
        )

        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.USER, content="Hi"))

        output = await runner.run(stack)
        assert output == "[hello!]"
