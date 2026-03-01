"""Tests for contextflow.core.models — ContextNode and ContextStack."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from contextflow.core.models import ContextNode, ContextStack, MessageRole


# ═══════════════════════════════════════════════════════════════════════════
#  MessageRole
# ═══════════════════════════════════════════════════════════════════════════


class TestMessageRole:
    def test_values(self):
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL.value == "tool"

    def test_str_enum(self):
        assert str(MessageRole.USER) == "MessageRole.USER" or isinstance(MessageRole.USER, str)


# ═══════════════════════════════════════════════════════════════════════════
#  ContextNode
# ═══════════════════════════════════════════════════════════════════════════


class TestContextNode:
    def test_basic_creation(self):
        node = ContextNode(role=MessageRole.USER, content="Hello")
        assert node.role == MessageRole.USER
        assert node.content == "Hello"
        assert node.metadata == {}
        assert node.priority == 0
        assert node.token_estimate is None
        assert isinstance(node.node_id, str)
        assert isinstance(node.created_at, datetime)

    def test_to_message_simple(self):
        node = ContextNode(role=MessageRole.USER, content="Hello")
        msg = node.to_message()
        assert msg == {"role": "user", "content": "Hello"}

    def test_to_message_with_name_metadata(self):
        node = ContextNode(
            role=MessageRole.TOOL,
            content="result",
            metadata={"name": "get_weather", "tool_call_id": "tc_123"},
        )
        msg = node.to_message()
        assert msg["role"] == "tool"
        assert msg["content"] == "result"
        assert msg["name"] == "get_weather"
        assert msg["tool_call_id"] == "tc_123"

    def test_to_message_ignores_unknown_metadata(self):
        node = ContextNode(
            role=MessageRole.ASSISTANT,
            content="Hi",
            metadata={"custom_key": "ignored"},
        )
        msg = node.to_message()
        assert "custom_key" not in msg
        assert msg == {"role": "assistant", "content": "Hi"}

    def test_unique_node_ids(self):
        n1 = ContextNode(role=MessageRole.USER, content="a")
        n2 = ContextNode(role=MessageRole.USER, content="b")
        assert n1.node_id != n2.node_id

    def test_custom_priority_and_token_estimate(self):
        node = ContextNode(
            role=MessageRole.SYSTEM,
            content="System prompt",
            priority=-100,
            token_estimate=42,
        )
        assert node.priority == -100
        assert node.token_estimate == 42


# ═══════════════════════════════════════════════════════════════════════════
#  ContextStack
# ═══════════════════════════════════════════════════════════════════════════


class TestContextStack:
    def test_empty_stack(self):
        stack = ContextStack()
        assert stack.nodes == []
        assert stack.render_messages() == []
        assert stack.estimate_tokens() == 0

    def test_add_single_node(self):
        stack = ContextStack()
        node = ContextNode(role=MessageRole.USER, content="Hello")
        stack.add(node)
        assert len(stack.nodes) == 1
        assert stack.nodes[0] is node

    def test_extend(self):
        stack = ContextStack()
        nodes = [
            ContextNode(role=MessageRole.USER, content="a"),
            ContextNode(role=MessageRole.ASSISTANT, content="b"),
        ]
        stack.extend(nodes)
        assert len(stack.nodes) == 2

    def test_render_messages_sorted_by_priority_then_time(self):
        """Nodes are sorted by (priority, created_at) during rendering."""
        stack = ContextStack()
        # Add user message (priority 0) first, then system (priority -100).
        user_node = ContextNode(role=MessageRole.USER, content="Hello", priority=0)
        system_node = ContextNode(role=MessageRole.SYSTEM, content="Sys", priority=-100)

        stack.add(user_node)
        stack.add(system_node)

        msgs = stack.render_messages()
        # System (-100) should come before user (0).
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_estimate_tokens_uses_explicit_estimate(self):
        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.USER, content="Hi", token_estimate=10))
        assert stack.estimate_tokens() == 10

    def test_estimate_tokens_fallback_heuristic(self):
        stack = ContextStack()
        content = "a" * 100  # 100 chars → ~25 tokens via len//4 heuristic
        stack.add(ContextNode(role=MessageRole.USER, content=content))
        assert stack.estimate_tokens() == 25

    def test_estimate_tokens_minimum_1(self):
        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.USER, content=""))
        # max(1, 0//4) = 1
        assert stack.estimate_tokens() == 1
