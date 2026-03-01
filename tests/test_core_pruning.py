"""Tests for contextflow.core.pruning — Pruning strategies."""

from __future__ import annotations

import pytest

from contextflow.core.models import ContextNode, ContextStack, MessageRole
from contextflow.core.pruning import (
    KeepSystemOnlyStrategy,
    DropMiddleStrategy,
    PruningStrategy,
)


# ═══════════════════════════════════════════════════════════════════════════
#  KeepSystemOnlyStrategy
# ═══════════════════════════════════════════════════════════════════════════


class TestKeepSystemOnlyStrategy:
    def test_keeps_only_system_messages(self):
        strategy = KeepSystemOnlyStrategy()
        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.SYSTEM, content="System prompt"))
        stack.add(ContextNode(role=MessageRole.USER, content="Hello"))
        stack.add(ContextNode(role=MessageRole.ASSISTANT, content="World"))

        pruned = strategy.prune(stack, max_tokens=1)
        assert len(pruned.nodes) == 1
        assert pruned.nodes[0].role == MessageRole.SYSTEM

    def test_keeps_multiple_system_messages(self):
        strategy = KeepSystemOnlyStrategy()
        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.SYSTEM, content="Sys 1"))
        stack.add(ContextNode(role=MessageRole.SYSTEM, content="Sys 2"))
        stack.add(ContextNode(role=MessageRole.USER, content="User"))

        pruned = strategy.prune(stack, max_tokens=1)
        assert len(pruned.nodes) == 2

    def test_empty_stack(self):
        strategy = KeepSystemOnlyStrategy()
        stack = ContextStack()
        pruned = strategy.prune(stack, max_tokens=100)
        assert len(pruned.nodes) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  DropMiddleStrategy
# ═══════════════════════════════════════════════════════════════════════════


class TestDropMiddleStrategy:
    def test_no_pruning_when_under_limit(self):
        strategy = DropMiddleStrategy()
        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.SYSTEM, content="S", token_estimate=5))
        stack.add(ContextNode(role=MessageRole.USER, content="U", token_estimate=5))

        pruned = strategy.prune(stack, max_tokens=100)
        assert len(pruned.nodes) == 2

    def test_drops_middle_when_over_limit(self):
        strategy = DropMiddleStrategy()
        stack = ContextStack()
        # Build a stack that clearly exceeds the limit
        for i in range(10):
            stack.add(
                ContextNode(
                    role=MessageRole.USER,
                    content=f"msg {i}",
                    token_estimate=10,
                )
            )
        # 10 * 10 = 100 tokens, limit = 30
        pruned = strategy.prune(stack, max_tokens=30)
        assert len(pruned.nodes) < 10
        assert pruned.estimate_tokens() <= 30

    def test_preserves_first_and_last(self):
        strategy = DropMiddleStrategy()
        stack = ContextStack()
        first = ContextNode(role=MessageRole.SYSTEM, content="first", token_estimate=5)
        middle = ContextNode(role=MessageRole.USER, content="middle", token_estimate=5)
        last = ContextNode(role=MessageRole.ASSISTANT, content="last", token_estimate=5)
        stack.extend([first, middle, last])

        pruned = strategy.prune(stack, max_tokens=10)
        # With limit 10 and 3 nodes at 5 tokens each (15 total), it should
        # drop down until within budget.
        assert len(pruned.nodes) == 2
        assert pruned.nodes[0].content == "first"
        assert pruned.nodes[-1].content == "last"

    def test_two_nodes_never_drops(self):
        strategy = DropMiddleStrategy()
        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.SYSTEM, content="a", token_estimate=100))
        stack.add(ContextNode(role=MessageRole.USER, content="b", token_estimate=100))

        pruned = strategy.prune(stack, max_tokens=1)
        assert len(pruned.nodes) == 2  # Can't go below 2


# ═══════════════════════════════════════════════════════════════════════════
#  ABC enforcement
# ═══════════════════════════════════════════════════════════════════════════


class TestPruningStrategyABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            PruningStrategy()
