"""Tests for contextflow.core.composer — Composer slot-filling and composition."""

from __future__ import annotations

import pytest

from contextflow.core.composer import Composer
from contextflow.core.models import ContextNode, ContextStack, MessageRole
from contextflow.core.pruning import KeepSystemOnlyStrategy, DropMiddleStrategy


# ═══════════════════════════════════════════════════════════════════════════
#  Slot filling
# ═══════════════════════════════════════════════════════════════════════════


class TestFillSlots:
    def test_static_slot(self):
        composer = Composer()
        result = composer.fill_slots("Hello {{ name }}", static_slots={"name": "Alice"})
        assert result == "Hello Alice"

    def test_dynamic_slot(self):
        composer = Composer()
        result = composer.fill_slots(
            "Time is {{ current_time }}",
            dynamic_slots={"current_time": lambda: "12:00"},
        )
        assert result == "Time is 12:00"

    def test_static_takes_precedence_over_dynamic(self):
        composer = Composer()
        result = composer.fill_slots(
            "{{ val }}",
            static_slots={"val": "static"},
            dynamic_slots={"val": lambda: "dynamic"},
        )
        assert result == "static"

    def test_unresolved_slot_kept_verbatim(self):
        composer = Composer()
        result = composer.fill_slots("Hi {{ unknown_slot }}", static_slots={})
        assert result == "Hi {{ unknown_slot }}"

    def test_multiple_slots(self):
        composer = Composer()
        result = composer.fill_slots(
            "{{ greeting }}, {{ name }}!",
            static_slots={"greeting": "Hi", "name": "Bob"},
        )
        assert result == "Hi, Bob!"

    def test_slot_with_hyphens_and_underscores(self):
        composer = Composer()
        result = composer.fill_slots(
            "{{ my-var }} and {{ my_var2 }}",
            static_slots={"my-var": "A", "my_var2": "B"},
        )
        assert result == "A and B"


# ═══════════════════════════════════════════════════════════════════════════
#  Composition
# ═══════════════════════════════════════════════════════════════════════════


class TestCompose:
    def test_system_prompt_only(self):
        composer = Composer()
        stack = composer.compose(system_prompt="Be helpful.")
        msgs = stack.render_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Be helpful."

    def test_system_prompt_with_slots(self):
        composer = Composer()
        stack = composer.compose(
            system_prompt="Time: {{ time }}",
            slots={"time": "now"},
        )
        msgs = stack.render_messages()
        assert msgs[0]["content"] == "Time: now"

    def test_system_prompt_with_dynamic_slots(self):
        composer = Composer()
        stack = composer.compose(
            system_prompt="Val={{ v }}",
            dynamic_slots={"v": lambda: "42"},
        )
        msgs = stack.render_messages()
        assert "42" in msgs[0]["content"]

    def test_rag_snippets(self):
        composer = Composer()
        stack = composer.compose(
            system_prompt="System",
            rag_snippets=["fact1", "fact2"],
        )
        msgs = stack.render_messages()
        assert len(msgs) == 2
        # System prompt has priority -100, RAG has -50 → system first
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "System"
        assert "fact1" in msgs[1]["content"]
        assert "fact2" in msgs[1]["content"]

    def test_history_nodes(self):
        composer = Composer()
        history = [
            ContextNode(role=MessageRole.USER, content="Hi"),
            ContextNode(role=MessageRole.ASSISTANT, content="Hello!"),
        ]
        stack = composer.compose(system_prompt="Sys", history=history)
        msgs = stack.render_messages()
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_injected_nodes(self):
        composer = Composer()
        injected = [ContextNode(role=MessageRole.USER, content="injected")]
        stack = composer.compose(system_prompt="Sys", injected_nodes=injected)
        msgs = stack.render_messages()
        assert any(m["content"] == "injected" for m in msgs)

    def test_full_composition(self):
        composer = Composer()
        history = [ContextNode(role=MessageRole.USER, content="prev")]
        stack = composer.compose(
            system_prompt="Sys {{ v }}",
            history=history,
            rag_snippets=["snippet"],
            slots={"v": "X"},
        )
        msgs = stack.render_messages()
        assert len(msgs) >= 3  # system + rag + history


# ═══════════════════════════════════════════════════════════════════════════
#  Render with pruning
# ═══════════════════════════════════════════════════════════════════════════


class TestRender:
    def test_render_without_pruning(self):
        composer = Composer()
        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.SYSTEM, content="Hello", priority=-100))
        stack.add(ContextNode(role=MessageRole.USER, content="Hi"))
        msgs = composer.render(stack)
        assert len(msgs) == 2

    def test_render_with_keep_system_only_strategy(self):
        composer = Composer(pruning_strategy=KeepSystemOnlyStrategy())
        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.SYSTEM, content="System"))
        stack.add(ContextNode(role=MessageRole.USER, content="User"))
        stack.add(ContextNode(role=MessageRole.ASSISTANT, content="Asst"))
        msgs = composer.render(stack, max_tokens=1)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_render_no_pruning_when_no_max_tokens(self):
        composer = Composer(pruning_strategy=KeepSystemOnlyStrategy())
        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.SYSTEM, content="System"))
        stack.add(ContextNode(role=MessageRole.USER, content="User"))
        # No max_tokens → no pruning
        msgs = composer.render(stack)
        assert len(msgs) == 2

    def test_render_no_pruning_when_no_strategy(self):
        composer = Composer()
        stack = ContextStack()
        stack.add(ContextNode(role=MessageRole.SYSTEM, content="System"))
        stack.add(ContextNode(role=MessageRole.USER, content="User"))
        msgs = composer.render(stack, max_tokens=1)
        assert len(msgs) == 2
