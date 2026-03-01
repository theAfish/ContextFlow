"""Tests for contextflow.streaming.events — StreamEvent, extract_*, normalize_chunk."""

from __future__ import annotations

import pytest

from contextflow.streaming.events import (
    StreamEvent,
    extract_content,
    extract_reasoning,
    normalize_chunk,
)


# ═══════════════════════════════════════════════════════════════════════════
#  StreamEvent
# ═══════════════════════════════════════════════════════════════════════════


class TestStreamEvent:
    def test_basic_creation(self):
        evt = StreamEvent(kind="content", text="hello")
        assert evt.kind == "content"
        assert evt.text == "hello"
        assert evt.metadata is None

    def test_metadata(self):
        evt = StreamEvent(kind="meta", metadata={"key": "val"})
        assert evt.metadata == {"key": "val"}

    def test_defaults(self):
        evt = StreamEvent(kind="done")
        assert evt.text == ""
        assert evt.metadata is None


# ═══════════════════════════════════════════════════════════════════════════
#  extract_content — dict-based chunks
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractContentDict:
    def test_valid_dict_chunk(self):
        chunk = {"choices": [{"delta": {"content": "Hello"}}]}
        assert extract_content(chunk) == "Hello"

    def test_empty_choices(self):
        chunk = {"choices": []}
        assert extract_content(chunk) == ""

    def test_no_choices_key(self):
        chunk = {}
        assert extract_content(chunk) == ""

    def test_no_content_in_delta(self):
        chunk = {"choices": [{"delta": {}}]}
        assert extract_content(chunk) == ""

    def test_none_content(self):
        chunk = {"choices": [{"delta": {"content": None}}]}
        assert extract_content(chunk) == ""


# ═══════════════════════════════════════════════════════════════════════════
#  extract_content — object-based chunks
# ═══════════════════════════════════════════════════════════════════════════


class _Delta:
    def __init__(self, content=None, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content


class _Choice:
    def __init__(self, delta):
        self.delta = delta


class _Chunk:
    def __init__(self, choices):
        self.choices = choices


class TestExtractContentObject:
    def test_valid_object_chunk(self):
        chunk = _Chunk([_Choice(_Delta(content="World"))])
        assert extract_content(chunk) == "World"

    def test_empty_choices_object(self):
        chunk = _Chunk([])
        assert extract_content(chunk) == ""

    def test_no_delta(self):
        choice = _Choice(delta=None)
        chunk = _Chunk([choice])
        assert extract_content(chunk) == ""


# ═══════════════════════════════════════════════════════════════════════════
#  extract_reasoning
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractReasoning:
    def test_dict_reasoning(self):
        chunk = {"choices": [{"delta": {"reasoning_content": "thinking..."}}]}
        assert extract_reasoning(chunk) == "thinking..."

    def test_object_reasoning(self):
        chunk = _Chunk([_Choice(_Delta(reasoning_content="step1"))])
        assert extract_reasoning(chunk) == "step1"

    def test_no_reasoning_dict(self):
        chunk = {"choices": [{"delta": {"content": "normal"}}]}
        assert extract_reasoning(chunk) == ""

    def test_no_reasoning_object(self):
        chunk = _Chunk([_Choice(_Delta(content="normal"))])
        assert extract_reasoning(chunk) == ""


# ═══════════════════════════════════════════════════════════════════════════
#  normalize_chunk
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalizeChunk:
    def test_content_chunk(self):
        chunk = {"choices": [{"delta": {"content": "Hello"}}]}
        events = normalize_chunk(chunk)
        assert len(events) == 1
        assert events[0].kind == "content"
        assert events[0].text == "Hello"

    def test_reasoning_chunk(self):
        chunk = {"choices": [{"delta": {"reasoning_content": "think"}}]}
        events = normalize_chunk(chunk)
        assert len(events) == 1
        assert events[0].kind == "reasoning"
        assert events[0].text == "think"

    def test_both_reasoning_and_content(self):
        chunk = {
            "choices": [
                {"delta": {"content": "text", "reasoning_content": "thought"}}
            ]
        }
        events = normalize_chunk(chunk)
        assert len(events) == 2
        kinds = {e.kind for e in events}
        assert kinds == {"content", "reasoning"}

    def test_empty_chunk_returns_meta(self):
        chunk = {"choices": [{"delta": {}}]}
        events = normalize_chunk(chunk)
        assert len(events) == 1
        assert events[0].kind == "meta"
