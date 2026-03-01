"""Tests for contextflow.core.parser — ResponseParser."""

from __future__ import annotations

import pytest

from contextflow.core.parser import ResponseParser
from contextflow.exceptions import ParseError
from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════════════════════
#  parse_json
# ═══════════════════════════════════════════════════════════════════════════


class TestParseJson:
    def test_valid_json_object(self):
        result = ResponseParser.parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_invalid_json_raises_parse_error(self):
        with pytest.raises(ParseError, match="Invalid JSON"):
            ResponseParser.parse_json("not json")

    def test_json_array_raises_parse_error(self):
        with pytest.raises(ParseError, match="Expected a JSON object"):
            ResponseParser.parse_json('[1, 2, 3]')

    def test_nested_json(self):
        raw = '{"a": {"b": [1, 2]}}'
        result = ResponseParser.parse_json(raw)
        assert result["a"]["b"] == [1, 2]


# ═══════════════════════════════════════════════════════════════════════════
#  parse_model
# ═══════════════════════════════════════════════════════════════════════════


class WeatherResponse(BaseModel):
    city: str
    temperature: float


class TestParseModel:
    def test_valid_model(self):
        raw = '{"city": "Tokyo", "temperature": 22.5}'
        result = ResponseParser.parse_model(raw, WeatherResponse)
        assert isinstance(result, WeatherResponse)
        assert result.city == "Tokyo"
        assert result.temperature == 22.5

    def test_invalid_model_raises_parse_error(self):
        with pytest.raises(ParseError, match="Model validation failed"):
            ResponseParser.parse_model('{"city": "Tokyo"}', WeatherResponse)

    def test_invalid_json_raises_parse_error(self):
        with pytest.raises(ParseError):
            ResponseParser.parse_model("bad json", WeatherResponse)


# ═══════════════════════════════════════════════════════════════════════════
#  parse_tool_call
# ═══════════════════════════════════════════════════════════════════════════


class TestParseToolCall:
    def test_direct_json_tool_call(self):
        raw = '{"tool_call": {"name": "get_weather", "args": {"city": "Tokyo"}}}'
        result = ResponseParser.parse_tool_call(raw)
        assert result is not None
        assert result["name"] == "get_weather"
        assert result["args"]["city"] == "Tokyo"

    def test_tool_call_in_markdown_fence(self):
        raw = """Here's my response:
```json
{"tool_call": {"name": "search", "args": {"q": "hello"}}}
```"""
        result = ResponseParser.parse_tool_call(raw)
        assert result is not None
        assert result["name"] == "search"

    def test_no_tool_call_returns_none(self):
        raw = "This is just a normal response, no tool call."
        result = ResponseParser.parse_tool_call(raw)
        assert result is None

    def test_json_without_tool_call_key_returns_none(self):
        raw = '{"key": "value"}'
        result = ResponseParser.parse_tool_call(raw)
        assert result is None

    def test_embedded_json_in_text(self):
        raw = 'Let me check that. {"tool_call": {"name": "calc", "args": {"n": 1}}}'
        result = ResponseParser.parse_tool_call(raw)
        assert result is not None
        assert result["name"] == "calc"


# ═══════════════════════════════════════════════════════════════════════════
#  parse_directive
# ═══════════════════════════════════════════════════════════════════════════


class TestParseDirective:
    def test_valid_json_directive(self):
        raw = '{"response": "Hello", "next_state": "idle"}'
        result = ResponseParser.parse_directive(raw)
        assert result["response"] == "Hello"
        assert result["next_state"] == "idle"

    def test_fallback_wraps_raw_text(self):
        raw = "Just a plain answer, no JSON."
        result = ResponseParser.parse_directive(raw, fallback_state="error")
        assert result["response"] == raw
        assert result["next_state"] == "error"

    def test_fallback_state_none_by_default(self):
        raw = "plain text"
        result = ResponseParser.parse_directive(raw)
        assert result["next_state"] is None

    def test_embedded_json_directive(self):
        raw = 'Some preamble {"response": "Done", "next_state": "finished"}'
        result = ResponseParser.parse_directive(raw)
        assert result["response"] == "Done"
        assert result["next_state"] == "finished"
