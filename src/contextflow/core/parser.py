from __future__ import annotations

import json
import re
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from contextflow.exceptions import ParseError  # noqa: F401 – re-exported for backward compat


TModel = TypeVar("TModel", bound=BaseModel)


class ResponseParser(Generic[TModel]):
    """Parses raw LLM output into JSON dictionaries or Pydantic models."""

    @staticmethod
    def parse_json(raw_text: str) -> dict:
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ParseError(f"Invalid JSON output: {exc}") from exc

        if not isinstance(parsed, dict):
            raise ParseError("Expected a JSON object at top-level.")

        return parsed

    @staticmethod
    def parse_model(raw_text: str, model_type: type[TModel]) -> TModel:
        payload = ResponseParser.parse_json(raw_text)
        try:
            return model_type.model_validate(payload)
        except ValidationError as exc:
            raise ParseError(f"Model validation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Tool-call extraction
    # ------------------------------------------------------------------

    @staticmethod
    def parse_tool_call(raw_text: str) -> dict[str, Any] | None:
        """Extract a ``{"name": ..., "args": ...}`` tool-call dict from LLM output.

        The method looks for the common ``{"tool_call": {"name": ..., "args": ...}}``
        pattern.  It tries ``json.loads`` first, then falls back to regex
        extraction from markdown-fenced or mixed text.

        Returns *None* when no tool call is found.
        """
        text = raw_text.strip()

        # 1. Direct parse
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "tool_call" in obj:
                return obj["tool_call"]
        except json.JSONDecodeError:
            pass

        # 2. Fallback — find the first JSON block (handles markdown fences)
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                obj = json.loads(match.group())
                if isinstance(obj, dict) and "tool_call" in obj:
                    return obj["tool_call"]
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def parse_directive(raw_text: str, *, fallback_state: str | None = None) -> dict[str, Any]:
        """Extract a JSON directive ``{"response": ..., "next_state": ...}`` from LLM output.

        Used by self-driven agents that embed state-transition instructions in
        their output.  If no valid JSON is found, returns the raw text as the
        ``"response"`` value with ``"next_state"`` set to *fallback_state*.
        """
        text = raw_text.strip()

        # Direct parse
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # Fallback — extract first JSON object
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                obj = json.loads(match.group())
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        # Last resort — wrap raw text
        return {"response": text, "next_state": fallback_state}
