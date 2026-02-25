from __future__ import annotations

import json
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError


class ParseError(ValueError):
    pass


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
