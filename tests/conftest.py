"""Shared test helpers and fixtures for the ContextFlow test suite."""

from __future__ import annotations

from typing import Any
from collections.abc import AsyncIterator
from contextflow.streaming import StreamEvent


class FakeLLMClient:
    """Minimal mock that satisfies the ``AsyncLLMClient`` protocol.

    By default, returns the canned ``response`` string.  When ``responses``
    (a list) is provided, returns them in order (cycling back to the last
    one if exhausted).
    """

    def __init__(
        self,
        response: str = "Hello from fake LLM",
        *,
        responses: list[str] | None = None,
    ) -> None:
        self._responses = responses or [response]
        self._call_count = 0
        self.last_messages: list[dict[str, Any]] | None = None

    async def complete(self, messages: list[dict[str, Any]]) -> str:
        self.last_messages = messages
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]

    async def stream(self, messages: list[dict[str, Any]]) -> AsyncIterator[StreamEvent]:
        text = await self.complete(messages)
        for char in text:
            yield StreamEvent(kind="content", text=char)
        yield StreamEvent(kind="done")
