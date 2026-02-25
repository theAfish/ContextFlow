from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("contextflow.lens")


LLMCall = Callable[[list[dict[str, Any]]], Awaitable[str]]


@dataclass(slots=True)
class LensEvent:
    request: list[dict[str, Any]]
    response: str


@dataclass(slots=True)
class LensMiddleware:
    """Transparent debug tracer for LLM request/response payloads."""

    history: list[LensEvent] = field(default_factory=list)

    async def observe(self, messages: list[dict[str, Any]], call: LLMCall) -> str:
        logger.info("lens.request=%s", messages)
        response = await call(messages)
        logger.info("lens.response=%s", response)

        self.history.append(LensEvent(request=messages, response=response))
        return response
