from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from contextflow.core.models import ContextStack
from contextflow.providers.client import AsyncLLMClient  # single protocol source

PreInterceptor = Callable[[ContextStack], Awaitable[ContextStack]]
PostInterceptor = Callable[[str], Awaitable[str]]


@dataclass(slots=True)
class AsyncAgentRunner:
    """Runs a single LLM turn with optional pre/post interceptors."""

    llm_client: AsyncLLMClient
    pre_interceptors: list[PreInterceptor] = field(default_factory=list)
    post_interceptors: list[PostInterceptor] = field(default_factory=list)

    async def run(self, stack: ContextStack) -> str:
        active_stack = stack
        for interceptor in self.pre_interceptors:
            active_stack = await interceptor(active_stack)

        output = await self.llm_client.complete(active_stack.render_messages())

        for interceptor in self.post_interceptors:
            output = await interceptor(output)

        return output
