from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from contextflow.core.models import ContextStack


class AsyncLLMClient(Protocol):
    async def complete(self, messages: list[dict[str, Any]]) -> str: ...


PreInterceptor = Callable[[ContextStack], Awaitable[ContextStack]]
PostInterceptor = Callable[[str], Awaitable[str]]


@dataclass(slots=True)
class AsyncAgentRunner:
    llm_client: AsyncLLMClient
    pre_interceptors: list[PreInterceptor]
    post_interceptors: list[PostInterceptor]

    async def run(self, stack: ContextStack) -> str:
        active_stack = stack
        for interceptor in self.pre_interceptors:
            active_stack = await interceptor(active_stack)

        output = await self.llm_client.complete(active_stack.render_messages())

        for interceptor in self.post_interceptors:
            output = await interceptor(output)

        return output
