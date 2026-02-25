from __future__ import annotations

import importlib
from collections.abc import AsyncIterator
from typing import Any, Protocol

from contextflow.providers.config import ProviderConfig
from contextflow.streaming import StreamEvent, normalize_chunk


class ModelClient(Protocol):
    async def complete(self, messages: list[dict[str, Any]]) -> str: ...

    async def stream(self, messages: list[dict[str, Any]]) -> AsyncIterator[StreamEvent]: ...


class OpenAICompatibleClient:
    """Unified client for OpenAI-compatible APIs via openai sdk or LiteLLM."""

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config

    async def complete(self, messages: list[dict[str, Any]]) -> str:
        if self._config.backend == "openai":
            openai_module = importlib.import_module("openai")
            async_openai_cls = getattr(openai_module, "AsyncOpenAI")
            client = async_openai_cls(api_key=self._config.api_key, base_url=self._config.base_url)
            response = await client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                temperature=self._config.temperature,
                extra_body={"enable_thinking": self._config.enable_thinking},
            )
            return response.choices[0].message.content or ""

        litellm_module = importlib.import_module("litellm")
        acompletion = getattr(litellm_module, "acompletion")
        response = await acompletion(
            model=self._config.model,
            messages=messages,
            api_key=self._config.api_key,
            api_base=self._config.base_url,
            temperature=self._config.temperature,
            extra_body={"enable_thinking": self._config.enable_thinking},
        )
        return response.choices[0].message.content or ""

    async def stream(self, messages: list[dict[str, Any]]) -> AsyncIterator[StreamEvent]:
        if self._config.backend == "openai":
            openai_module = importlib.import_module("openai")
            async_openai_cls = getattr(openai_module, "AsyncOpenAI")
            client = async_openai_cls(api_key=self._config.api_key, base_url=self._config.base_url)
            stream = await client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                temperature=self._config.temperature,
                extra_body={"enable_thinking": self._config.enable_thinking},
                stream=True,
            )
        else:
            litellm_module = importlib.import_module("litellm")
            acompletion = getattr(litellm_module, "acompletion")
            stream = await acompletion(
                model=self._config.model,
                messages=messages,
                api_key=self._config.api_key,
                api_base=self._config.base_url,
                temperature=self._config.temperature,
                extra_body={"enable_thinking": self._config.enable_thinking},
                stream=True,
            )

        async for chunk in stream:
            for event in normalize_chunk(chunk):
                if event.kind != "meta":
                    yield event

        yield StreamEvent(kind="done")


def create_client(config: ProviderConfig) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(config)
