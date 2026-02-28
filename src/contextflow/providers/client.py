from __future__ import annotations

import importlib
from collections.abc import AsyncIterator
from typing import Any, Protocol

from contextflow.exceptions import ProviderError
from contextflow.providers.config import ProviderConfig
from contextflow.streaming import StreamEvent, normalize_chunk


class ModelClient(Protocol):
    """Protocol that all LLM clients must satisfy."""

    async def complete(self, messages: list[dict[str, Any]]) -> str: ...

    async def stream(self, messages: list[dict[str, Any]]) -> AsyncIterator[StreamEvent]: ...


# Backward-compatible alias used by runners and agents.
AsyncLLMClient = ModelClient


class OpenAICompatibleClient:
    """Unified client for OpenAI-compatible APIs via openai sdk or LiteLLM."""

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Backend helpers – centralise the openai / litellm branching so that
    # ``complete`` and ``stream`` share the same wiring logic.
    # ------------------------------------------------------------------

    def _get_openai_client(self) -> Any:
        """Lazily import and return an ``AsyncOpenAI`` instance."""
        openai_module = importlib.import_module("openai")
        async_openai_cls = getattr(openai_module, "AsyncOpenAI")
        return async_openai_cls(
            api_key=self._config.api_key,
            base_url=self._config.base_url,
        )

    def _common_params(self, messages: list[dict[str, Any]], *, stream: bool = False) -> dict[str, Any]:
        """Build the parameter dict shared by both backends."""
        params: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "extra_body": {"enable_thinking": self._config.enable_thinking},
        }
        if stream:
            params["stream"] = True
        return params

    async def _openai_call(self, messages: list[dict[str, Any]], *, stream: bool = False) -> Any:
        client = self._get_openai_client()
        return await client.chat.completions.create(**self._common_params(messages, stream=stream))

    async def _litellm_call(self, messages: list[dict[str, Any]], *, stream: bool = False) -> Any:
        litellm_module = importlib.import_module("litellm")
        acompletion = getattr(litellm_module, "acompletion")
        params = self._common_params(messages, stream=stream)
        # litellm uses ``api_key`` / ``api_base`` instead of client-level config.
        params.update(api_key=self._config.api_key, api_base=self._config.base_url)
        return await acompletion(**params)

    async def _backend_call(self, messages: list[dict[str, Any]], *, stream: bool = False) -> Any:
        """Dispatch to the correct backend."""
        if self._config.backend == "openai":
            return await self._openai_call(messages, stream=stream)
        return await self._litellm_call(messages, stream=stream)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _should_retry_with_streaming(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "switch to streaming mode" in text
            or "generation timed out" in text
            or ("invalidparameter" in text and "timed out" in text)
        )

    async def _complete_via_stream(self, messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        async for event in self.stream(messages):
            if event.kind == "content" and event.text:
                parts.append(event.text)
        return "".join(parts)

    async def complete(self, messages: list[dict[str, Any]]) -> str:
        try:
            response = await self._backend_call(messages)
            return response.choices[0].message.content or ""
        except Exception as exc:
            if not self._should_retry_with_streaming(exc):
                raise ProviderError(str(exc)) from exc
            return await self._complete_via_stream(messages)

    async def stream(self, messages: list[dict[str, Any]]) -> AsyncIterator[StreamEvent]:
        raw_stream = await self._backend_call(messages, stream=True)

        async for chunk in raw_stream:
            for event in normalize_chunk(chunk):
                if event.kind != "meta":
                    yield event

        yield StreamEvent(kind="done")


def create_client(config: ProviderConfig) -> OpenAICompatibleClient:
    """Factory helper – returns an ``OpenAICompatibleClient`` wired to *config*."""
    return OpenAICompatibleClient(config)
