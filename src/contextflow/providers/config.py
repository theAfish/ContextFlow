from __future__ import annotations

import os
from pydantic import BaseModel, ConfigDict, Field, model_validator


SUPPORTED_BACKENDS = {"openai", "litellm"}

_ENV_KEY_CHAIN = ("QWEN_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY")
_ENV_URL_CHAIN = ("QWEN_BASE_URL", "OPENAI_BASE_URL")
_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen-flash"


def resolve_api_key(explicit: str | None = None) -> str:
    """Resolve an API key from an explicit value or well-known env vars."""
    if explicit:
        return explicit
    for var in _ENV_KEY_CHAIN:
        val = os.getenv(var)
        if val:
            return val
    return "dummy"


def resolve_base_url(explicit: str | None = None) -> str:
    """Resolve a base URL from an explicit value or well-known env vars."""
    if explicit:
        return explicit
    for var in _ENV_URL_CHAIN:
        val = os.getenv(var)
        if val:
            return val
    return _DEFAULT_BASE_URL


def split_model_identifier(model: str) -> tuple[str | None, str]:
    if "/" not in model:
        return None, model

    prefix, rest = model.split("/", 1)
    if prefix in SUPPORTED_BACKENDS and rest:
        return prefix, rest

    return None, model


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    backend: str = Field(default="openai", pattern=r"^(openai|litellm)$")
    model: str = "qwen-flash"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = "dummy"
    enable_thinking: bool = True
    temperature: float = 1.0

    @model_validator(mode="after")
    def normalize_model_identifier(self) -> "ProviderConfig":
        parsed_backend, parsed_model = split_model_identifier(self.model)
        if parsed_backend is not None:
            self.backend = parsed_backend
            self.model = parsed_model
        return self

    @classmethod
    def from_env(
        cls,
        *,
        backend: str = "openai",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        enable_thinking: bool = True,
        temperature: float = 1.0,
    ) -> "ProviderConfig":
        return cls(
            backend=backend,
            model=model or os.getenv("QWEN_MODEL", _DEFAULT_MODEL),
            base_url=resolve_base_url(base_url),
            api_key=resolve_api_key(api_key),
            enable_thinking=enable_thinking,
            temperature=temperature,
        )
