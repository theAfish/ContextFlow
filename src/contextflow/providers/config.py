from __future__ import annotations

import os
from pydantic import BaseModel, ConfigDict, Field, model_validator


SUPPORTED_BACKENDS = {"openai", "litellm"}


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
        resolved_key = (
            api_key
            or os.getenv("QWEN_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or "dummy"
        )
        return cls(
            backend=backend,
            model=model or os.getenv("QWEN_MODEL", "qwen-flash"),
            base_url=(
                base_url
                or os.getenv("QWEN_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            api_key=resolved_key,
            enable_thinking=enable_thinking,
            temperature=temperature,
        )
