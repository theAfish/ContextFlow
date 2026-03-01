"""Tests for contextflow.providers.config — ProviderConfig, resolve_api_key, etc."""

from __future__ import annotations

import os
import pytest

from contextflow.providers.config import (
    ProviderConfig,
    resolve_api_key,
    resolve_base_url,
    split_model_identifier,
    _DEFAULT_BASE_URL,
)


# ═══════════════════════════════════════════════════════════════════════════
#  split_model_identifier
# ═══════════════════════════════════════════════════════════════════════════


class TestSplitModelIdentifier:
    def test_plain_model_name(self):
        backend, model = split_model_identifier("qwen-flash")
        assert backend is None
        assert model == "qwen-flash"

    def test_openai_prefix(self):
        backend, model = split_model_identifier("openai/gpt-4")
        assert backend == "openai"
        assert model == "gpt-4"

    def test_litellm_prefix(self):
        backend, model = split_model_identifier("litellm/claude-3")
        assert backend == "litellm"
        assert model == "claude-3"

    def test_unknown_prefix_not_split(self):
        backend, model = split_model_identifier("custom/model")
        assert backend is None
        assert model == "custom/model"

    def test_no_slash(self):
        backend, model = split_model_identifier("gpt-4o")
        assert backend is None
        assert model == "gpt-4o"


# ═══════════════════════════════════════════════════════════════════════════
#  resolve_api_key
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveApiKey:
    def test_explicit_key(self):
        assert resolve_api_key("my-key") == "my-key"

    def test_env_key(self, monkeypatch):
        monkeypatch.setenv("QWEN_API_KEY", "env-qwen")
        assert resolve_api_key(None) == "env-qwen"

    def test_env_key_chain(self, monkeypatch):
        monkeypatch.delenv("QWEN_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "env-openai")
        assert resolve_api_key(None) == "env-openai"

    def test_default_dummy(self, monkeypatch):
        for var in ("QWEN_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        assert resolve_api_key(None) == "dummy"


# ═══════════════════════════════════════════════════════════════════════════
#  resolve_base_url
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveBaseUrl:
    def test_explicit_url(self):
        assert resolve_base_url("http://custom") == "http://custom"

    def test_env_url(self, monkeypatch):
        monkeypatch.setenv("QWEN_BASE_URL", "http://envqwen")
        assert resolve_base_url(None) == "http://envqwen"

    def test_default_url(self, monkeypatch):
        for var in ("QWEN_BASE_URL", "OPENAI_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        assert resolve_base_url(None) == _DEFAULT_BASE_URL


# ═══════════════════════════════════════════════════════════════════════════
#  ProviderConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestProviderConfig:
    def test_defaults(self):
        config = ProviderConfig()
        assert config.backend == "openai"
        assert config.model == "qwen-flash"
        assert config.enable_thinking is True

    def test_model_identifier_normalization(self):
        config = ProviderConfig(model="openai/gpt-4")
        assert config.backend == "openai"
        assert config.model == "gpt-4"

    def test_litellm_normalization(self):
        config = ProviderConfig(model="litellm/claude-3")
        assert config.backend == "litellm"
        assert config.model == "claude-3"

    def test_invalid_backend_raises(self):
        with pytest.raises(Exception):
            ProviderConfig(backend="unsupported_backend")

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("QWEN_API_KEY", "test-key")
        monkeypatch.setenv("QWEN_BASE_URL", "http://test")
        config = ProviderConfig.from_env(model="test-model")
        assert config.api_key == "test-key"
        assert config.base_url == "http://test"
        assert config.model == "test-model"

    def test_from_env_with_explicit_overrides(self):
        config = ProviderConfig.from_env(
            api_key="explicit-key",
            base_url="http://explicit",
            model="my-model",
        )
        assert config.api_key == "explicit-key"
        assert config.base_url == "http://explicit"

    def test_temperature(self):
        config = ProviderConfig(temperature=0.5)
        assert config.temperature == 0.5
