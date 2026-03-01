"""Tests for contextflow.config — Settings."""

from __future__ import annotations

import pytest

from contextflow.config import Settings, settings


class TestSettings:
    def test_default_values(self):
        s = Settings()
        assert s.app_name == "ContextFlow"
        assert s.app_env == "dev"
        assert s.max_context_tokens == 16_000

    def test_valid_envs(self):
        for env in ("dev", "test", "prod"):
            s = Settings(app_env=env)
            assert s.app_env == env

    def test_invalid_env_raises(self):
        with pytest.raises(Exception):
            Settings(app_env="staging")

    def test_custom_token_limit(self):
        s = Settings(max_context_tokens=8000)
        assert s.max_context_tokens == 8000

    def test_module_level_singleton(self):
        assert isinstance(settings, Settings)
        assert settings.app_name == "ContextFlow"
