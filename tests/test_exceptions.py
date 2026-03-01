"""Tests for contextflow.exceptions — Exception hierarchy."""

from __future__ import annotations

from contextflow.exceptions import (
    ContextFlowError,
    ParseError,
    ProviderError,
    SandboxError,
)


class TestExceptionHierarchy:
    def test_context_flow_error_is_exception(self):
        assert issubclass(ContextFlowError, Exception)

    def test_parse_error_inherits_context_flow_and_value(self):
        assert issubclass(ParseError, ContextFlowError)
        assert issubclass(ParseError, ValueError)

    def test_provider_error_inherits_context_flow(self):
        assert issubclass(ProviderError, ContextFlowError)

    def test_sandbox_error_inherits_context_flow(self):
        assert issubclass(SandboxError, ContextFlowError)

    def test_catch_all_with_base(self):
        """All specific errors should be caught by catching ContextFlowError."""
        for exc_cls in (ParseError, ProviderError, SandboxError):
            try:
                raise exc_cls("test")
            except ContextFlowError:
                pass  # expected

    def test_error_message(self):
        err = ParseError("bad json")
        assert str(err) == "bad json"
