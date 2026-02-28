"""ContextFlow exception hierarchy.

All framework-specific errors inherit from ``ContextFlowError`` so callers
can catch a single base class when desired.
"""

from __future__ import annotations


class ContextFlowError(Exception):
    """Base exception for all ContextFlow errors."""


class ParseError(ContextFlowError, ValueError):
    """Raised when LLM output cannot be parsed into the expected format."""


class ProviderError(ContextFlowError):
    """Raised for LLM provider configuration or communication failures."""


class SandboxError(ContextFlowError):
    """Raised for sandbox initialisation or execution failures."""
