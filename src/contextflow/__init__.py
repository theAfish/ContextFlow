"""ContextFlow public package interface."""

from contextflow.core.models import ContextNode, ContextStack, MessageRole
from contextflow.core.composer import Composer
from contextflow.core.parser import ResponseParser
from contextflow.exceptions import ContextFlowError, ParseError, ProviderError, SandboxError
from contextflow.agents import Agent, AgentRunResult, ToolSpec, AgentStateMachine
from contextflow.sandbox import AgentSandbox, SandboxExecutionResult, SandboxFileEntry
from contextflow.providers import (
    ProviderConfig,
    OpenAICompatibleClient,
    create_client,
    resolve_api_key,
    resolve_base_url,
)
from contextflow.streaming import StreamEvent
from contextflow.debug import DebugSession, launch_debug

__all__ = [
    # Core
    "ContextNode",
    "ContextStack",
    "MessageRole",
    "Composer",
    "ResponseParser",
    # Exceptions
    "ContextFlowError",
    "ParseError",
    "ProviderError",
    "SandboxError",
    # Agent
    "Agent",
    "ToolSpec",
    "AgentRunResult",
    "AgentStateMachine",
    # Sandbox
    "AgentSandbox",
    "SandboxExecutionResult",
    "SandboxFileEntry",
    # Providers
    "ProviderConfig",
    "OpenAICompatibleClient",
    "create_client",
    "resolve_api_key",
    "resolve_base_url",
    # Streaming
    "StreamEvent",
    # Debug
    "DebugSession",
    "launch_debug",
]
