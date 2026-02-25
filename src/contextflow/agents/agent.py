from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from contextflow.core import Composer, ContextNode, MessageRole
from contextflow.providers import ProviderConfig, create_client
from contextflow.runners import AsyncAgentRunner, AsyncLLMClient

ToolFunc = Callable[..., Any]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: list[dict[str, str]]

    @classmethod
    def from_callable(cls, tool: ToolFunc) -> "ToolSpec":
        signature = inspect.signature(tool)
        params: list[dict[str, str]] = []

        for param_name, param in signature.parameters.items():
            annotation = "Any"
            if param.annotation is not inspect._empty:
                annotation = getattr(param.annotation, "__name__", str(param.annotation))
            params.append({"name": param_name, "type": annotation})

        return cls(
            name=tool.__name__,
            description=(inspect.getdoc(tool) or "No description provided.").strip(),
            parameters=params,
        )


@dataclass(slots=True)
class AgentRunResult:
    output_text: str
    messages: list[dict[str, Any]]


@dataclass(slots=True)
class Agent:
    model: str
    name: str
    description: str
    instruction: str
    backend: str = "openai"
    base_url: str | None = None
    api_key: str | None = None
    enable_thinking: bool = True
    temperature: float = 1.0
    llm_client: AsyncLLMClient | None = None
    tools: list[ToolFunc] = field(default_factory=list)
    composer: Composer = field(default_factory=Composer)
    _resolved_llm_client: AsyncLLMClient | None = field(default=None, init=False, repr=False)

    def _build_provider_config(self) -> ProviderConfig:
        return ProviderConfig.from_env(
            backend=self.backend,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            enable_thinking=self.enable_thinking,
            temperature=self.temperature,
        )

    def resolve_llm_client(self) -> AsyncLLMClient:
        if self.llm_client is not None:
            return self.llm_client

        if self._resolved_llm_client is None:
            self._resolved_llm_client = create_client(self._build_provider_config())

        return self._resolved_llm_client

    def tool_specs(self) -> list[ToolSpec]:
        return [ToolSpec.from_callable(tool) for tool in self.tools]

    def system_prompt(self) -> str:
        tool_specs = [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
            for spec in self.tool_specs()
        ]

        tool_block = "No tools available."
        if tool_specs:
            tool_block = json.dumps(tool_specs, ensure_ascii=False, indent=2)

        return (
            f"Your name is {self.name}\n"
            f"Here is some instructions for you: {self.instruction}\n\n"
            "Tools:\n"
            f"{tool_block}"
        )

    def build_messages(
        self,
        user_input: str,
        *,
        history: list[ContextNode] | None = None,
        rag_snippets: list[str] | None = None,
        slots: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        merged_history = list(history or [])
        merged_history.append(ContextNode(role=MessageRole.USER, content=user_input))

        stack = self.composer.compose(
            system_prompt=self.system_prompt(),
            history=merged_history,
            rag_snippets=rag_snippets,
            slots=slots,
        )
        return self.composer.render(stack)

    async def run_once(
        self,
        *,
        user_input: str,
        llm_client: AsyncLLMClient | None = None,
        history: list[ContextNode] | None = None,
        rag_snippets: list[str] | None = None,
        slots: dict[str, str] | None = None,
    ) -> AgentRunResult:
        merged_history = list(history or [])
        merged_history.append(ContextNode(role=MessageRole.USER, content=user_input))

        stack = self.composer.compose(
            system_prompt=self.system_prompt(),
            history=merged_history,
            rag_snippets=rag_snippets,
            slots=slots,
        )
        messages = self.composer.render(stack)

        effective_llm_client = llm_client or self.resolve_llm_client()
        runner = AsyncAgentRunner(llm_client=effective_llm_client, pre_interceptors=[], post_interceptors=[])
        output = await runner.run(stack)
        return AgentRunResult(output_text=output, messages=messages)
