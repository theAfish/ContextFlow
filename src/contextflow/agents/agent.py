from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from contextflow.core import Composer, ContextNode, MessageRole
from contextflow.providers import ProviderConfig, create_client
from contextflow.providers.client import AsyncLLMClient
from contextflow.providers.config import split_model_identifier
from contextflow.runners import AsyncAgentRunner
from contextflow.agents.state_machine import AgentStateMachine, RunBlockedByState

ToolFunc = Callable[..., Any]


@dataclass(slots=True)
class ToolSpec:
    """Introspected description of a callable tool."""

    name: str
    description: str
    parameters: list[dict[str, str]]

    @classmethod
    def from_callable(cls, tool: ToolFunc) -> ToolSpec:
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
    """Returned by ``Agent.run_once``."""

    output_text: str
    messages: list[dict[str, Any]]


@dataclass(slots=True)
class Agent:
    """Lightweight, reusable agent definition.

    Supports ``model="backend/model"`` shorthand (e.g. ``openai/qwen-flash``).
    The *backend* field is auto-resolved from the model string at init time.
    """

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
    state_machine: AgentStateMachine | None = None
    _resolved_llm_client: AsyncLLMClient | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Auto-split ``"openai/qwen-flash"`` into backend + model."""
        parsed_backend, parsed_model = split_model_identifier(self.model)
        if parsed_backend is not None:
            object.__setattr__(self, "backend", parsed_backend)
            object.__setattr__(self, "model", parsed_model)

    # ------------------------------------------------------------------
    # State machine helpers
    # ------------------------------------------------------------------

    @property
    def state(self) -> str | None:
        """Shortcut – current state label, or *None* if no machine attached."""
        if self.state_machine is None:
            return None
        return self.state_machine.current

    async def transition_to(
        self, new_state: str, *, context: dict[str, Any] | None = None
    ) -> str:
        """Transition the attached state machine to *new_state*.

        Raises ``RuntimeError`` if no state machine is attached.
        """
        if self.state_machine is None:
            raise RuntimeError("No state machine attached to this agent.")
        return await self.state_machine.transition_to(new_state, context=context)

    async def force_transition(
        self, new_state: str, *, context: dict[str, Any] | None = None
    ) -> str:
        """Force-transition (skip guards & transition table)."""
        if self.state_machine is None:
            raise RuntimeError("No state machine attached to this agent.")
        return await self.state_machine.force_transition(new_state, context=context)

    # ------------------------------------------------------------------
    # Provider wiring
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Tool introspection & execution
    # ------------------------------------------------------------------

    def tool_specs(self) -> list[ToolSpec]:
        return [ToolSpec.from_callable(tool) for tool in self.tools]

    def tool_map(self) -> dict[str, ToolFunc]:
        """Return ``{name: callable}`` for all registered tools."""
        return {tool.__name__: tool for tool in self.tools}

    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Invoke a registered tool by name. Returns ``{"ok": True, "result": ...}``
        on success, or ``{"ok": False, "error": ...}`` on failure."""
        tools = self.tool_map()
        if tool_name not in tools:
            return {"ok": False, "error": f"Unknown tool: {tool_name}"}
        try:
            output = tools[tool_name](**args)
            return {"ok": True, "result": output}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Prompt / message construction
    # ------------------------------------------------------------------

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

    def _compose_stack(
        self,
        user_input: str,
        *,
        history: list[ContextNode] | None = None,
        rag_snippets: list[str] | None = None,
        slots: dict[str, str] | None = None,
    ):
        """Shared helper used by both ``build_messages`` and ``run_once``."""
        merged_history = list(history or [])
        merged_history.append(ContextNode(role=MessageRole.USER, content=user_input))

        return self.composer.compose(
            system_prompt=self.system_prompt(),
            history=merged_history,
            rag_snippets=rag_snippets,
            slots=slots,
        )

    def build_messages(
        self,
        user_input: str,
        *,
        history: list[ContextNode] | None = None,
        rag_snippets: list[str] | None = None,
        slots: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        stack = self._compose_stack(
            user_input, history=history, rag_snippets=rag_snippets, slots=slots,
        )
        return self.composer.render(stack)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run_once(
        self,
        *,
        user_input: str,
        llm_client: AsyncLLMClient | None = None,
        history: list[ContextNode] | None = None,
        rag_snippets: list[str] | None = None,
        slots: dict[str, str] | None = None,
    ) -> AgentRunResult:
        # --- state-machine run gate ------------------------------------
        if self.state_machine is not None and not self.state_machine.can_run():
            raise RunBlockedByState(
                f"Agent {self.name!r} cannot run in state "
                f"{self.state_machine.current!r}. "
                f"Allowed run-states: {self.state_machine._run_states!r}"
            )
        # ---------------------------------------------------------------

        stack = self._compose_stack(
            user_input, history=history, rag_snippets=rag_snippets, slots=slots,
        )
        messages = self.composer.render(stack)

        effective_llm_client = llm_client or self.resolve_llm_client()
        runner = AsyncAgentRunner(llm_client=effective_llm_client)
        output = await runner.run(stack)
        return AgentRunResult(output_text=output, messages=messages)
