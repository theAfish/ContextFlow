from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from contextflow.core import Composer, ContextNode, MessageRole
from contextflow.core.parser import ResponseParser
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
    instruction: str | dict[str, str]
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

    def _resolve_instruction(self) -> str:
        """Return the effective instruction string.

        When ``instruction`` is a ``dict[str, str]`` mapping state names to
        instructions, the entry matching the current state machine state is
        returned.  Falls back to the ``"*"`` key or an empty string.
        """
        if isinstance(self.instruction, str):
            return self.instruction

        # dict[str, str] — look up by current state
        current = self.state_machine.current if self.state_machine else None
        if current and current in self.instruction:
            return self.instruction[current]
        # wildcard / default key
        return self.instruction.get("*", "")

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

        instruction = self._resolve_instruction()

        return (
            f"Your name is {self.name}\n"
            f"Here is some instructions for you: {instruction}\n\n"
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

    # ------------------------------------------------------------------
    # Agentic tool-calling loop
    # ------------------------------------------------------------------

    async def run_with_tools(
        self,
        user_input: str,
        *,
        history: list[ContextNode] | None = None,
        max_rounds: int = 6,
        on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_result: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> str:
        """Run the agent in an agentic loop: LLM → parse tool calls →
        execute tools → feed result back → repeat until a final answer.

        Parameters
        ----------
        user_input:
            The initial user message.
        history:
            Optional prior conversation context.
        max_rounds:
            Safety cap on tool-calling iterations.
        on_tool_call:
            Optional callback ``(tool_name, tool_args)`` fired before each tool execution.
        on_tool_result:
            Optional callback ``(tool_name, result_dict)`` fired after each tool execution.

        Returns
        -------
        str
            The agent's final natural-language answer.
        """
        current_history = list(history or [])
        current_input = user_input

        for _ in range(max_rounds):
            result = await self.run_once(
                user_input=current_input,
                history=current_history,
            )

            raw_output = result.output_text
            tool_call = ResponseParser.parse_tool_call(raw_output)

            if tool_call is None:
                # No tool call → final answer
                return raw_output

            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})

            if on_tool_call is not None:
                on_tool_call(tool_name, tool_args)

            tool_result = self.execute_tool(tool_name, tool_args)

            if on_tool_result is not None:
                on_tool_result(tool_name, tool_result)

            # Append exchange to history and loop back
            current_history.append(ContextNode(role=MessageRole.USER, content=current_input))
            current_history.append(ContextNode(role=MessageRole.ASSISTANT, content=raw_output))
            current_input = (
                f"Tool '{tool_name}' returned:\n"
                f"{json.dumps(tool_result, ensure_ascii=False, indent=2)}\n\n"
                "Use this data to answer the user's question in natural language."
            )

        return "[Max tool rounds reached]"

    # ------------------------------------------------------------------
    # Interactive chat REPL
    # ------------------------------------------------------------------

    async def chat_repl(
        self,
        *,
        prompt: str = "You: ",
        exit_commands: tuple[str, ...] = ("exit", "quit"),
        use_tools: bool = False,
        max_tool_rounds: int = 6,
        on_response: Callable[[str], None] | None = None,
    ) -> None:
        """Run an interactive terminal chat loop.

        Parameters
        ----------
        prompt:
            Input prompt string shown to the user.
        exit_commands:
            User inputs (case-insensitive) that terminate the loop.
        use_tools:
            When *True*, uses :meth:`run_with_tools` instead of :meth:`run_once`.
        max_tool_rounds:
            Passed to :meth:`run_with_tools` when *use_tools* is True.
        on_response:
            Optional callback receiving the assistant's text after each turn.
        """
        history: list[ContextNode] = []

        while True:
            try:
                user_text = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_text:
                continue
            if user_text.lower() in exit_commands:
                break

            if use_tools:
                answer = await self.run_with_tools(
                    user_text, history=history, max_rounds=max_tool_rounds,
                )
            else:
                result = await self.run_once(user_input=user_text, history=history)
                answer = result.output_text

            history.append(ContextNode(role=MessageRole.USER, content=user_text))
            history.append(ContextNode(role=MessageRole.ASSISTANT, content=answer))

            if on_response is not None:
                on_response(answer)
            else:
                print(f"\nAssistant:\n{answer}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  ChatSession — lightweight conversation wrapper
# ═══════════════════════════════════════════════════════════════════════════

class ChatSession:
    """Manages conversation history for an :class:`Agent`.

    Wraps ``run_once`` / ``run_with_tools`` and automatically maintains the
    ``history`` list so callers don't have to.

    Usage::

        session = ChatSession(agent)
        answer = await session.send("Hello!")
        answer2 = await session.send("Tell me more.")
        print(session.history)
    """

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.history: list[ContextNode] = []

    async def send(self, user_input: str) -> str:
        """Send a message and get the assistant reply.  History is updated
        automatically."""
        result = await self.agent.run_once(
            user_input=user_input,
            history=self.history,
        )
        self.history.append(ContextNode(role=MessageRole.USER, content=user_input))
        self.history.append(ContextNode(role=MessageRole.ASSISTANT, content=result.output_text))
        return result.output_text

    async def send_with_tools(
        self,
        user_input: str,
        *,
        max_rounds: int = 6,
        on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_result: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> str:
        """Send a message using the agentic tool loop.  History is updated
        automatically."""
        answer = await self.agent.run_with_tools(
            user_input,
            history=self.history,
            max_rounds=max_rounds,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
        )
        self.history.append(ContextNode(role=MessageRole.USER, content=user_input))
        self.history.append(ContextNode(role=MessageRole.ASSISTANT, content=answer))
        return answer

    def clear(self) -> None:
        """Reset conversation history."""
        self.history.clear()
