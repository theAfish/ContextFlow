from __future__ import annotations

import re
from typing import Any, Callable, Iterable

from contextflow.core.models import ContextNode, ContextStack, MessageRole
from contextflow.core.pruning import PruningStrategy

SlotProvider = Callable[[], str]


class Composer:
    """Builds a ContextStack with explicit, inspectable composition steps."""

    _slot_pattern = re.compile(r"{{\s*([a-zA-Z0-9_\-]+)\s*}}")

    def __init__(self, *, pruning_strategy: PruningStrategy | None = None) -> None:
        self._pruning_strategy = pruning_strategy

    def fill_slots(
        self,
        template: str,
        static_slots: dict[str, str] | None = None,
        dynamic_slots: dict[str, SlotProvider] | None = None,
    ) -> str:
        static_slots = static_slots or {}
        dynamic_slots = dynamic_slots or {}

        def _replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key in static_slots:
                return static_slots[key]
            if key in dynamic_slots:
                return dynamic_slots[key]()
            return match.group(0)

        return self._slot_pattern.sub(_replace, template)

    def compose(
        self,
        *,
        system_prompt: str,
        history: Iterable[ContextNode] | None = None,
        rag_snippets: Iterable[str] | None = None,
        injected_nodes: Iterable[ContextNode] | None = None,
        slots: dict[str, str] | None = None,
        dynamic_slots: dict[str, SlotProvider] | None = None,
    ) -> ContextStack:
        stack = ContextStack()

        rendered_system = self.fill_slots(
            system_prompt,
            static_slots=slots,
            dynamic_slots=dynamic_slots,
        )
        stack.add(ContextNode(role=MessageRole.SYSTEM, content=rendered_system, priority=-100))

        if rag_snippets:
            rag_content = "\n\n".join(f"- {snippet}" for snippet in rag_snippets)
            stack.add(
                ContextNode(
                    role=MessageRole.SYSTEM,
                    content=f"Retrieved context:\n{rag_content}",
                    metadata={"source": "rag"},
                    priority=-50,
                )
            )

        if history:
            stack.extend(list(history))

        if injected_nodes:
            stack.extend(list(injected_nodes))

        return stack

    def render(self, stack: ContextStack, max_tokens: int | None = None) -> list[dict[str, Any]]:
        active_stack = stack
        if max_tokens is not None and self._pruning_strategy is not None:
            active_stack = self._pruning_strategy.prune(stack, max_tokens=max_tokens)
        return active_stack.render_messages()
