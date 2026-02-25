from __future__ import annotations

from abc import ABC, abstractmethod

from contextflow.core.models import ContextNode, ContextStack, MessageRole


class PruningStrategy(ABC):
    @abstractmethod
    def prune(self, stack: ContextStack, max_tokens: int) -> ContextStack:
        raise NotImplementedError


class KeepSystemOnlyStrategy(PruningStrategy):
    def prune(self, stack: ContextStack, max_tokens: int) -> ContextStack:
        system_nodes = [node for node in stack.nodes if node.role == MessageRole.SYSTEM]
        return ContextStack(nodes=system_nodes)


class DropMiddleStrategy(PruningStrategy):
    def prune(self, stack: ContextStack, max_tokens: int) -> ContextStack:
        if stack.estimate_tokens() <= max_tokens:
            return stack

        nodes: list[ContextNode] = list(stack.nodes)
        while len(nodes) > 2 and ContextStack(nodes=nodes).estimate_tokens() > max_tokens:
            middle_index = len(nodes) // 2
            nodes.pop(middle_index)

        return ContextStack(nodes=nodes)
