from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class ContextNode:
    role: MessageRole
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    token_estimate: int | None = None
    node_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role.value, "content": self.content}

        if self.metadata:
            if "name" in self.metadata:
                message["name"] = self.metadata["name"]
            if "tool_call_id" in self.metadata:
                message["tool_call_id"] = self.metadata["tool_call_id"]

        return message


@dataclass(slots=True)
class ContextStack:
    nodes: list[ContextNode] = field(default_factory=list)

    def add(self, node: ContextNode) -> None:
        self.nodes.append(node)

    def extend(self, nodes: list[ContextNode]) -> None:
        self.nodes.extend(nodes)

    def render_messages(self) -> list[dict[str, Any]]:
        ordered = sorted(self.nodes, key=lambda node: (node.priority, node.created_at))
        return [node.to_message() for node in ordered]

    def estimate_tokens(self) -> int:
        return sum(node.token_estimate or max(1, len(node.content) // 4) for node in self.nodes)
