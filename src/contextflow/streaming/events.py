from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class StreamEvent:
    kind: str
    text: str = ""
    metadata: dict[str, Any] | None = None


def extract_content(chunk: Any) -> str:
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        return delta.get("content") or ""

    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return ""
    return getattr(delta, "content", "") or ""


def extract_reasoning(chunk: Any) -> str:
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        return delta.get("reasoning_content") or ""

    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return ""
    return getattr(delta, "reasoning_content", "") or ""


def normalize_chunk(chunk: Any) -> list[StreamEvent]:
    events: list[StreamEvent] = []

    reasoning = extract_reasoning(chunk)
    if reasoning:
        events.append(StreamEvent(kind="reasoning", text=reasoning))

    content = extract_content(chunk)
    if content:
        events.append(StreamEvent(kind="content", text=content))

    if not events:
        events.append(StreamEvent(kind="meta", metadata={"chunk": chunk}))

    return events
