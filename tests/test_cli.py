"""Tests for contextflow.cli agent selection behavior."""

from __future__ import annotations

from types import SimpleNamespace

from contextflow.agents.agent import Agent
from contextflow.cli import _find_agent, _resolve_attr_path


def _mk_agent(name: str) -> Agent:
    return Agent(
        model="test-model",
        name=name,
        description="test",
        instruction="test",
    )


def test_resolve_attr_path_dotted_name():
    target = _mk_agent("target")
    mod = SimpleNamespace(container=SimpleNamespace(main=target))

    resolved = _resolve_attr_path(mod, "container.main")

    assert resolved is target


def test_find_agent_prefers_root_agent_by_default():
    root = _mk_agent("root")
    orchestrator = _mk_agent("orchestrator")
    specialist = _mk_agent("specialist")
    mod = SimpleNamespace(root_agent=root, orchestrator=orchestrator, knowledge_agent=specialist)

    found = _find_agent(mod)

    assert found is root


def test_find_agent_prefers_orchestrator_when_no_root_or_agent():
    orchestrator = _mk_agent("orchestrator")
    specialist = _mk_agent("specialist")
    mod = SimpleNamespace(orchestrator=orchestrator, knowledge_agent=specialist)

    found = _find_agent(mod)

    assert found is orchestrator


def test_find_agent_uses_explicit_agent_name():
    orchestrator = _mk_agent("orchestrator")
    specialist = _mk_agent("specialist")
    mod = SimpleNamespace(orchestrator=orchestrator, knowledge_agent=specialist)

    found = _find_agent(mod, agent_name="knowledge_agent")

    assert found is specialist


def test_find_agent_uses_explicit_dotted_agent_name():
    nested = _mk_agent("nested")
    orchestrator = _mk_agent("orchestrator")
    mod = SimpleNamespace(orchestrator=orchestrator, agents=SimpleNamespace(main=nested))

    found = _find_agent(mod, agent_name="agents.main")

    assert found is nested


def test_find_agent_defaults_to_first_declared_when_multiple():
    first = _mk_agent("first")
    second = _mk_agent("second")
    mod = SimpleNamespace(first_agent=first, second_agent=second)

    found = _find_agent(mod)

    assert found is first
