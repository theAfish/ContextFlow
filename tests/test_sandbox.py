"""Tests for contextflow.sandbox — AgentSandbox (local mode)."""

from __future__ import annotations

import platform
import pytest
import tempfile
from pathlib import Path

from contextflow.sandbox.agent_sandbox import (
    AgentSandbox,
    SandboxExecutionResult,
    SandboxFileEntry,
)
from contextflow.agents.agent import Agent
from contextflow.exceptions import SandboxError


# ═══════════════════════════════════════════════════════════════════════════
#  SandboxExecutionResult
# ═══════════════════════════════════════════════════════════════════════════


class TestSandboxExecutionResult:
    def test_fields(self):
        r = SandboxExecutionResult(stdout="hi", stderr="", exit_code=0)
        assert r.stdout == "hi"
        assert r.stderr == ""
        assert r.exit_code == 0

    def test_returncode_alias(self):
        r = SandboxExecutionResult(stdout="", stderr="", exit_code=42)
        assert r.returncode == 42


# ═══════════════════════════════════════════════════════════════════════════
#  SandboxFileEntry
# ═══════════════════════════════════════════════════════════════════════════


class TestSandboxFileEntry:
    def test_fields(self):
        e = SandboxFileEntry(name="test.py", kind="file", size=100)
        assert e.name == "test.py"
        assert e.kind == "file"
        assert e.size == 100

    def test_optional_size(self):
        e = SandboxFileEntry(name="dir", kind="dir")
        assert e.size is None


# ═══════════════════════════════════════════════════════════════════════════
#  AgentSandbox — local mode
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentSandboxLocal:
    @pytest.fixture
    def sandbox(self, tmp_path):
        return AgentSandbox.create(tmp_path)

    def test_create(self, tmp_path):
        sb = AgentSandbox.create(tmp_path)
        assert sb.mode == "local"
        assert sb.workspace is not None

    def test_write_and_read_text(self, sandbox):
        sandbox.write_text("hello.txt", "world")
        content = sandbox.read_text("hello.txt")
        assert content == "world"

    def test_write_and_read_bytes(self, sandbox):
        sandbox.write_bytes("data.bin", b"\x00\x01\x02")
        content = sandbox.read_bytes("data.bin")
        assert content == b"\x00\x01\x02"

    def test_exists(self, sandbox):
        assert sandbox.exists("nope.txt") is False
        sandbox.write_text("yep.txt", "content")
        assert sandbox.exists("yep.txt") is True

    def test_list_files(self, sandbox):
        sandbox.write_text("a.txt", "aaa")
        sandbox.write_text("b.txt", "bbb")
        entries = sandbox.list_files(".")
        names = {e.name for e in entries}
        assert "a.txt" in names
        assert "b.txt" in names

    def test_run_command(self, sandbox):
        if platform.system() == "Windows":
            result = sandbox.run("echo hello")
        else:
            result = sandbox.run("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.stdout

    def test_run_python_code(self, sandbox):
        result = sandbox.run_python_code("print('hi from sandbox')")
        assert result.exit_code == 0
        assert "hi from sandbox" in result.stdout

    def test_path_escape_raises(self, sandbox):
        with pytest.raises(SandboxError, match="escapes sandbox"):
            sandbox.read_text("../../etc/passwd")

    def test_write_creates_subdirectories(self, sandbox):
        sandbox.write_text("sub/dir/file.txt", "nested")
        assert sandbox.read_text("sub/dir/file.txt") == "nested"

    def test_context_manager(self, tmp_path):
        with AgentSandbox.create(tmp_path) as sb:
            sb.write_text("test.txt", "ctx")
            assert sb.read_text("test.txt") == "ctx"


# ═══════════════════════════════════════════════════════════════════════════
#  AgentSandbox — attach_agent
# ═══════════════════════════════════════════════════════════════════════════


class TestAttachAgent:
    @pytest.fixture
    def sandbox(self, tmp_path):
        return AgentSandbox.create(tmp_path)

    def test_attach_creates_new_agent_with_tools(self, sandbox):
        agent = Agent(
            model="test", name="A", description="d", instruction="i",
        )
        new_agent = sandbox.attach_agent(agent)
        assert new_agent is not agent
        tool_names = {t.__name__ for t in new_agent.tools}
        assert "sandbox_run" in tool_names
        assert "sandbox_run_python" in tool_names
        assert "sandbox_write_text" in tool_names
        assert "sandbox_read_text" in tool_names
        assert "sandbox_exists" in tool_names
        assert "sandbox_list" in tool_names

    def test_attach_mutate(self, sandbox):
        agent = Agent(
            model="test", name="A", description="d", instruction="i",
        )
        same_agent = sandbox.attach_agent(agent, mutate=True)
        assert same_agent is agent
        assert len(agent.tools) == 6

    def test_attach_does_not_duplicate_tools(self, sandbox):
        def sandbox_run(command: str) -> dict:
            """Custom sandbox_run."""
            return {}

        agent = Agent(
            model="test", name="A", description="d", instruction="i",
            tools=[sandbox_run],
        )
        new_agent = sandbox.attach_agent(agent)
        names = [t.__name__ for t in new_agent.tools]
        assert names.count("sandbox_run") == 1


# ═══════════════════════════════════════════════════════════════════════════
#  AgentSandbox — k8s mode guards
# ═══════════════════════════════════════════════════════════════════════════


class TestSandboxK8sGuards:
    def test_k8s_no_client_raises_on_run(self):
        sb = AgentSandbox(workspace=None, mode="k8s", client=None)
        with pytest.raises(SandboxError, match="not initialized"):
            sb.run("echo hi")

    def test_local_no_workspace_raises(self):
        sb = AgentSandbox(workspace=None, mode="local", client=None)
        with pytest.raises(SandboxError, match="not configured"):
            sb.run("echo hi")
