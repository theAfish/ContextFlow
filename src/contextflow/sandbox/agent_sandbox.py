from __future__ import annotations

import importlib
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from contextflow.agents.agent import Agent, ToolFunc
from contextflow.exceptions import SandboxError


@dataclass(slots=True)
class SandboxExecutionResult:
    stdout: str
    stderr: str
    exit_code: int

    @property
    def returncode(self) -> int:
        return self.exit_code


@dataclass(slots=True)
class SandboxFileEntry:
    name: str
    kind: str
    size: int | None = None


class AgentSandbox:
    """Sandbox for isolated file/code operations with optional k8s backend."""

    def __init__(
        self,
        *,
        workspace: Path | None,
        mode: str,
        client: Any | None = None,
        python_executable: str = "python",
    ) -> None:
        self.workspace = workspace.resolve() if workspace else None
        self.mode = mode
        self._client = client
        self._python_executable = python_executable
        self._entered = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, workspace: str | Path, *, python_executable: str = "python") -> AgentSandbox:
        root = Path(workspace).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return cls(workspace=root, mode="local", python_executable=python_executable)

    @classmethod
    def from_agent_sandbox(
        cls,
        *,
        template_name: str,
        namespace: str = "default",
        gateway_name: str | None = None,
        api_url: str | None = None,
        server_port: int = 8888,
        enable_tracing: bool = False,
    ) -> AgentSandbox:
        try:
            sandbox_module = importlib.import_module("k8s_agent_sandbox")
        except ImportError as exc:
            raise ImportError(
                "k8s-agent-sandbox is not installed. Install with: pip install k8s-agent-sandbox"
            ) from exc

        SandboxClient = getattr(sandbox_module, "SandboxClient")

        client = SandboxClient(
            template_name=template_name,
            namespace=namespace,
            gateway_name=gateway_name,
            api_url=api_url,
            server_port=server_port,
            enable_tracing=enable_tracing,
        )
        return cls(workspace=None, mode="k8s", client=client)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> AgentSandbox:
        if self.mode == "k8s" and self._client and not self._entered:
            self._client.__enter__()
            self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.mode == "k8s" and self._client and self._entered:
            self._client.__exit__(None, None, None)
            self._entered = False

    # ------------------------------------------------------------------
    # Internal guards – eliminates the ``if mode == "k8s": if not client``
    # boilerplate that was previously repeated in every public method.
    # ------------------------------------------------------------------

    @property
    def _is_k8s(self) -> bool:
        return self.mode == "k8s"

    def _require_k8s_client(self) -> Any:
        """Return the k8s client or raise ``SandboxError``."""
        if self._client is None:
            raise SandboxError("Sandbox client is not initialized.")
        return self._client

    def _require_local_workspace(self) -> Path:
        """Return the local workspace path or raise ``SandboxError``."""
        if self.workspace is None:
            raise SandboxError("Local workspace is not configured.")
        return self.workspace

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self, command: str, *, timeout: int = 60, stdin: str | None = None
    ) -> SandboxExecutionResult:
        if self._is_k8s:
            client = self._require_k8s_client()
            if stdin is not None:
                raise ValueError("stdin is not supported for k8s sandbox run().")
            result = client.run(command, timeout=timeout)
            return SandboxExecutionResult(
                stdout=result.stdout, stderr=result.stderr, exit_code=result.exit_code,
            )

        workspace = self._require_local_workspace()
        process = subprocess.run(
            command,
            cwd=workspace,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            input=stdin,
        )
        return SandboxExecutionResult(
            stdout=process.stdout, stderr=process.stderr, exit_code=process.returncode,
        )

    def run_python_code(self, code: str, *, timeout: int = 60) -> SandboxExecutionResult:
        if self._is_k8s:
            escaped = shlex.quote(code)
            return self.run(f"{self._python_executable} -c {escaped}", timeout=timeout)

        workspace = self._require_local_workspace()
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".py", dir=workspace, delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(code)

        try:
            process = subprocess.run(
                [self._python_executable, "-I", str(temp_path.name)],
                cwd=workspace,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            return SandboxExecutionResult(
                stdout=process.stdout, stderr=process.stderr, exit_code=process.returncode,
            )
        finally:
            temp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def write_text(self, path: str | Path, content: str, *, encoding: str = "utf-8") -> None:
        if self._is_k8s:
            self._require_k8s_client().write(str(path), content)
            return
        target = self._resolve_local_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=encoding)

    def read_text(self, path: str | Path, *, encoding: str = "utf-8") -> str:
        if self._is_k8s:
            return self._require_k8s_client().read(str(path)).decode(encoding)
        return self._resolve_local_path(path).read_text(encoding=encoding)

    def write_bytes(self, path: str | Path, content: bytes) -> None:
        if self._is_k8s:
            self._require_k8s_client().write(str(path), content)
            return
        target = self._resolve_local_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def read_bytes(self, path: str | Path) -> bytes:
        if self._is_k8s:
            return self._require_k8s_client().read(str(path))
        return self._resolve_local_path(path).read_bytes()

    def exists(self, path: str | Path) -> bool:
        if self._is_k8s:
            return self._require_k8s_client().exists(str(path))
        return self._resolve_local_path(path).exists()

    def list_files(self, path: str | Path = ".") -> list[SandboxFileEntry]:
        if self._is_k8s:
            files = self._require_k8s_client().list(str(path))
            return [
                SandboxFileEntry(name=file.name, kind=file.type, size=file.size)
                for file in files
            ]

        base = self._resolve_local_path(path)
        if base.is_file():
            return [SandboxFileEntry(name=str(Path(path)), kind="file", size=base.stat().st_size)]

        entries: list[SandboxFileEntry] = []
        for child in sorted(base.rglob("*")):
            if child.is_dir():
                kind, size = "dir", None
            else:
                kind, size = "file", child.stat().st_size
            entries.append(
                SandboxFileEntry(
                    name=str(child.relative_to(base if base.is_dir() else self.workspace)),
                    kind=kind,
                    size=size,
                )
            )
        return entries

    # ------------------------------------------------------------------
    # Agent attachment
    # ------------------------------------------------------------------

    def attach_agent(self, agent: Agent, *, mutate: bool = False) -> Agent:
        sandbox_tools = self._build_default_tools()
        if mutate:
            existing_names = {tool.__name__ for tool in agent.tools}
            for tool in sandbox_tools:
                if tool.__name__ not in existing_names:
                    agent.tools.append(tool)
            return agent

        combined_tools: list[ToolFunc] = list(agent.tools)
        existing_names = {tool.__name__ for tool in combined_tools}
        for tool in sandbox_tools:
            if tool.__name__ not in existing_names:
                combined_tools.append(tool)

        return Agent(
            model=agent.model,
            name=agent.name,
            description=agent.description,
            instruction=agent.instruction,
            backend=agent.backend,
            base_url=agent.base_url,
            api_key=agent.api_key,
            enable_thinking=agent.enable_thinking,
            temperature=agent.temperature,
            tools=combined_tools,
            composer=agent.composer,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_local_path(self, path: str | Path) -> Path:
        workspace = self._require_local_workspace()
        candidate = (workspace / Path(path)).resolve()
        if not str(candidate).startswith(str(workspace)):
            raise SandboxError("Path escapes sandbox workspace.")
        return candidate

    def _build_default_tools(self) -> list[Callable[..., Any]]:
        sandbox = self

        def sandbox_run(
            command: str,
            stdin: str | None = None,
            timeout: int = 60,
        ) -> dict[str, Any]:
            """Execute a shell command inside the sandbox. Use stdin for interactive commands."""
            result = sandbox.run(command, timeout=timeout, stdin=stdin)
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
            }

        def sandbox_run_python(code: str) -> dict[str, Any]:
            """Execute Python code inside the sandbox."""
            result = sandbox.run_python_code(code)
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
            }

        def sandbox_write_text(path: str, content: str) -> dict[str, Any]:
            """Write text content to a file in the sandbox."""
            sandbox.write_text(path, content)
            return {"ok": True, "path": path}

        def sandbox_read_text(path: str) -> dict[str, Any]:
            """Read text content from a file in the sandbox."""
            return {"path": path, "content": sandbox.read_text(path)}

        def sandbox_exists(path: str) -> dict[str, Any]:
            """Check whether a path exists in the sandbox."""
            return {"path": path, "exists": sandbox.exists(path)}

        def sandbox_list(path: str = ".") -> dict[str, Any]:
            """List files and directories in the sandbox."""
            entries = sandbox.list_files(path)
            return {
                "path": path,
                "entries": [
                    {"name": entry.name, "type": entry.kind, "size": entry.size}
                    for entry in entries
                ],
            }

        return [
            sandbox_run,
            sandbox_run_python,
            sandbox_write_text,
            sandbox_read_text,
            sandbox_exists,
            sandbox_list,
        ]
