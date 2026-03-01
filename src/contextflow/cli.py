"""ContextFlow CLI – ``contextflow web agent_script.py``."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Callable


def _load_module_from_path(script: str) -> Any:
    """Import a .py file as a module and return it."""
    path = Path(script).resolve()
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)
    if not path.suffix == ".py":
        print(f"Error: expected a .py file, got: {path}")
        sys.exit(1)

    # Add the script's directory to sys.path so relative imports work
    script_dir = str(path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    spec = importlib.util.spec_from_file_location("__user_agent__", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_attr_path(root: Any, path: str) -> Any:
    """Resolve dotted attribute path from *root* (e.g., ``foo.bar.agent``)."""
    obj = root
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _find_all_agents(mod: Any) -> list[tuple[str, Any]]:
    """Return all top-level Agent instances in declaration order."""
    from contextflow.agents.agent import Agent

    found: list[tuple[str, Agent]] = []
    for name, obj in getattr(mod, "__dict__", {}).items():
        if name.startswith("_"):
            continue
        if isinstance(obj, Agent):
            found.append((name, obj))
    return found


def _find_agent(mod: Any, *, agent_name: str | None = None) -> Any:
    """Discover an Agent instance in the loaded module.

    Search order:
    1. Explicit ``agent_name`` (if provided).
    2. A variable named ``root_agent``.
    3. A variable literally named ``agent``.
    4. A variable named ``orchestrator``.
    5. The first top-level Agent encountered in module declaration order.
    """
    from contextflow.agents.agent import Agent

    if agent_name:
        obj = _resolve_attr_path(mod, agent_name)
        if isinstance(obj, Agent):
            return obj
        return None

    if hasattr(mod, "root_agent") and isinstance(mod.root_agent, Agent):
        return mod.root_agent

    if hasattr(mod, "agent") and isinstance(mod.agent, Agent):
        return mod.agent

    if hasattr(mod, "orchestrator") and isinstance(mod.orchestrator, Agent):
        return mod.orchestrator

    found = _find_all_agents(mod)
    if found:
        return found[0][1]

    return None


def _find_debug_session(mod: Any) -> Any:
    """Check if the user already created a DebugSession."""
    from contextflow.debug.session import DebugSession

    if hasattr(mod, "session") and isinstance(mod.session, DebugSession):
        return mod.session

    for name in dir(mod):
        obj = getattr(mod, name, None)
        if isinstance(obj, DebugSession):
            return obj

    return None


def _find_debug_chat_handler(mod: Any, root_agent: Any) -> Callable[[str], Any] | None:
    """Discover a script-level chat handler for orchestrated flows.

    Preferred conventions:
    - ``debug_chat`` / ``debug_chat_handler`` callable(message)
    - ``run_orchestrator`` callable(message, history?)
    """
    explicit = getattr(mod, "debug_chat", None) or getattr(mod, "debug_chat_handler", None)
    if callable(explicit):
        return explicit

    run_orchestrator = getattr(mod, "run_orchestrator", None)
    if not callable(run_orchestrator):
        return None

    from contextflow.agents.agent import ChatSession
    from contextflow.core.models import ContextNode, MessageRole

    orchestrator_session = ChatSession(root_agent)

    async def _handler(user_message: str) -> str:
        try:
            maybe_answer = run_orchestrator(user_message, orchestrator_session.history)
        except TypeError:
            maybe_answer = run_orchestrator(user_message)

        answer = await maybe_answer if inspect.isawaitable(maybe_answer) else maybe_answer
        answer_text = str(answer)

        orchestrator_session.history.append(ContextNode(role=MessageRole.USER, content=user_message))
        orchestrator_session.history.append(ContextNode(role=MessageRole.ASSISTANT, content=answer_text))
        return answer_text

    return _handler


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_web(args: argparse.Namespace) -> None:
    """Launch the debug frontend for an agent script."""
    from contextflow.debug.session import DebugSession
    from contextflow.debug.server import launch_debug

    mod = _load_module_from_path(args.script)

    agent_name = args.agent
    if args.agent_path and args.agent and args.agent_path != args.agent:
        print("Error: please use only one of positional <agent_path> or --agent.")
        sys.exit(1)
    if not agent_name:
        agent_name = args.agent_path

    # Try to find an existing DebugSession first
    session = None if agent_name else _find_debug_session(mod)

    if session is None:
        agent = _find_agent(mod, agent_name=agent_name)
        if agent is None:
            details = (
                f" with name/path '{agent_name}'" if agent_name else ""
            )
            print(
                f"Error: could not find an Agent instance in the script{details}.\n"
                "Make sure your script defines a module-level variable "
                "named `root_agent` (an instance of contextflow.Agent), or "
                "pass --agent <variable_name>."
            )
            sys.exit(1)

        all_agents = [obj for _, obj in _find_all_agents(mod)]
        chat_handler = _find_debug_chat_handler(mod, agent)
        session = DebugSession(agent, agents=all_agents, chat_handler=chat_handler)

    launch_debug(
        session,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="contextflow",
        description="ContextFlow CLI – run and debug LLM agents.",
    )
    sub = parser.add_subparsers(dest="command")

    # ── contextflow web ───────────────────────────────────────────────
    web_parser = sub.add_parser(
        "web",
        help="Launch the debug frontend for an agent script.",
        description=(
            "Load a Python script that defines an Agent, "
            "wrap it in a DebugSession, and start the web-based "
            "debug frontend."
        ),
    )
    web_parser.add_argument(
        "script",
        help="Path to a .py file that defines a `root_agent` variable (a contextflow.Agent).",
    )
    web_parser.add_argument(
        "agent_path",
        nargs="?",
        help="Optional agent variable path (e.g. `orchestrator` or `container.main_agent`).",
    )
    web_parser.add_argument(
        "--agent", "-a",
        help="Explicit agent variable path to debug (overrides auto-discovery).",
    )
    web_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8790,
        help="Port for the debug server (default: 8790).",
    )
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1).",
    )
    web_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open a browser tab.",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "web":
        cmd_web(args)


if __name__ == "__main__":
    main()
