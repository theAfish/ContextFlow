"""ContextFlow CLI – ``contextflow web agent_script.py``."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any


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


def _find_agent(mod: Any) -> Any:
    """Discover an Agent instance in the loaded module.

    Search order:
    1. A variable literally named ``agent``.
    2. Any variable whose type name is ``Agent``.
    """
    from contextflow.agents.agent import Agent

    # 1. Explicit name
    if hasattr(mod, "agent") and isinstance(mod.agent, Agent):
        return mod.agent

    # 2. Scan all module-level attributes
    for name in dir(mod):
        obj = getattr(mod, name, None)
        if isinstance(obj, Agent):
            return obj

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


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_web(args: argparse.Namespace) -> None:
    """Launch the debug frontend for an agent script."""
    from contextflow.debug.session import DebugSession
    from contextflow.debug.server import launch_debug

    mod = _load_module_from_path(args.script)

    # Try to find an existing DebugSession first
    session = _find_debug_session(mod)

    if session is None:
        agent = _find_agent(mod)
        if agent is None:
            print(
                "Error: could not find an Agent instance in the script.\n"
                "Make sure your script defines a module-level variable "
                "named `agent` (an instance of contextflow.Agent)."
            )
            sys.exit(1)
        session = DebugSession(agent)

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
        help="Path to a .py file that defines an `agent` variable (a contextflow.Agent).",
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
