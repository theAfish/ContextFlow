"""Per-agent finite state machine with transitions, guards, and lifecycle hooks.

Example usage::

    sm = AgentStateMachine(
        initial="idle",
        transitions={
            "idle":    ["running"],
            "running": ["paused", "done", "error"],
            "paused":  ["running", "done"],
            "error":   ["idle"],
        },
    )
    sm.allow_run_when("running")          # agent.run_once only works in these

    @sm.on_enter("error")
    def _on_error(old_state, new_state, context):
        print(f"Entered error from {old_state}")

    @sm.guard("running", "done")
    def _must_be_complete(current_state, target_state, context):
        return context.get("task_complete", False)
"""

from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StateError(Exception):
    """Raised for invalid state transitions or state-gated operations."""


class TransitionBlockedByGuard(StateError):
    """A guard callback returned ``False``, blocking the transition."""


class InvalidTransition(StateError):
    """The requested transition is not declared in the transition table."""


class RunBlockedByState(StateError):
    """``Agent.run_once`` was called while the agent is in a non-runnable state."""


# ---------------------------------------------------------------------------
# History entry
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class StateTransition:
    """Immutable record kept in the history log."""
    from_state: str
    to_state: str
    timestamp: datetime
    context: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HookFn = Callable[..., Any]  # sync or async


async def _call_maybe_async(fn: HookFn, *args: Any, **kwargs: Any) -> Any:
    """Invoke *fn* regardless of whether it is sync or async."""
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


# ---------------------------------------------------------------------------
# AgentStateMachine
# ---------------------------------------------------------------------------

class AgentStateMachine:
    """Finite state machine attached to a single :class:`Agent`.

    Parameters
    ----------
    initial:
        The state the machine starts in (a plain string).
    transitions:
        ``{from_state: [allowed_to_states, ...]}``.  If *None*, **any**
        transition is allowed (open mode).
    """

    def __init__(
        self,
        *,
        initial: str,
        transitions: dict[str, list[str]] | None = None,
    ) -> None:
        self._current: str = initial
        self._transitions: dict[str, list[str]] | None = transitions

        # Lifecycle hooks: {state: [callable, ...]}
        self._on_enter: dict[str, list[HookFn]] = defaultdict(list)
        self._on_exit: dict[str, list[HookFn]] = defaultdict(list)

        # Transition guards: {(from, to): [callable, ...]}
        self._guards: dict[tuple[str, str], list[HookFn]] = defaultdict(list)

        # Global hooks (fire on every transition)
        self._on_change: list[HookFn] = []

        # Run-gate: if non-empty, run_once only works when current ∈ this set
        self._run_states: set[str] | None = None

        # Audit log
        self._history: list[StateTransition] = []

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def current(self) -> str:
        """The current state label."""
        return self._current

    @property
    def history(self) -> list[StateTransition]:
        """Full transition history (read-only copy)."""
        return list(self._history)

    @property
    def all_states(self) -> set[str]:
        """All states mentioned in the transition table (plus the initial)."""
        states: set[str] = {self._current}
        if self._transitions:
            for src, dsts in self._transitions.items():
                states.add(src)
                states.update(dsts)
        return states

    # ------------------------------------------------------------------
    # Transition checks
    # ------------------------------------------------------------------

    def can_transition(self, to_state: str) -> bool:
        """Return *True* if *to_state* is reachable from the current state
        (ignoring guards)."""
        if self._transitions is None:
            return True
        allowed = self._transitions.get(self._current, [])
        return to_state in allowed

    def can_run(self) -> bool:
        """Return *True* if ``Agent.run_once`` is allowed in the current state."""
        if self._run_states is None:
            return True
        return self._current in self._run_states

    # ------------------------------------------------------------------
    # Transition execution
    # ------------------------------------------------------------------

    async def transition_to(
        self, new_state: str, *, context: dict[str, Any] | None = None
    ) -> str:
        """Transition to *new_state*, running guards and hooks.

        Returns the new current state.

        Raises
        ------
        InvalidTransition
            If the transition table forbids the move.
        TransitionBlockedByGuard
            If any guard returns a falsy value.
        """
        ctx = context or {}
        old_state = self._current

        # 1. Check transition table
        if not self.can_transition(new_state):
            raise InvalidTransition(
                f"Transition {old_state!r} -> {new_state!r} is not allowed. "
                f"Allowed targets: {self._transitions.get(old_state, [])!r}"
            )

        # 2. Run guards
        for guard_fn in self._guards.get((old_state, new_state), []):
            ok = await _call_maybe_async(guard_fn, old_state, new_state, ctx)
            if not ok:
                raise TransitionBlockedByGuard(
                    f"Guard {guard_fn.__name__!r} blocked {old_state!r} -> {new_state!r}"
                )

        # 3. on_exit hooks for the old state
        for hook in self._on_exit.get(old_state, []):
            await _call_maybe_async(hook, old_state, new_state, ctx)

        # 4. Actually switch
        self._current = new_state

        # 5. Record history
        self._history.append(
            StateTransition(
                from_state=old_state,
                to_state=new_state,
                timestamp=datetime.now(timezone.utc),
                context=ctx,
            )
        )

        # 6. on_enter hooks for the new state
        for hook in self._on_enter.get(new_state, []):
            await _call_maybe_async(hook, old_state, new_state, ctx)

        # 7. Global on_change hooks
        for hook in self._on_change:
            await _call_maybe_async(hook, old_state, new_state, ctx)

        return self._current

    async def force_transition(
        self, new_state: str, *, context: dict[str, Any] | None = None
    ) -> str:
        """Move to *new_state* **skipping** the transition table and guards.

        on_exit / on_enter / on_change hooks still fire.
        """
        ctx = context or {}
        old_state = self._current

        for hook in self._on_exit.get(old_state, []):
            await _call_maybe_async(hook, old_state, new_state, ctx)

        self._current = new_state

        self._history.append(
            StateTransition(
                from_state=old_state,
                to_state=new_state,
                timestamp=datetime.now(timezone.utc),
                context=ctx,
            )
        )

        for hook in self._on_enter.get(new_state, []):
            await _call_maybe_async(hook, old_state, new_state, ctx)

        for hook in self._on_change:
            await _call_maybe_async(hook, old_state, new_state, ctx)

        return self._current

    # Synchronous convenience wrappers -----------------------------------

    def transition_to_sync(
        self, new_state: str, *, context: dict[str, Any] | None = None
    ) -> str:
        """Blocking wrapper around :meth:`transition_to` for non-async code."""
        return asyncio.get_event_loop().run_until_complete(
            self.transition_to(new_state, context=context)
        )

    def force_transition_sync(
        self, new_state: str, *, context: dict[str, Any] | None = None
    ) -> str:
        """Blocking wrapper around :meth:`force_transition`."""
        return asyncio.get_event_loop().run_until_complete(
            self.force_transition(new_state, context=context)
        )

    # ------------------------------------------------------------------
    # Decorator API for hooks & guards
    # ------------------------------------------------------------------

    def on_enter(self, state: str) -> Callable[[HookFn], HookFn]:
        """Decorator – register a hook that fires when entering *state*.

        The hook receives ``(old_state, new_state, context)`` and may be
        sync or async.
        """
        def decorator(fn: HookFn) -> HookFn:
            self._on_enter[state].append(fn)
            return fn
        return decorator

    def on_exit(self, state: str) -> Callable[[HookFn], HookFn]:
        """Decorator – register a hook that fires when leaving *state*."""
        def decorator(fn: HookFn) -> HookFn:
            self._on_exit[state].append(fn)
            return fn
        return decorator

    def on_change(self, fn: HookFn) -> HookFn:
        """Decorator – register a hook that fires on **every** transition.

        The hook receives ``(old_state, new_state, context)``.
        """
        self._on_change.append(fn)
        return fn

    def guard(self, from_state: str, to_state: str) -> Callable[[HookFn], HookFn]:
        """Decorator – register a guard for a specific transition.

        The guard receives ``(from_state, to_state, context)`` and must
        return a truthy value to allow the transition.
        """
        def decorator(fn: HookFn) -> HookFn:
            self._guards[(from_state, to_state)].append(fn)
            return fn
        return decorator

    # ------------------------------------------------------------------
    # Run-gate configuration
    # ------------------------------------------------------------------

    def allow_run_when(self, *states: str) -> None:
        """Restrict ``Agent.run_once`` to the listed states only.

        Calling this multiple times **replaces** the previous set.
        Pass no arguments to remove the restriction.
        """
        if not states:
            self._run_states = None
        else:
            self._run_states = set(states)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self, state: str | None = None) -> None:
        """Reset the machine.  Optionally move to *state* (defaults to the
        first recorded state in history, or current if no history)."""
        if state is not None:
            self._current = state
        elif self._history:
            self._current = self._history[0].from_state
        self._history.clear()

    def __repr__(self) -> str:
        return (
            f"AgentStateMachine(current={self._current!r}, "
            f"states={sorted(self.all_states)}, "
            f"history_len={len(self._history)})"
        )
