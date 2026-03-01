"""Tests for contextflow.agents.state_machine — AgentStateMachine."""

from __future__ import annotations

import pytest

from contextflow.agents.state_machine import (
    AgentStateMachine,
    InvalidTransition,
    RunBlockedByState,
    StateError,
    StateTransition,
    TransitionBlockedByGuard,
    _call_maybe_async,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Basic state machine
# ═══════════════════════════════════════════════════════════════════════════


class TestStateMachineBasics:
    def test_initial_state(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        assert sm.current == "idle"

    def test_all_states_includes_initial(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        assert "idle" in sm.all_states

    def test_all_states_from_transitions(self):
        sm = AgentStateMachine(
            initial="idle",
            transitions={"idle": ["running"], "running": ["done", "error"]},
        )
        assert sm.all_states == {"idle", "running", "done", "error"}

    def test_history_starts_empty(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        assert sm.history == []

    def test_repr(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        r = repr(sm)
        assert "idle" in r
        assert "AgentStateMachine" in r


# ═══════════════════════════════════════════════════════════════════════════
#  Transitions
# ═══════════════════════════════════════════════════════════════════════════


class TestTransitions:
    @pytest.mark.asyncio
    async def test_valid_transition(self):
        sm = AgentStateMachine(
            initial="idle",
            transitions={"idle": ["running"], "running": ["done"]},
        )
        await sm.transition_to("running")
        assert sm.current == "running"

    @pytest.mark.asyncio
    async def test_invalid_transition_raises(self):
        sm = AgentStateMachine(
            initial="idle",
            transitions={"idle": ["running"]},
        )
        with pytest.raises(InvalidTransition, match="not allowed"):
            await sm.transition_to("done")

    @pytest.mark.asyncio
    async def test_open_mode_allows_any_transition(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        await sm.transition_to("anything")
        assert sm.current == "anything"
        await sm.transition_to("whatever")
        assert sm.current == "whatever"

    def test_can_transition(self):
        sm = AgentStateMachine(
            initial="idle",
            transitions={"idle": ["running"], "running": ["done"]},
        )
        assert sm.can_transition("running") is True
        assert sm.can_transition("done") is False

    def test_can_transition_open_mode(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        assert sm.can_transition("anything") is True

    @pytest.mark.asyncio
    async def test_transition_records_history(self):
        sm = AgentStateMachine(
            initial="idle",
            transitions={"idle": ["running"], "running": ["done"]},
        )
        await sm.transition_to("running")
        await sm.transition_to("done")
        history = sm.history
        assert len(history) == 2
        assert history[0].from_state == "idle"
        assert history[0].to_state == "running"
        assert history[1].from_state == "running"
        assert history[1].to_state == "done"

    @pytest.mark.asyncio
    async def test_transition_with_context(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        await sm.transition_to("running", context={"reason": "started"})
        h = sm.history
        assert h[0].context == {"reason": "started"}


# ═══════════════════════════════════════════════════════════════════════════
#  Force transition
# ═══════════════════════════════════════════════════════════════════════════


class TestForceTransition:
    @pytest.mark.asyncio
    async def test_force_ignores_transition_table(self):
        sm = AgentStateMachine(
            initial="idle",
            transitions={"idle": ["running"]},
        )
        # "error" is not in allowed transitions from "idle"
        await sm.force_transition("error")
        assert sm.current == "error"

    @pytest.mark.asyncio
    async def test_force_still_records_history(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        await sm.force_transition("forced_state")
        assert len(sm.history) == 1
        assert sm.history[0].to_state == "forced_state"


# ═══════════════════════════════════════════════════════════════════════════
#  Guards
# ═══════════════════════════════════════════════════════════════════════════


class TestGuards:
    @pytest.mark.asyncio
    async def test_guard_allows_transition(self):
        sm = AgentStateMachine(
            initial="idle",
            transitions={"idle": ["running"]},
        )

        @sm.guard("idle", "running")
        def always_allow(from_s, to_s, ctx):
            return True

        await sm.transition_to("running")
        assert sm.current == "running"

    @pytest.mark.asyncio
    async def test_guard_blocks_transition(self):
        sm = AgentStateMachine(
            initial="idle",
            transitions={"idle": ["running"]},
        )

        @sm.guard("idle", "running")
        def always_block(from_s, to_s, ctx):
            return False

        with pytest.raises(TransitionBlockedByGuard):
            await sm.transition_to("running")
        assert sm.current == "idle"

    @pytest.mark.asyncio
    async def test_async_guard(self):
        sm = AgentStateMachine(
            initial="idle",
            transitions={"idle": ["running"]},
        )

        @sm.guard("idle", "running")
        async def async_guard(from_s, to_s, ctx):
            return ctx.get("ready", False)

        with pytest.raises(TransitionBlockedByGuard):
            await sm.transition_to("running", context={})

        await sm.transition_to("running", context={"ready": True})
        assert sm.current == "running"


# ═══════════════════════════════════════════════════════════════════════════
#  Hooks
# ═══════════════════════════════════════════════════════════════════════════


class TestHooks:
    @pytest.mark.asyncio
    async def test_on_enter_hook(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        entered = []

        @sm.on_enter("running")
        def on_enter_running(old, new, ctx):
            entered.append((old, new))

        await sm.transition_to("running")
        assert entered == [("idle", "running")]

    @pytest.mark.asyncio
    async def test_on_exit_hook(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        exited = []

        @sm.on_exit("idle")
        def on_exit_idle(old, new, ctx):
            exited.append((old, new))

        await sm.transition_to("running")
        assert exited == [("idle", "running")]

    @pytest.mark.asyncio
    async def test_on_change_hook(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        changes = []

        @sm.on_change
        def on_any_change(old, new, ctx):
            changes.append((old, new))

        await sm.transition_to("running")
        await sm.transition_to("done")
        assert changes == [("idle", "running"), ("running", "done")]

    @pytest.mark.asyncio
    async def test_async_hooks(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        entered = []

        @sm.on_enter("running")
        async def async_enter(old, new, ctx):
            entered.append(new)

        await sm.transition_to("running")
        assert entered == ["running"]

    @pytest.mark.asyncio
    async def test_force_transition_fires_hooks(self):
        sm = AgentStateMachine(
            initial="idle",
            transitions={"idle": ["running"]},
        )
        entered = []

        @sm.on_enter("error")
        def on_error(old, new, ctx):
            entered.append(new)

        # "error" not in transitions, but force should still fire hooks
        await sm.force_transition("error")
        assert entered == ["error"]


# ═══════════════════════════════════════════════════════════════════════════
#  Run gate
# ═══════════════════════════════════════════════════════════════════════════


class TestRunGate:
    def test_can_run_no_restriction(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        assert sm.can_run() is True

    def test_can_run_allowed_state(self):
        sm = AgentStateMachine(initial="running", transitions=None)
        sm.allow_run_when("running", "active")
        assert sm.can_run() is True

    def test_can_run_disallowed_state(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        sm.allow_run_when("running")
        assert sm.can_run() is False

    def test_allow_run_when_replace(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        sm.allow_run_when("running")
        sm.allow_run_when("idle")
        assert sm.can_run() is True

    def test_allow_run_when_no_args_removes_restriction(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        sm.allow_run_when("running")
        assert sm.can_run() is False
        sm.allow_run_when()
        assert sm.can_run() is True


# ═══════════════════════════════════════════════════════════════════════════
#  Reset
# ═══════════════════════════════════════════════════════════════════════════


class TestReset:
    @pytest.mark.asyncio
    async def test_reset_to_explicit_state(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        await sm.transition_to("running")
        sm.reset(state="idle")
        assert sm.current == "idle"
        assert sm.history == []

    @pytest.mark.asyncio
    async def test_reset_to_first_history_state(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        await sm.transition_to("running")
        await sm.transition_to("done")
        sm.reset()
        assert sm.current == "idle"
        assert sm.history == []

    def test_reset_no_history_keeps_current(self):
        sm = AgentStateMachine(initial="idle", transitions=None)
        sm.reset()
        assert sm.current == "idle"


# ═══════════════════════════════════════════════════════════════════════════
#  _call_maybe_async
# ═══════════════════════════════════════════════════════════════════════════


class TestCallMaybeAsync:
    @pytest.mark.asyncio
    async def test_sync_function(self):
        def sync_fn(x):
            return x * 2

        result = await _call_maybe_async(sync_fn, 5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_async_function(self):
        async def async_fn(x):
            return x * 3

        result = await _call_maybe_async(async_fn, 5)
        assert result == 15
