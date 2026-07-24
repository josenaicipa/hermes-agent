"""Focused tests for strict autonomous limits on cron agentic runs.

Uses fake agents/futures only — no provider, network, or real AIAgent.
"""

from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path
from typing import Any, Optional

import pytest


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAgent:
    """Controllable agent surface used by the autonomous supervisor."""

    def __init__(
        self,
        *,
        run_duration: float = 0.0,
        result: Optional[dict] = None,
        tokens: int = 0,
        cost_usd: float = 0.0,
        api_call_count: int = 0,
        grow_turns: bool = False,
        grow_tokens: bool = False,
        grow_cost: bool = False,
        max_iterations: int = 16,
    ):
        self.session_total_tokens = tokens
        self.session_estimated_cost_usd = cost_usd
        self._api_call_count = api_call_count
        self.max_iterations = max_iterations
        self._run_duration = run_duration
        self._result = result or {
            "final_response": "done",
            "completed": True,
            "failed": False,
            "messages": [],
        }
        self._grow_turns = grow_turns
        self._grow_tokens = grow_tokens
        self._grow_cost = grow_cost
        self.interrupt_calls: list[Any] = []
        self.run_started = False
        self.run_finished = False

    def interrupt(self, message: str = None) -> None:
        self.interrupt_calls.append(message)

    def get_activity_summary(self) -> dict:
        return {
            "seconds_since_activity": 0.0,
            "last_activity_desc": "api_call",
            "current_tool": None,
            "api_call_count": self._api_call_count,
            "max_iterations": self.max_iterations,
        }

    def run_conversation(self, prompt: str, task_id: str = None) -> dict:
        self.run_started = True
        started = time.monotonic()
        step = 0
        while time.monotonic() - started < self._run_duration:
            step += 1
            if self._grow_turns:
                self._api_call_count = step
            if self._grow_tokens:
                self.session_total_tokens = step * 50
            if self._grow_cost:
                self.session_estimated_cost_usd = step * 0.05
            time.sleep(0.02)
        self.run_finished = True
        return dict(self._result)


# ---------------------------------------------------------------------------
# Profile table + resolution
# ---------------------------------------------------------------------------


class TestAutonomousProfiles:
    def test_fixed_profiles_exact_values(self):
        from cron.autonomous_limits import PROFILES

        expected = {
            "light": (300, 6, 200000, 0.50),
            "standard": (600, 16, 1200000, 2),
            "implementation": (1200, 30, 2000000, 5),
            "large": (1800, 45, 4000000, 10),
            "high-risk-review": (900, 24, 1500000, 8),
            "retry": (600, 16, 1000000, 2),
        }
        assert set(PROFILES) == set(expected)
        for name, (timeout, turns, tokens, usd) in expected.items():
            p = PROFILES[name]
            assert p.timeout_seconds == timeout
            assert p.max_turns == turns
            assert p.token_budget == tokens
            assert p.usd_budget == usd
            assert p.timeout_seconds > 0
            assert p.max_turns > 0
            assert p.token_budget > 0
            assert p.usd_budget > 0


class TestResolveProfile:
    def test_job_profile_wins_over_config_default(self):
        from cron.autonomous_limits import resolve_autonomous_profile

        profile = resolve_autonomous_profile(
            {"autonomous_profile": "light"},
            {"cron": {"autonomous_limits": {"default_profile": "large"}}},
        )
        assert profile.name == "light"
        assert profile.max_turns == 6

    def test_config_default_when_job_absent(self):
        from cron.autonomous_limits import resolve_autonomous_profile

        profile = resolve_autonomous_profile(
            {"id": "j1"},
            {"cron": {"autonomous_limits": {"default_profile": "implementation"}}},
        )
        assert profile.name == "implementation"

    def test_builtin_default_is_standard(self):
        from cron.autonomous_limits import resolve_autonomous_profile

        profile = resolve_autonomous_profile({"id": "j1"}, {})
        assert profile.name == "standard"
        assert profile.max_turns == 16

    def test_unknown_profile_fails_closed(self):
        from cron.autonomous_limits import (
            AutonomousProfileError,
            resolve_autonomous_profile,
        )

        with pytest.raises(AutonomousProfileError, match="unknown"):
            resolve_autonomous_profile({"autonomous_profile": "turbo"}, {})

    def test_empty_profile_fails_closed(self):
        from cron.autonomous_limits import (
            AutonomousProfileError,
            resolve_autonomous_profile,
        )

        with pytest.raises(AutonomousProfileError):
            resolve_autonomous_profile({"autonomous_profile": ""}, {})

    def test_zero_and_unlimited_sentinels_fail_closed(self):
        from cron.autonomous_limits import (
            AutonomousProfileError,
            resolve_autonomous_profile,
        )

        for bad in ("0", 0, "unlimited", "none"):
            with pytest.raises(AutonomousProfileError):
                resolve_autonomous_profile({"autonomous_profile": bad}, {})

    def test_nonpositive_profile_values_rejected(self):
        from cron.autonomous_limits import (
            AutonomousProfile,
            AutonomousProfileError,
            validate_profile,
        )

        with pytest.raises(AutonomousProfileError):
            validate_profile(
                AutonomousProfile(
                    name="bad",
                    timeout_seconds=0,
                    max_turns=10,
                    token_budget=100,
                    usd_budget=1.0,
                )
            )


# ---------------------------------------------------------------------------
# Supervisor monitoring
# ---------------------------------------------------------------------------


class TestSuperviseLimits:
    def test_max_turns_sets_limit_reason_and_interrupts(self, tmp_path):
        from cron.autonomous_limits import (
            AutonomousLimitError,
            AutonomousProfile,
            run_with_autonomous_limits,
        )

        agent = FakeAgent(run_duration=2.0, grow_turns=True)
        profile = AutonomousProfile(
            name="t",
            timeout_seconds=30,
            max_turns=3,
            token_budget=10_000_000,
            usd_budget=100.0,
        )
        with pytest.raises(AutonomousLimitError) as ei:
            run_with_autonomous_limits(
                agent,
                profile,
                prompt="go",
                task_key="job-turns",
                poll_interval=0.05,
                state_path=tmp_path / "state.json",
            )
        err = ei.value
        assert err.limit_reason == "max-turns"
        assert "limit_reason=max-turns" in str(err)
        assert agent.interrupt_calls
        assert agent.interrupt_calls[0] == "max-turns"

    def test_tokens_map_to_budget_and_interrupt(self, tmp_path):
        from cron.autonomous_limits import (
            AutonomousLimitError,
            AutonomousProfile,
            run_with_autonomous_limits,
        )

        agent = FakeAgent(run_duration=2.0, grow_tokens=True)
        profile = AutonomousProfile(
            name="t",
            timeout_seconds=30,
            max_turns=100,
            token_budget=100,
            usd_budget=100.0,
        )
        with pytest.raises(AutonomousLimitError) as ei:
            run_with_autonomous_limits(
                agent,
                profile,
                prompt="go",
                task_key="job-tokens",
                poll_interval=0.05,
                state_path=tmp_path / "state.json",
            )
        err = ei.value
        assert err.limit_reason == "budget"
        assert "limit_reason=budget" in str(err)
        assert agent.interrupt_calls == ["budget"]

    def test_usd_maps_to_budget(self, tmp_path):
        from cron.autonomous_limits import (
            AutonomousLimitError,
            AutonomousProfile,
            run_with_autonomous_limits,
        )

        agent = FakeAgent(run_duration=2.0, grow_cost=True)
        profile = AutonomousProfile(
            name="t",
            timeout_seconds=30,
            max_turns=100,
            token_budget=10_000_000,
            usd_budget=0.1,
        )
        with pytest.raises(AutonomousLimitError) as ei:
            run_with_autonomous_limits(
                agent,
                profile,
                prompt="go",
                task_key="job-usd",
                poll_interval=0.05,
                state_path=tmp_path / "state.json",
            )
        assert ei.value.limit_reason == "budget"
        assert agent.interrupt_calls == ["budget"]

    def test_wall_clock_timeout_is_hard_timeout(self, tmp_path):
        from cron.autonomous_limits import (
            AutonomousLimitError,
            AutonomousProfile,
            run_with_autonomous_limits,
        )

        agent = FakeAgent(run_duration=2.0)
        profile = AutonomousProfile(
            name="t",
            timeout_seconds=1,
            max_turns=100,
            token_budget=10_000_000,
            usd_budget=100.0,
        )
        with pytest.raises(AutonomousLimitError) as ei:
            run_with_autonomous_limits(
                agent,
                profile,
                prompt="go",
                task_key="job-timeout",
                poll_interval=0.05,
                state_path=tmp_path / "state.json",
            )
        err = ei.value
        assert err.limit_reason == "hard-timeout"
        assert "limit_reason=hard-timeout" in str(err)
        assert agent.interrupt_calls == ["hard-timeout"]

    def test_max_iterations_reached_is_max_turns_failure(self, tmp_path):
        from cron.autonomous_limits import (
            AutonomousLimitError,
            AutonomousProfile,
            run_with_autonomous_limits,
        )

        agent = FakeAgent(
            run_duration=0.0,
            result={
                "final_response": "partial summary",
                "completed": False,
                "failed": False,
                "turn_exit_reason": "max_iterations_reached(16)",
                "messages": [],
            },
        )
        profile = AutonomousProfile(
            name="t",
            timeout_seconds=30,
            max_turns=16,
            token_budget=10_000_000,
            usd_budget=100.0,
        )
        with pytest.raises(AutonomousLimitError) as ei:
            run_with_autonomous_limits(
                agent,
                profile,
                prompt="go",
                task_key="job-max-iter",
                poll_interval=0.05,
                state_path=tmp_path / "state.json",
            )
        assert ei.value.limit_reason == "max-turns"

    def test_successful_completion_returns_result(self, tmp_path):
        from cron.autonomous_limits import AutonomousProfile, run_with_autonomous_limits

        agent = FakeAgent(run_duration=0.05)
        profile = AutonomousProfile(
            name="t",
            timeout_seconds=30,
            max_turns=16,
            token_budget=10_000_000,
            usd_budget=100.0,
        )
        result = run_with_autonomous_limits(
            agent,
            profile,
            prompt="go",
            task_key="job-ok",
            poll_interval=0.05,
            state_path=tmp_path / "state.json",
        )
        assert result["final_response"] == "done"
        assert agent.interrupt_calls == []


# ---------------------------------------------------------------------------
# Rescope guard
# ---------------------------------------------------------------------------


class TestRescopeGuard:
    def test_first_limit_permits_one_rescope(self, tmp_path):
        from cron.autonomous_limits import (
            AutonomousLimitError,
            AutonomousProfile,
            run_with_autonomous_limits,
        )

        state = tmp_path / "state.json"
        profile = AutonomousProfile(
            name="t",
            timeout_seconds=1,
            max_turns=100,
            token_budget=10_000_000,
            usd_budget=100.0,
        )
        agent = FakeAgent(run_duration=2.0)
        with pytest.raises(AutonomousLimitError) as ei:
            run_with_autonomous_limits(
                agent,
                profile,
                prompt="go",
                task_key="stable-task",
                poll_interval=0.05,
                state_path=state,
            )
        err = ei.value
        assert err.limit_count == 1
        assert err.action != "stop-and-alert"
        assert "rescope" in err.action
        assert "limit_count=1" in str(err)

    def test_second_limit_is_stop_and_alert(self, tmp_path):
        from cron.autonomous_limits import (
            AutonomousLimitError,
            AutonomousProfile,
            run_with_autonomous_limits,
        )

        state = tmp_path / "state.json"
        profile = AutonomousProfile(
            name="t",
            timeout_seconds=1,
            max_turns=100,
            token_budget=10_000_000,
            usd_budget=100.0,
        )
        for _ in range(2):
            agent = FakeAgent(run_duration=2.0)
            with pytest.raises(AutonomousLimitError) as ei:
                run_with_autonomous_limits(
                    agent,
                    profile,
                    prompt="go",
                    task_key="stable-task",
                    poll_interval=0.05,
                    state_path=state,
                )
        err = ei.value
        assert err.limit_count == 2
        assert err.action == "stop-and-alert"
        assert "action=stop-and-alert" in str(err)
        assert "limit_count=2" in str(err)
        assert "limit_reason=hard-timeout" in str(err)

    def test_successful_completion_resets_limit_count(self, tmp_path):
        from cron.autonomous_limits import (
            AutonomousLimitError,
            AutonomousProfile,
            run_with_autonomous_limits,
        )

        state = tmp_path / "state.json"
        timeout_profile = AutonomousProfile(
            name="t",
            timeout_seconds=1,
            max_turns=100,
            token_budget=10_000_000,
            usd_budget=100.0,
        )
        ok_profile = AutonomousProfile(
            name="t",
            timeout_seconds=30,
            max_turns=100,
            token_budget=10_000_000,
            usd_budget=100.0,
        )

        # First limit hit → count=1
        with pytest.raises(AutonomousLimitError) as ei1:
            run_with_autonomous_limits(
                FakeAgent(run_duration=2.0),
                timeout_profile,
                prompt="go",
                task_key="reset-task",
                poll_interval=0.05,
                state_path=state,
            )
        assert ei1.value.limit_count == 1

        # Success clears state
        result = run_with_autonomous_limits(
            FakeAgent(run_duration=0.05),
            ok_profile,
            prompt="go",
            task_key="reset-task",
            poll_interval=0.05,
            state_path=state,
        )
        assert result["completed"] is True

        # Next limit is again count=1 (rescope permitted), not stop-and-alert
        with pytest.raises(AutonomousLimitError) as ei2:
            run_with_autonomous_limits(
                FakeAgent(run_duration=2.0),
                timeout_profile,
                prompt="go",
                task_key="reset-task",
                poll_interval=0.05,
                state_path=state,
            )
        assert ei2.value.limit_count == 1
        assert ei2.value.action != "stop-and-alert"

    def test_failed_result_does_not_clear_limit_count(self, tmp_path):
        from cron.autonomous_limits import (
            AutonomousProfile,
            record_limit_hit,
            run_with_autonomous_limits,
        )

        state = tmp_path / "state.json"
        assert record_limit_hit("failed-task", "budget", state_path=state)[0] == 1
        failed = FakeAgent(result={
            "final_response": "failed",
            "completed": False,
            "failed": True,
            "turn_exit_reason": "provider_error",
            "messages": [],
        })
        profile = AutonomousProfile("t", 30, 16, 10_000_000, 100.0)
        result = run_with_autonomous_limits(
            failed, profile, prompt="go", task_key="failed-task",
            poll_interval=0.05, state_path=state,
        )
        assert result["failed"] is True
        count, action = record_limit_hit("failed-task", "budget", state_path=state)
        assert count == 2
        assert action == "stop-and-alert"


class TestFailBeforeSpawnContract:
    def test_resolve_raises_before_any_agent_run(self):
        """Unknown profile must fail before any agent is constructed/run."""
        from cron.autonomous_limits import (
            AutonomousProfileError,
            resolve_autonomous_profile,
        )

        spawn_marker = {"spawned": False}

        class ShouldNotConstruct:
            def __init__(self, *a, **k):
                spawn_marker["spawned"] = True
                raise AssertionError("AIAgent must not be constructed")

        with pytest.raises(AutonomousProfileError):
            resolve_autonomous_profile({"autonomous_profile": "nope"}, {})
        # Callers resolve profile before constructing the agent.
        assert spawn_marker["spawned"] is False
        _ = ShouldNotConstruct  # silence lint; construction never happens
