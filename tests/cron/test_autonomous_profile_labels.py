"""Hermes V2.9 — ``autonomous_profile`` is a routing/audit label only.

Contract under test:

* the label set is fail-closed validated (``light``, ``standard``,
  ``implementation``, ``large``, ``high-risk-review``, ``retry``,
  ``experimental``) with precedence job → config default → ``standard``;
* a label carries **no** resource envelope — no wall-clock, no max turns,
  no token budget, no USD budget;
* an agentic cron run is never stopped by scheduler wall-clock duration,
  API-call count, token totals, estimated USD, or inactivity;
* the supervised run still preserves the worker future, the caller's
  ContextVars, ``task_id`` propagation and the run_claim heartbeat;
* the scheduler selects *explicit* unlimited iterations for the agent —
  never ``agent.max_turns``, never the 500 default, never ``0``.

Fakes only — no provider, no network, no real AIAgent.
Written as ``unittest.TestCase`` classes so pytest collects them normally
while the file also stays runnable standalone.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import importlib
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

# Ensure the repo root is importable when this file runs standalone.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001 — probing an optional runtime dep
        return False


# The end-to-end ``run_job`` wiring needs the full provider runtime
# (dotenv/httpx/openai). CI installs it; a stripped sandbox may not.
_HAS_RUNTIME_DEPS = _importable("run_agent") and _importable(
    "hermes_cli.runtime_provider"
)


EXPECTED_LABELS = {
    "light",
    "standard",
    "implementation",
    "large",
    "high-risk-review",
    "retry",
    "experimental",
}

_PROBE: contextvars.ContextVar = contextvars.ContextVar("probe")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAgent:
    """Agent surface that blows through every historical resource cap."""

    def __init__(
        self,
        *,
        run_duration: float = 0.0,
        result: Optional[dict] = None,
        tokens: int = 0,
        cost_usd: float = 0.0,
        api_call_count: int = 0,
        idle_seconds: float = 0.0,
    ):
        self.session_total_tokens = tokens
        self.session_estimated_cost_usd = cost_usd
        self._api_call_count = api_call_count
        self._idle_seconds = idle_seconds
        self.max_iterations = 16
        self._run_duration = run_duration
        self._result = result or {
            "final_response": "done",
            "completed": True,
            "failed": False,
            "messages": [],
        }
        self.interrupt_calls: list[Any] = []
        self.run_started = False
        self.run_finished = False
        self.seen_task_id: Any = "<unset>"
        self.seen_context_value: Any = None

    def interrupt(self, message: str = None) -> None:
        self.interrupt_calls.append(message)

    def get_activity_summary(self) -> dict:
        return {
            "seconds_since_activity": self._idle_seconds,
            "last_activity_desc": "api_call",
            "current_tool": None,
            "api_call_count": self._api_call_count,
            "max_iterations": self.max_iterations,
        }

    def run_conversation(self, prompt: str, task_id: str = None) -> dict:
        self.run_started = True
        self.seen_task_id = task_id
        self.seen_context_value = _PROBE.get(None)
        deadline = time.monotonic() + self._run_duration
        while time.monotonic() < deadline:
            time.sleep(0.01)
        self.run_finished = True
        return dict(self._result)


def _supervise(agent, profile=None, **kwargs):
    """Call the supervised-run entry point with sane test defaults."""
    from cron.autonomous_limits import (
        resolve_autonomous_profile,
        run_supervised_agentic_run,
    )

    if profile is None:
        profile = resolve_autonomous_profile({}, {})
    kwargs.setdefault("prompt", "go")
    kwargs.setdefault("poll_interval", 0.05)
    return run_supervised_agentic_run(agent, profile, **kwargs)


# ---------------------------------------------------------------------------
# Label table
# ---------------------------------------------------------------------------


class TestProfileLabels(unittest.TestCase):
    def test_label_set_is_exactly_the_seven_allowed_values(self):
        from cron.autonomous_limits import PROFILE_LABELS

        self.assertEqual(set(PROFILE_LABELS), EXPECTED_LABELS)

    def test_experimental_is_a_valid_label(self):
        from cron.autonomous_limits import resolve_autonomous_profile

        self.assertEqual(
            resolve_autonomous_profile(
                {"autonomous_profile": "experimental"}, {}
            ).name,
            "experimental",
        )

    def test_label_carries_no_resource_envelope(self):
        """A profile is a label — it must not expose any cap fields."""
        from cron.autonomous_limits import resolve_autonomous_profile

        profile = resolve_autonomous_profile({"autonomous_profile": "large"}, {})
        for banned in ("timeout_seconds", "max_turns", "token_budget", "usd_budget"):
            with self.subTest(field=banned):
                self.assertFalse(
                    hasattr(profile, banned),
                    f"autonomous_profile must be a label only; found cap {banned!r}",
                )


class TestResolutionPrecedence(unittest.TestCase):
    def test_job_label_wins_over_config_default(self):
        from cron.autonomous_limits import resolve_autonomous_profile

        profile = resolve_autonomous_profile(
            {"autonomous_profile": "light"},
            {"cron": {"autonomous_limits": {"default_profile": "large"}}},
        )
        self.assertEqual(profile.name, "light")

    def test_config_default_used_when_job_has_no_label(self):
        from cron.autonomous_limits import resolve_autonomous_profile

        profile = resolve_autonomous_profile(
            {"id": "j1"},
            {"cron": {"autonomous_limits": {"default_profile": "implementation"}}},
        )
        self.assertEqual(profile.name, "implementation")

    def test_builtin_default_is_standard(self):
        from cron.autonomous_limits import resolve_autonomous_profile

        self.assertEqual(resolve_autonomous_profile({"id": "j1"}, {}).name, "standard")

    def test_existing_jobs_need_no_migration(self):
        """Every label persisted by the previous release stays valid."""
        from cron.autonomous_limits import resolve_autonomous_profile

        for legacy in (
            "light",
            "standard",
            "implementation",
            "large",
            "high-risk-review",
            "retry",
        ):
            with self.subTest(label=legacy):
                self.assertEqual(
                    resolve_autonomous_profile(
                        {"autonomous_profile": legacy}, {}
                    ).name,
                    legacy,
                )

    def test_unknown_and_sentinel_labels_fail_closed(self):
        from cron.autonomous_limits import (
            AutonomousProfileError,
            resolve_autonomous_profile,
        )

        for bad in ("turbo", "", "   ", "0", 0, 7, "unlimited", "none", "null", "false"):
            with self.subTest(bad=bad):
                with self.assertRaises(AutonomousProfileError):
                    resolve_autonomous_profile({"autonomous_profile": bad}, {})

    def test_unknown_config_default_fails_closed(self):
        from cron.autonomous_limits import (
            AutonomousProfileError,
            resolve_autonomous_profile,
        )

        with self.assertRaises(AutonomousProfileError):
            resolve_autonomous_profile(
                {"id": "j1"},
                {"cron": {"autonomous_limits": {"default_profile": "turbo"}}},
            )


class TestLabelSurfacesStayInSync(unittest.TestCase):
    """Every user-facing surface must offer exactly the canonical label set.

    The tool schema and the CLI ``--autonomous-profile`` choices are separate
    hard-coded lists; when ``experimental`` was added they were a real drift
    risk (the CLI would reject a label the resolver accepts).
    """

    def test_tool_schema_enum_matches_the_label_set(self):
        from cron.autonomous_limits import PROFILE_LABELS
        from tools.cronjob_tools import CRONJOB_SCHEMA

        enum = CRONJOB_SCHEMA["parameters"]["properties"]["autonomous_profile"][
            "enum"
        ]
        self.assertEqual(set(enum), set(PROFILE_LABELS))

    def test_cli_choices_match_the_label_set(self):
        import argparse

        from cron.autonomous_limits import PROFILE_LABELS
        from hermes_cli.subcommands.cron import build_cron_parser

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        build_cron_parser(subparsers, cmd_cron=lambda *_a, **_k: 0)

        found = 0
        for action in _walk_parser_actions(subparsers):
            if "--autonomous-profile" in (action.option_strings or []):
                found += 1
                self.assertEqual(
                    set(action.choices or ()),
                    set(PROFILE_LABELS),
                    "CLI --autonomous-profile choices drifted from PROFILE_LABELS",
                )
        self.assertGreaterEqual(found, 2, "expected create + edit to offer the flag")


def _walk_parser_actions(subparsers_action):
    """Yield every action of every (nested) subparser."""
    import argparse

    for parser in getattr(subparsers_action, "choices", {}).values():
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                yield from _walk_parser_actions(action)
            else:
                yield action


class TestValidateProfile(unittest.TestCase):
    """``validate_profile`` is the fail-closed gate the supervisor calls."""

    def test_accepts_a_known_label(self):
        from cron.autonomous_limits import AutonomousProfile, validate_profile

        profile = AutonomousProfile(name="retry")
        self.assertIs(validate_profile(profile), profile)

    def test_rejects_a_non_profile_object(self):
        from cron.autonomous_limits import AutonomousProfileError, validate_profile

        for bogus in ("standard", 3, None, object()):
            with self.subTest(bogus=bogus):
                with self.assertRaises(AutonomousProfileError):
                    validate_profile(bogus)

    def test_rejects_unknown_and_sentinel_labels(self):
        from cron.autonomous_limits import (
            AutonomousProfile,
            AutonomousProfileError,
            validate_profile,
        )

        for bad in ("turbo", "", "0", "unlimited", "none"):
            with self.subTest(bad=bad):
                with self.assertRaises(AutonomousProfileError):
                    validate_profile(AutonomousProfile(name=bad))

    def test_supervisor_refuses_an_invalid_label(self):
        """A bad label must never reach ``run_conversation``."""
        from cron.autonomous_limits import AutonomousProfile, AutonomousProfileError

        agent = FakeAgent()
        with self.assertRaises(AutonomousProfileError):
            _supervise(agent, profile=AutonomousProfile(name="turbo"))
        self.assertIs(agent.run_started, False)


# ---------------------------------------------------------------------------
# No autonomous caps
# ---------------------------------------------------------------------------


class TestNoAutonomousCaps(unittest.TestCase):
    def test_long_wall_clock_run_is_not_stopped(self):
        agent = FakeAgent(run_duration=0.6)
        result = _supervise(agent)
        self.assertIs(result["completed"], True)
        self.assertIs(agent.run_finished, True)
        self.assertEqual(agent.interrupt_calls, [])

    def test_huge_api_call_count_is_not_stopped(self):
        agent = FakeAgent(run_duration=0.4, api_call_count=10**6)
        self.assertIs(_supervise(agent)["completed"], True)
        self.assertEqual(agent.interrupt_calls, [])

    def test_huge_token_total_is_not_stopped(self):
        agent = FakeAgent(run_duration=0.4, tokens=10**9)
        self.assertIs(_supervise(agent)["completed"], True)
        self.assertEqual(agent.interrupt_calls, [])

    def test_huge_estimated_usd_is_not_stopped(self):
        agent = FakeAgent(run_duration=0.4, cost_usd=10_000.0)
        self.assertIs(_supervise(agent)["completed"], True)
        self.assertEqual(agent.interrupt_calls, [])

    def test_inactivity_does_not_stop_the_run(self):
        agent = FakeAgent(run_duration=0.4, idle_seconds=10_000.0)
        self.assertIs(_supervise(agent)["completed"], True)
        self.assertEqual(agent.interrupt_calls, [])

    def test_max_iterations_reached_is_not_converted_into_a_limit_failure(self):
        """No cap exists, so an agent-reported cap is passed through as-is."""
        agent = FakeAgent(
            result={
                "final_response": "partial",
                "completed": False,
                "failed": False,
                "turn_exit_reason": "max_iterations_reached(16/16)",
                "messages": [],
            }
        )
        result = _supervise(agent)
        self.assertEqual(result["turn_exit_reason"], "max_iterations_reached(16/16)")

    def test_supervisor_exposes_no_threshold_evaluator(self):
        import cron.autonomous_limits as al

        self.assertFalse(
            hasattr(al, "evaluate_thresholds"),
            "resource-threshold evaluation must not survive in cron agentic runs",
        )


# ---------------------------------------------------------------------------
# Explicit cancellation (the one thing that DOES stop a run)
# ---------------------------------------------------------------------------


class TestExplicitCancellation(unittest.TestCase):
    """Removing the caps must not remove caller-driven cancellation."""

    def test_operator_interrupt_ends_the_run_and_is_reported(self):
        """An external interrupt unwinds the agent; the supervisor returns it.

        This is the V2.9 stop mechanism: a real caller (operator, gateway
        shutdown, client disconnect) calls ``agent.interrupt()``; the
        conversation loop breaks and reports ``completed=False``. The
        supervisor must neither block nor rewrite that result.
        """
        cancelled = threading.Event()

        class Cancellable(FakeAgent):
            def run_conversation(self, prompt: str, task_id: str = None) -> dict:
                self.run_started = True
                # Wait for the operator's interrupt, then unwind like the
                # real conversation loop does.
                cancelled.wait(timeout=5)
                self.run_finished = True
                return {
                    "final_response": "partial work",
                    "completed": False,
                    "failed": False,
                    "turn_exit_reason": "interrupted_by_user",
                    "messages": [],
                }

        agent = Cancellable()

        def _operator_cancels():
            for _ in range(200):
                if agent.run_started:
                    break
                time.sleep(0.01)
            agent.interrupt("operator cancelled")
            cancelled.set()

        canceller = threading.Thread(target=_operator_cancels, daemon=True)
        canceller.start()
        try:
            result = _supervise(agent)
        finally:
            cancelled.set()
            canceller.join(timeout=5)

        self.assertIs(agent.run_finished, True)
        self.assertIs(result["completed"], False)
        self.assertEqual(result["turn_exit_reason"], "interrupted_by_user")
        # The interrupt came from the caller, never from the supervisor.
        self.assertEqual(agent.interrupt_calls, ["operator cancelled"])

    def test_supervisor_does_not_swallow_a_cancelled_error(self):
        """A worker cancelled at the future level propagates, not hangs."""

        class Cancelled(FakeAgent):
            def run_conversation(self, prompt: str, task_id: str = None) -> dict:
                raise concurrent.futures.CancelledError()

        with self.assertRaises(concurrent.futures.CancelledError):
            _supervise(Cancelled())


# ---------------------------------------------------------------------------
# Preserved run plumbing
# ---------------------------------------------------------------------------


class TestPreservedRunPlumbing(unittest.TestCase):
    def test_worker_future_is_attached_for_teardown(self):
        agent = FakeAgent(run_duration=0.1)
        _supervise(agent)
        future = getattr(agent, "_cron_worker_future", None)
        self.assertIsInstance(future, concurrent.futures.Future)
        self.assertIs(future.done(), True)

    def test_task_id_is_propagated(self):
        agent = FakeAgent()
        _supervise(agent, task_id="task-abc")
        self.assertEqual(agent.seen_task_id, "task-abc")

    def test_caller_contextvars_reach_the_worker(self):
        agent = FakeAgent()
        ctx = contextvars.copy_context()
        ctx.run(_PROBE.set, "from-scheduler")
        _supervise(agent, context=ctx)
        self.assertEqual(agent.seen_context_value, "from-scheduler")

    def test_heartbeat_keeps_firing_during_a_long_run(self):
        beats: list[int] = []
        agent = FakeAgent(run_duration=0.4)
        _supervise(agent, heartbeat_fn=lambda: beats.append(1))
        self.assertTrue(
            beats, "run_claim heartbeat must keep firing while the agent runs"
        )

    def test_agent_failure_is_propagated_not_swallowed(self):
        agent = FakeAgent(
            result={
                "final_response": "boom",
                "completed": False,
                "failed": True,
                "messages": [],
            }
        )
        self.assertIs(_supervise(agent)["failed"], True)

    def test_worker_exception_propagates(self):
        class Boom(FakeAgent):
            def run_conversation(self, prompt: str, task_id: str = None) -> dict:
                raise RuntimeError("provider exploded")

        with self.assertRaisesRegex(RuntimeError, "provider exploded"):
            _supervise(Boom())

    def test_non_dict_result_is_rejected(self):
        class Weird(FakeAgent):
            def run_conversation(self, prompt: str, task_id: str = None):
                return "not a dict"

        with self.assertRaises(RuntimeError):
            _supervise(Weird())


# ---------------------------------------------------------------------------
# Scheduler wiring
# ---------------------------------------------------------------------------


class TestSchedulerWiring(unittest.TestCase):
    """Dependency-free check that ``run_job`` sources its cap from the helper."""

    def test_run_job_uses_the_unlimited_cron_iteration_cap(self):
        import inspect

        from cron.scheduler import run_job

        src = inspect.getsource(run_job)
        self.assertTrue(
            "resolve_cron_iteration_cap()" in src,
            "run_job must take max_iterations from the explicit cron cap helper",
        )
        self.assertTrue(
            "max_iterations=max_iterations" in src,
            "run_job must pass the resolved cap straight to AIAgent",
        )

    def test_run_job_no_longer_derives_a_cap_or_inactivity_limit(self):
        import inspect

        from cron.scheduler import run_job

        src = inspect.getsource(run_job)
        for banned in (
            "_auto_profile.max_turns",
            "_cron_inactivity_limit",
            "inactivity_limit=",
            "run_with_autonomous_limits",
        ):
            with self.subTest(banned=banned):
                self.assertNotIn(banned, src)

    def test_cron_iteration_cap_helper_is_explicit_unlimited(self):
        from agent.iteration_budget import UNLIMITED_ITERATIONS
        from cron.autonomous_limits import resolve_cron_iteration_cap

        cap = resolve_cron_iteration_cap()
        self.assertIs(cap, UNLIMITED_ITERATIONS)
        self.assertNotIsInstance(cap, int)


@unittest.skipUnless(
    _HAS_RUNTIME_DEPS, "requires the full provider runtime (dotenv/httpx/openai)"
)
class TestSchedulerSelectsUnlimitedIterations(unittest.TestCase):
    """``run_job`` must hand AIAgent an explicit unlimited iteration budget."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self._env = patch.dict(
            os.environ,
            {
                "HERMES_HOME": str(self.tmp_path / "hermes-home"),
                "HERMES_MODEL": "test-cron-default-model",
            },
        )
        self._env.start()
        # Keep the agentic-efficiency ledger inside the sandbox.
        import cron.agentic_efficiency as ae

        self._ledger = patch.object(
            ae, "AGENTIC_EFFICIENCY_FILE", self.tmp_path / "agentic.db"
        )
        self._ledger.start()

    def tearDown(self):
        self._ledger.stop()
        self._env.stop()
        self._tmp.cleanup()

    def _agent_cls(self):
        agent = MagicMock()
        agent.session_total_tokens = 7
        agent.close = MagicMock()
        agent.run_conversation.return_value = {
            "final_response": "ok",
            "completed": True,
            "failed": False,
            "messages": [],
        }
        return MagicMock(return_value=agent), agent

    def _run_job(self, job: dict, agent_cls: MagicMock):
        # Import every patch target so ``patch("pkg.mod.attr")`` resolves.
        import hermes_cli.env_loader  # noqa: F401
        import hermes_cli.runtime_provider  # noqa: F401
        import hermes_state  # noqa: F401
        import run_agent  # noqa: F401
        from cron.scheduler import run_job

        fake_db = MagicMock()
        with patch("cron.scheduler._hermes_home", self.tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "test-key",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent", agent_cls):
            return run_job(job)

    def test_agent_is_constructed_with_explicit_unlimited(self):
        from agent.iteration_budget import UNLIMITED_ITERATIONS

        cls, _agent = self._agent_cls()
        success, _out, _final, error = self._run_job(
            {"id": "j-unlimited", "name": "t", "prompt": "hello"}, cls
        )
        self.assertEqual((success, error), (True, None))
        self.assertIs(cls.call_args.kwargs["max_iterations"], UNLIMITED_ITERATIONS)

    def test_cron_does_not_inherit_agent_max_turns_or_500(self):
        cls, _agent = self._agent_cls()
        with patch.dict(os.environ, {"HERMES_MAX_ITERATIONS": "500"}):
            self._run_job({"id": "j-no-inherit", "name": "t", "prompt": "hello"}, cls)
        chosen = cls.call_args.kwargs["max_iterations"]
        self.assertNotIn(
            chosen,
            (0, 500, 16),
            "cron must select unlimited explicitly, not a config/default cap",
        )
        self.assertNotIsInstance(chosen, int)

    def test_invalid_label_still_fails_closed_before_agent_spawn(self):
        cls, _agent = self._agent_cls()
        success, _out, _final, error = self._run_job(
            {
                "id": "j-bad-label",
                "name": "t",
                "prompt": "hello",
                "autonomous_profile": "turbo",
            },
            cls,
        )
        self.assertIs(success, False)
        self.assertTrue(error)
        self.assertEqual(cls.call_count, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
