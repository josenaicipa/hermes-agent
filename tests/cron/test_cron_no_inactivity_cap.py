"""V2.9 contract: cron agentic runs have NO inactivity timeout.

This file previously simulated the scheduler's idle-agent watchdog (poll
``get_activity_summary``, interrupt after N seconds of no activity) and
asserted that ``HERMES_CRON_TIMEOUT`` was an agent kill threshold. Both are
gone: a long quiet stretch (slow provider, long-running tool) is not evidence
of a hung run, and killing on it truncated legitimate work. See
``docs/adr/ADR-0002-autonomous-profiles-as-labels.md``.

What remains here is the *negative* contract — a regression guard so no
wall-clock / inactivity / turn / token / spend cap can come back into the
scheduler or the supervisor.

Related coverage lives elsewhere and is deliberately not duplicated:

* positive behavior (an agent reporting huge idle time still completes;
  heartbeat, worker future, ContextVars, task_id and cancellation are
  preserved) → ``tests/cron/test_autonomous_profile_labels.py``;
* the one surviving role of ``HERMES_CRON_TIMEOUT`` — deriving the one-shot
  run-claim dead-owner TTL, which never stops a run →
  ``tests/cron/test_jobs.py::test_run_claim_ttl_derived_from_cron_timeout``.
"""

import ast
import inspect
import sys
import textwrap
from pathlib import Path


# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestNoInactivityWatchdog:
    """The scheduler must not reacquire an inactivity/wall-clock kill path."""

    def test_run_job_has_no_inactivity_machinery(self):
        from cron.scheduler import run_job

        src = inspect.getsource(run_job)
        for banned in (
            "HERMES_CRON_TIMEOUT",
            "_cron_inactivity_limit",
            "inactivity_limit",
            "seconds_since_activity",
        ):
            assert banned not in src, (
                f"run_job must not reference {banned!r}: agentic cron runs are "
                "not stopped by inactivity or wall-clock duration"
            )

    def test_supervisor_takes_no_inactivity_or_budget_arguments(self):
        from cron.autonomous_limits import run_supervised_agentic_run

        params = set(inspect.signature(run_supervised_agentic_run).parameters)
        for banned in (
            "inactivity_limit",
            "timeout",
            "timeout_seconds",
            "max_turns",
            "token_budget",
            "usd_budget",
        ):
            assert banned not in params, f"supervisor must not accept {banned!r}"

    def test_supervisor_never_interrupts_the_agent(self):
        """Only a real caller/operator cancels; the supervisor never does."""
        from cron.autonomous_limits import run_supervised_agentic_run

        # Inspect executable calls, not source text: the docstring legitimately
        # describes caller-driven ``agent.interrupt()``.
        tree = ast.parse(textwrap.dedent(inspect.getsource(run_supervised_agentic_run)))
        interrupt_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "interrupt"
        ]
        assert not interrupt_calls, (
            "the supervisor must never call agent.interrupt() — cancellation "
            "is caller-driven in V2.9"
        )

    def test_scheduler_does_not_import_a_limit_supervisor(self):
        from cron.scheduler import run_job

        src = inspect.getsource(run_job)
        assert "run_with_autonomous_limits" not in src
        assert "run_supervised_agentic_run" in src


class TestSysPathOrdering:
    """Test that sys.path is set before repo-level imports."""

    def test_hermes_time_importable(self):
        """hermes_time should be importable when cron.scheduler loads."""
        # This import would fail if sys.path.insert comes after the import
        from cron.scheduler import _hermes_now
        assert callable(_hermes_now)

    def test_hermes_constants_importable(self):
        """hermes_constants should be importable from cron context."""
        from hermes_constants import get_hermes_home
        assert callable(get_hermes_home)
