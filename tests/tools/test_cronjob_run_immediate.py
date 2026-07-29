"""Tests for non-blocking cronjob action='run' execution (#41037).

Before this fix, `cronjob(action='run')` only set next_run_at=now and returned
success, relying on the scheduler ticker to actually run the job. With no
gateway/ticker active (e.g. a CLI-only Windows setup) the job never executed
and last_run_at stayed null forever.

The run path claims the job immediately (blocking a concurrent tick and a
duplicate in-process manual run), then dispatches the shared run_one_job body
on a worker thread and returns without freezing the calling conversation -- a
cron that waited on TERMINAL_CWD or did long agent work used to hang the
interactive turn for minutes.
"""

import json
import threading
import time
from unittest.mock import patch

from tools.cronjob_tools import (
    cronjob,
    _dispatch_job_now,
    _execute_job_now,
    _manual_run_threads,
    _manual_runs_lock,
)


_JOB = {
    "id": "job-run-1",
    "name": "manual run",
    "prompt": "hi",
    "schedule": {"kind": "cron", "expr": "0 9 * * *"},
}


def _claimed_job(job: dict, *, owner: str = "test-owner:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") -> dict:
    """Post-claim store record with the caller's fencing token (exact match required)."""
    return {
        **job,
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": owner},
    }


class TestCronjobRunExecutesImmediately:
    def test_run_action_returns_pending_dispatch(self):
        """action='run' returns immediately with an explicit pending state."""
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools._dispatch_job_now", return_value={
                 "claimed": True, "dispatched": True, "error": None,
             }), \
             patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["success"] is True
        assert out["job"]["dispatched"] is True
        assert out["job"]["execution_pending"] is True
        assert out["job"]["executed"] is True
        assert out["job"]["execution_success"] is None

    def test_dispatch_claims_then_fires_on_worker_thread(self):
        """The actual scheduler body runs after the tool has dispatched it."""
        started = threading.Event()
        release = threading.Event()
        done = threading.Event()
        worker_daemon = []
        token = "dispatch-owner:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        claimed = _claimed_job(_JOB, owner=token)

        def _run(_job):
            worker_daemon.append(threading.current_thread().daemon)
            started.set()
            try:
                release.wait(timeout=5)
            finally:
                done.set()
            return True

        with patch("tools.cronjob_tools.new_fire_claim_owner", return_value=token), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=True) as m_claim, \
             patch("tools.cronjob_tools.get_job", return_value=claimed), \
             patch("cron.scheduler.run_one_job", side_effect=_run) as m_run:
            before = time.monotonic()
            result = _dispatch_job_now(dict(_JOB))
            elapsed = time.monotonic() - before
            assert result == {"claimed": True, "dispatched": True, "error": None}
            assert elapsed < 0.5
            assert started.wait(timeout=5)
            release.set()
            assert done.wait(timeout=5)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with _manual_runs_lock:
                    if "job-run-1" not in _manual_run_threads:
                        break
                time.sleep(0.01)
            with _manual_runs_lock:
                assert "job-run-1" not in _manual_run_threads

        assert worker_daemon == [False]
        m_claim.assert_called_once_with("job-run-1", claim_owner=token)
        m_run.assert_called_once()
        assert m_run.call_args[0][0]["fire_claim"]["by"] == token

    def test_dispatch_skips_when_claim_lost(self):
        """If the scheduler owns the fire claim, do not start a worker."""
        job = dict(_JOB, id="job-run-claim-lost")
        with patch("tools.cronjob_tools.claim_job_for_fire", return_value=False), \
             patch("cron.scheduler.run_one_job") as m_run:
            result = _dispatch_job_now(job)

        assert result["claimed"] is False
        assert result["dispatched"] is False
        m_run.assert_not_called()

    def test_dispatch_marks_failure_when_thread_cannot_start(self):
        """A start failure clears the local handle and durable fire claim."""
        job = dict(_JOB, id="job-run-start-failure")
        token = "start-owner:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        claimed = _claimed_job(job, owner=token)
        with patch("tools.cronjob_tools.new_fire_claim_owner", return_value=token), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
             patch("tools.cronjob_tools.get_job", return_value=claimed), \
             patch("tools.cronjob_tools.threading.Thread.start", side_effect=RuntimeError("no thread")), \
             patch("tools.cronjob_tools.mark_job_run") as m_mark:
            result = _dispatch_job_now(job)

        assert result["claimed"] is True
        assert result["dispatched"] is False
        assert "no thread" in result["error"]
        m_mark.assert_called_once_with(
            "job-run-start-failure",
            False,
            "no thread",
            expected_fire_claim_owner=token,
        )
        with _manual_runs_lock:
            assert "job-run-start-failure" not in _manual_run_threads

    def test_run_skips_when_claim_lost(self):
        """If the scheduler already holds the fire claim, do NOT double-run.

        Integration variant of test_dispatch_skips_when_claim_lost: goes
        through the real cronjob() -> _dispatch_job_now path (only
        claim_job_for_fire is mocked), so it also pins that a lost claim
        never triggers the provider reconciliation notify.
        """
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=False), \
             patch("cron.scheduler.run_one_job") as m_run, \
             patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools._notify_provider_jobs_changed_safe") as m_notify:
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["success"] is True
        assert out["job"]["executed"] is False
        assert out["job"]["execution_success"] is False
        assert out["job"]["execution_pending"] is False
        assert "execution_skipped" in out["job"]
        m_run.assert_not_called()  # claim lost -> never fired
        m_notify.assert_not_called()  # the winning scheduler owns the re-arm

    def test_run_response_reports_claim_loss(self):
        """Tool-level response shape when _dispatch_job_now itself reports a
        lost claim (unit test: _dispatch_job_now mocked directly)."""
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools._dispatch_job_now", return_value={
                 "claimed": False,
                 "dispatched": False,
                 "error": "Job is already being fired by the scheduler; not run again.",
             }), \
             patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["success"] is True
        assert out["job"]["executed"] is False
        assert out["job"]["execution_success"] is False
        assert out["job"]["execution_pending"] is False
        assert "execution_skipped" in out["job"]
        assert "execution_error" in out["job"]

    def test_run_reconciles_external_provider_after_claimed_execution(self):
        """A direct run must re-arm Chronos after it advances next_run_at.

        Otherwise a scheduled Chronos fire that loses its claim to this direct
        run is consumed without a successor one-shot, permanently stalling the
        recurring job. Claiming happens synchronously inside cronjob(); the
        run itself now happens on a worker thread (see _dispatch_job_now), so
        reconciliation fires once at claim time and again once the run
        persists its final state -- both are exercised here, in order, via a
        gate on the mocked run_one_job.
        """
        order = []
        release_run = threading.Event()

        def _run(*_a, **_kw):
            release_run.wait(timeout=5)
            order.append("run")
            return True

        ran = {
            "id": "job-run-1", "last_status": "ok", "last_error": None,
            "fire_claim": {"by": "test-owner"},
        }
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools.new_fire_claim_owner", return_value="test-owner"), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
             patch("cron.scheduler.run_one_job", side_effect=_run), \
             patch("tools.cronjob_tools.get_job", return_value=ran), \
             patch("tools.cronjob_tools._notify_provider_jobs_changed_safe",
                   side_effect=lambda: order.append("notify")) as m_notify:
            out = json.loads(cronjob(action="run", job_id="job-run-1"))
            # The claim-time reconciliation has already run synchronously on
            # this thread by the time cronjob() returns; the worker is still
            # gated before appending "run".
            assert order == ["notify"]
            worker = _manual_run_threads.get("job-run-1")
            release_run.set()
            if worker is not None:
                worker.join(timeout=5)

        assert out["job"]["executed"] is True
        # Reconciled once at claim time and once more after the run persisted
        # its final state (mark_job_run inside run_one_job in production).
        assert order == ["notify", "run", "notify"]
        assert m_notify.call_count == 2

    def test_run_reconciles_external_provider_even_when_claimed_run_fails(self):
        """A claimed direct run advances next_run_at at claim time, so the
        provider must be reconciled even when the execution itself fails --
        again both at claim time and once the failure is persisted."""
        order = []
        release_run = threading.Event()

        def _run(*_a, **_kw):
            release_run.wait(timeout=5)
            order.append("run")
            raise RuntimeError("boom")

        failed = {
            "id": "job-run-1", "last_status": "error", "last_error": "provider 500",
            "fire_claim": {"by": "test-owner"},
        }
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools.new_fire_claim_owner", return_value="test-owner"), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
             patch("cron.scheduler.run_one_job", side_effect=_run), \
             patch("tools.cronjob_tools.mark_job_run"), \
             patch("tools.cronjob_tools.get_job", return_value=failed), \
             patch("tools.cronjob_tools._notify_provider_jobs_changed_safe",
                   side_effect=lambda: order.append("notify")) as m_notify:
            out = json.loads(cronjob(action="run", job_id="job-run-1"))
            assert order == ["notify"]
            worker = _manual_run_threads.get("job-run-1")
            release_run.set()
            if worker is not None:
                worker.join(timeout=5)

        assert out["job"]["executed"] is True
        # Dispatch is non-blocking, so the synchronous response can only ever
        # report "pending" -- the failure this test is really about is only
        # visible once the worker thread persists it (mark_job_run), which is
        # exercised above via the join() + order/call-count assertions.
        assert out["job"]["execution_success"] is None
        assert order == ["notify", "run", "notify"]
        assert m_notify.call_count == 2

    def test_execute_job_now_bails_without_claim(self):
        """The synchronous helper still preserves its at-most-once contract."""
        with patch("tools.cronjob_tools.claim_job_for_fire", return_value=False), \
             patch("cron.scheduler.run_one_job") as m_run:
            res = _execute_job_now(dict(_JOB))
        assert res["claimed"] is False
        assert res["success"] is False
        m_run.assert_not_called()

    def test_execute_job_now_marks_failure_on_exception(self):
        """An exception during synchronous fire is marked failed, not propagated."""
        token = "sync-owner:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        claimed = _claimed_job(_JOB, owner=token)
        with patch("tools.cronjob_tools.new_fire_claim_owner", return_value=token), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
             patch("cron.scheduler.run_one_job", side_effect=RuntimeError("boom")), \
             patch("tools.cronjob_tools.mark_job_run") as m_mark, \
             patch("tools.cronjob_tools.get_job", return_value=claimed):
            res = _execute_job_now(dict(_JOB))
        assert res["claimed"] is True
        assert res["success"] is False
        assert "boom" in res["error"]
        m_mark.assert_called_once_with(
            "job-run-1",
            False,
            "boom",
            expected_fire_claim_owner=token,
        )
