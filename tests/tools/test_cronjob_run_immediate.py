"""Tests for non-blocking cronjob action='run' execution.

The run path claims the job immediately (blocking a concurrent tick), dispatches
the shared run_one_job body on a worker thread, and returns without freezing the
calling conversation.
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

        def _run(_job):
            worker_daemon.append(threading.current_thread().daemon)
            started.set()
            try:
                release.wait(timeout=5)
            finally:
                done.set()
            return True

        with patch("tools.cronjob_tools.claim_job_for_fire", return_value=True) as m_claim, \
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
        m_claim.assert_called_once_with("job-run-1")
        m_run.assert_called_once()

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
        with patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
             patch("tools.cronjob_tools.threading.Thread.start", side_effect=RuntimeError("no thread")), \
             patch("tools.cronjob_tools.mark_job_run") as m_mark:
            result = _dispatch_job_now(job)

        assert result["claimed"] is True
        assert result["dispatched"] is False
        assert "no thread" in result["error"]
        m_mark.assert_called_once_with("job-run-start-failure", False, "no thread")
        with _manual_runs_lock:
            assert "job-run-start-failure" not in _manual_run_threads

    def test_run_response_reports_claim_loss(self):
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
        with patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
             patch("cron.scheduler.run_one_job", side_effect=RuntimeError("boom")), \
             patch("tools.cronjob_tools.mark_job_run") as m_mark, \
             patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)):
            res = _execute_job_now(dict(_JOB))
        assert res["claimed"] is True
        assert res["success"] is False
        assert "boom" in res["error"]
        m_mark.assert_called_once()
