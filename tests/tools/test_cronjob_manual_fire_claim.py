"""Manual cronjob(action='run') must dispatch the claimed job record.

The durable fire_claim owner lives on the store record stamped by
``claim_job_for_fire``. The worker must run that refreshed record so the
fire_claim heartbeat can see ``claim["by"]`` — the pre-claim snapshot does
not carry it.

Post-claim verification requires the durable ``fire_claim.by`` to match the
exact fencing token this caller generated and passed into the claim.
"""

import threading
import time
from unittest.mock import patch

from tools.cronjob_tools import _dispatch_job_now, _execute_job_now, _manual_runs_lock, _manual_run_threads


_JOB = {
    "id": "job-claimed-record",
    "name": "manual claim record",
    "prompt": "hi",
    "schedule": {"kind": "interval", "minutes": 5},
}


def test_dispatch_runs_claimed_store_record_with_fire_claim():
    """After winning the claim, the worker receives the store's fire_claim."""
    token = "dispatch-owner:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    claimed = {
        **_JOB,
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": token},
        "next_run_at": "2026-07-12T12:05:00+00:00",
    }
    started = threading.Event()
    seen = {}

    def _run(job):
        seen["job"] = job
        started.set()
        return True

    with patch("tools.cronjob_tools.new_fire_claim_owner", return_value=token), \
         patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
         patch("tools.cronjob_tools.get_job", return_value=claimed), \
         patch("cron.scheduler.run_one_job", side_effect=_run):
        result = _dispatch_job_now(dict(_JOB))
        assert result == {"claimed": True, "dispatched": True, "error": None}
        assert started.wait(timeout=5)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with _manual_runs_lock:
            if "job-claimed-record" not in _manual_run_threads:
                break
        time.sleep(0.01)

    assert seen["job"]["fire_claim"]["by"] == token
    assert seen["job"]["next_run_at"] == "2026-07-12T12:05:00+00:00"


def test_execute_job_now_runs_claimed_store_record_with_fire_claim():
    """Synchronous immediate run also uses the post-claim store record."""
    token = "sync-owner:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    claimed = {
        **_JOB,
        "id": "job-sync-claimed",
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": token},
    }
    seen = {}

    def _run(job):
        seen["job"] = job
        return True

    with patch("tools.cronjob_tools.new_fire_claim_owner", return_value=token), \
         patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
         patch("tools.cronjob_tools.get_job", side_effect=[claimed, claimed]), \
         patch("cron.scheduler.run_one_job", side_effect=_run):
        res = _execute_job_now(dict(_JOB, id="job-sync-claimed"))

    assert res["claimed"] is True
    assert seen["job"]["fire_claim"]["by"] == token


def test_execute_job_now_outer_exception_marks_with_expected_fire_claim_owner():
    """Outer failure after a successful claim must not erase a replacement owner."""
    token = "sync-owner:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    claimed = {
        **_JOB,
        "id": "job-outer-fail",
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": token},
    }
    mark_calls = []

    def _run(_job):
        raise RuntimeError("outer boom")

    def _mark(job_id, success, error=None, **kwargs):
        mark_calls.append(
            {
                "job_id": job_id,
                "success": success,
                "error": error,
                "kwargs": kwargs,
            }
        )

    with patch("tools.cronjob_tools.new_fire_claim_owner", return_value=token), \
         patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
         patch("tools.cronjob_tools.get_job", return_value=claimed), \
         patch("cron.scheduler.run_one_job", side_effect=_run), \
         patch("tools.cronjob_tools.mark_job_run", side_effect=_mark):
        res = _execute_job_now(dict(_JOB, id="job-outer-fail"))

    assert res["claimed"] is True
    assert res["success"] is False
    assert mark_calls
    assert mark_calls[0]["job_id"] == "job-outer-fail"
    assert mark_calls[0]["success"] is False
    assert mark_calls[0]["kwargs"].get("expected_fire_claim_owner") == token


def test_dispatch_job_now_worker_exception_marks_with_expected_fire_claim_owner():
    """Background worker outer failure passes the stable claim owner to mark."""
    token = "dispatch-owner:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    claimed = {
        **_JOB,
        "id": "job-dispatch-outer",
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": token},
    }
    mark_calls = []
    failed = threading.Event()

    def _run(_job):
        raise RuntimeError("worker boom")

    def _mark(job_id, success, error=None, **kwargs):
        mark_calls.append(
            {
                "job_id": job_id,
                "success": success,
                "error": error,
                "kwargs": kwargs,
            }
        )
        failed.set()

    with patch("tools.cronjob_tools.new_fire_claim_owner", return_value=token), \
         patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
         patch("tools.cronjob_tools.get_job", return_value=claimed), \
         patch("cron.scheduler.run_one_job", side_effect=_run), \
         patch("tools.cronjob_tools.mark_job_run", side_effect=_mark):
        result = _dispatch_job_now(dict(_JOB, id="job-dispatch-outer"))
        assert result == {"claimed": True, "dispatched": True, "error": None}
        assert failed.wait(timeout=5)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with _manual_runs_lock:
            if "job-dispatch-outer" not in _manual_run_threads:
                break
        time.sleep(0.01)

    assert mark_calls
    assert mark_calls[0]["job_id"] == "job-dispatch-outer"
    assert mark_calls[0]["success"] is False
    assert mark_calls[0]["kwargs"].get("expected_fire_claim_owner") == token


def test_execute_job_now_post_claim_reread_missing_fail_closed():
    """claim won but get_job is None => no run_one_job, no pre-claim fallback."""
    token = "sync-owner:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    run_calls = []
    mark_calls = []

    def _run(job):
        run_calls.append(job)
        return True

    def _mark(*_a, **_k):
        mark_calls.append(True)

    with patch("tools.cronjob_tools.new_fire_claim_owner", return_value=token), \
         patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
         patch("tools.cronjob_tools.get_job", return_value=None), \
         patch("cron.scheduler.run_one_job", side_effect=_run), \
         patch("tools.cronjob_tools.mark_job_run", side_effect=_mark):
        res = _execute_job_now(dict(_JOB, id="job-missing-after-claim"))

    assert res["claimed"] is True
    assert res["success"] is False
    assert res["error"]
    assert "fire_claim" in res["error"].lower() or "token" in res["error"].lower() or "missing" in res["error"].lower()
    assert run_calls == []
    # Owner cannot be recovered — leave durable claim for TTL recovery.
    assert mark_calls == []


def test_dispatch_job_now_post_claim_reread_missing_fail_closed():
    """claim won but get_job is None => no worker, no pre-claim fallback."""
    token = "dispatch-owner:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    run_calls = []
    mark_calls = []

    def _run(job):
        run_calls.append(job)
        return True

    def _mark(*_a, **_k):
        mark_calls.append(True)

    with patch("tools.cronjob_tools.new_fire_claim_owner", return_value=token), \
         patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
         patch("tools.cronjob_tools.get_job", return_value=None), \
         patch("cron.scheduler.run_one_job", side_effect=_run), \
         patch("tools.cronjob_tools.mark_job_run", side_effect=_mark):
        res = _dispatch_job_now(dict(_JOB, id="job-dispatch-missing"))

    assert res["claimed"] is True
    assert res["dispatched"] is False
    assert res["error"]
    assert "fire_claim" in res["error"].lower() or "token" in res["error"].lower() or "missing" in res["error"].lower()
    assert run_calls == []
    assert mark_calls == []
    with _manual_runs_lock:
        assert "job-dispatch-missing" not in _manual_run_threads


def test_execute_job_now_post_claim_malformed_fire_claim_fail_closed():
    """Post-claim record without a non-empty fire_claim owner is fail-closed."""
    token = "sync-owner:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    malformed = {
        **_JOB,
        "id": "job-malformed-claim",
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": ""},
    }
    run_calls = []

    with patch("tools.cronjob_tools.new_fire_claim_owner", return_value=token), \
         patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
         patch("tools.cronjob_tools.get_job", return_value=malformed), \
         patch("cron.scheduler.run_one_job", side_effect=lambda j: run_calls.append(j)):
        res = _execute_job_now(dict(_JOB, id="job-malformed-claim"))

    assert res == {
        "claimed": True,
        "success": False,
        "error": res["error"],
    }
    assert res["success"] is False
    assert res["error"]
    assert run_calls == []


def test_execute_job_now_preflight_abort_not_reported_success_from_stale_ok():
    """run_one_job False (pre-execution abort) must not report success via last_status.

    A prior stored last_status='ok' plus processed=True would look successful
    even though no side effect ran. Abort returns False so success stays False.
    """
    token = "sync-owner:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    claimed = {
        **_JOB,
        "id": "job-stale-ok",
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": token},
        "last_status": "ok",
        "last_error": None,
    }
    after = {
        **claimed,
        # Replacement/no-op path left prior ok status untouched.
        "last_status": "ok",
        "last_error": None,
    }

    with patch("tools.cronjob_tools.new_fire_claim_owner", return_value=token), \
         patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
         patch("tools.cronjob_tools.get_job", side_effect=[claimed, after]), \
         patch("cron.scheduler.run_one_job", return_value=False):
        res = _execute_job_now(dict(_JOB, id="job-stale-ok"))

    assert res["claimed"] is True
    assert res["success"] is False
    assert res.get("error") is None or res.get("error") == after.get("last_error")


def test_execute_job_now_post_claim_wrong_token_fail_closed():
    """Claim wins for token A but store re-read shows token B => no run_one_job."""
    token_a = "gw:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    token_b = "gw:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    claimed_wrong = {
        **_JOB,
        "id": "job-wrong-token-sync",
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": token_b},
    }
    run_calls = []
    mark_calls = []

    def _run(job):
        run_calls.append(job)
        return True

    def _mark(*_a, **_k):
        mark_calls.append(True)

    with patch("tools.cronjob_tools.new_fire_claim_owner", return_value=token_a), \
         patch("tools.cronjob_tools.claim_job_for_fire", return_value=True) as m_claim, \
         patch("tools.cronjob_tools.get_job", return_value=claimed_wrong), \
         patch("cron.scheduler.run_one_job", side_effect=_run), \
         patch("tools.cronjob_tools.mark_job_run", side_effect=_mark):
        res = _execute_job_now(dict(_JOB, id="job-wrong-token-sync"))

    assert res["claimed"] is True
    assert res["success"] is False
    assert res["error"]
    err = res["error"].lower()
    assert "token" in err or "owner" in err or "fire_claim" in err
    assert run_calls == []
    # Leave durable claim for TTL recovery — do not clear/mutate as A.
    assert mark_calls == []
    assert m_claim.call_args.kwargs.get("claim_owner") == token_a


def test_dispatch_job_now_post_claim_wrong_token_fail_closed():
    """Claim wins for token A but store re-read shows token B => no worker."""
    token_a = "gw:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    token_b = "gw:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    claimed_wrong = {
        **_JOB,
        "id": "job-wrong-token-dispatch",
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": token_b},
    }
    run_calls = []
    mark_calls = []

    def _run(job):
        run_calls.append(job)
        return True

    def _mark(*_a, **_k):
        mark_calls.append(True)

    with patch("tools.cronjob_tools.new_fire_claim_owner", return_value=token_a), \
         patch("tools.cronjob_tools.claim_job_for_fire", return_value=True) as m_claim, \
         patch("tools.cronjob_tools.get_job", return_value=claimed_wrong), \
         patch("cron.scheduler.run_one_job", side_effect=_run), \
         patch("tools.cronjob_tools.mark_job_run", side_effect=_mark):
        res = _dispatch_job_now(dict(_JOB, id="job-wrong-token-dispatch"))

    assert res["claimed"] is True
    assert res["dispatched"] is False
    assert res["error"]
    err = res["error"].lower()
    assert "token" in err or "owner" in err or "fire_claim" in err
    assert run_calls == []
    assert mark_calls == []
    assert m_claim.call_args.kwargs.get("claim_owner") == token_a
    with _manual_runs_lock:
        assert "job-wrong-token-dispatch" not in _manual_run_threads
