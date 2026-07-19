"""Regression coverage for long-lived manual/external fire_claim singleflight.

A manual or external fire stamps a durable ``fire_claim`` (TTL 300s) and then
runs outside ``scheduler._running_job_ids``. Without a heartbeat that keeps the
claim fresh — and without ``get_due_jobs`` treating a fresh claim as non-due —
a scheduled tick can re-dispatch the same recurring job after the TTL expires
while the first run is still alive.
"""

from datetime import datetime, timedelta
import threading
from unittest.mock import MagicMock, patch

import pytest


FIRE_CLAIM_TTL_SECONDS = 300


def test_long_running_manual_fire_keeps_job_non_due_past_ttl(tmp_path, monkeypatch):
    """A claimed recurring fire stays non-due past the original 300s TTL.

    While the run is still blocked, simulated wall time advances past the
    original claim TTL. The refreshed claim must keep the job out of the due
    set so the scheduled ticker cannot launch a concurrent copy.
    """
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    default_cron = tmp_path / "default" / "cron"
    default_cron.mkdir(parents=True)
    profile_home.mkdir()

    # If ContextVars are not propagated to the heartbeat thread, writes would
    # land here instead of the active profile store.
    monkeypatch.setattr(jobs, "CRON_DIR", default_cron)
    monkeypatch.setattr(jobs, "JOBS_FILE", default_cron / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", default_cron / "output")
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)

    original_timestamp = "2026-07-12T12:00:00+00:00"
    original_time = datetime.fromisoformat(original_timestamp)
    # Start near the end of the original claim TTL so a refresh is required to
    # keep the job non-due once wall time advances past the original stamp.
    current_time = [original_time + timedelta(seconds=FIRE_CLAIM_TTL_SECONDS - 60)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: current_time[0])

    def _job() -> dict:
        return {
            "id": "manual-long",
            "name": "manual long",
            "prompt": "long running work",
            "script": "work.py",
            "no_agent": True,
            "schedule": {"kind": "interval", "minutes": 5},
            # Due relative to the advanced clock without a fresh fire_claim.
            "next_run_at": original_timestamp,
            "enabled": True,
            "fire_claim": {
                "at": original_timestamp,
                "by": "manual-owner",
            },
        }

    jobs.save_jobs([_job()])
    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([_job()])
        claimed_job = jobs.get_job("manual-long")

    heartbeat_seen = threading.Event()
    real_heartbeat = jobs.heartbeat_fire_claim
    second_scheduler_scan = {}

    def _observed_heartbeat(job_id: str, *, expected_owner: str) -> bool:
        updated = real_heartbeat(job_id, expected_owner=expected_owner)
        # A different scheduler scans after the ORIGINAL claim's TTL while the
        # manual run is still blocked. The refreshed claim must keep the job
        # out of the due set and preserve ownership on the profile store.
        current_time[0] = original_time + timedelta(seconds=FIRE_CLAIM_TTL_SECONDS + 10)
        second_scheduler_scan["due"] = jobs.get_due_jobs()
        second_scheduler_scan["record_present"] = jobs.get_job(job_id) is not None
        second_scheduler_scan["claim"] = (jobs.get_job(job_id) or {}).get("fire_claim")
        second_scheduler_scan["reclaim"] = jobs.claim_job_for_fire(
            job_id, claim_ttl_seconds=FIRE_CLAIM_TTL_SECONDS
        )
        heartbeat_seen.set()
        return updated

    def _blocking_script(_script_path: str, workdir=None) -> tuple[bool, str]:
        assert heartbeat_seen.wait(timeout=2), (
            "fire_claim was not refreshed while the manual/external run blocked"
        )
        return True, "manual run complete"

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _observed_heartbeat)
    monkeypatch.setattr(scheduler, "_run_job_script", _blocking_script)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **k: None)

    with (
        jobs.use_cron_store(profile_home),
        patch("hermes_state.SessionDB", return_value=MagicMock()),
    ):
        # Heartbeat ownership lives on the full run_one_job lifecycle.
        ok = scheduler.run_one_job(claimed_job)
        # mark_job_run clears the claim once execute→save→deliver completes.
        profile_claim = jobs.get_job("manual-long")["fire_claim"]

    assert ok is True
    assert second_scheduler_scan["due"] == []
    assert second_scheduler_scan["record_present"] is True
    assert second_scheduler_scan["reclaim"] is False
    mid_claim = second_scheduler_scan["claim"]
    assert isinstance(mid_claim, dict)
    assert mid_claim["by"] == "manual-owner"
    assert mid_claim["at"] != original_timestamp
    assert profile_claim is None
    # Fallback store must not have been heartbeated (profile isolation).
    assert jobs.get_job("manual-long")["fire_claim"] == {
        "at": original_timestamp,
        "by": "manual-owner",
    }


def test_stale_owner_preflight_aborts_before_any_side_effect(tmp_path, monkeypatch):
    """Dispatched owner A must fail closed when durable store already has owner B.

    Synchronous ownership preflight via heartbeat_fire_claim(expected_owner=A)
    must see False and abort before script/run_job/delivery/teardown. The
    replacement owner's record remains byte-for-byte unchanged.
    """
    import copy
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    original_timestamp = "2026-07-12T12:00:00+00:00"
    replacement_timestamp = "2026-07-12T12:00:30+00:00"
    next_run = "2026-07-12T12:10:00+00:00"
    replacement_record = {
        "id": "stale-preflight",
        "name": "stale preflight",
        "prompt": "x",
        "script": "work.py",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": next_run,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": 3, "completed": 1},
        "last_status": None,
        "last_error": None,
        "last_run_at": None,
        "fire_claim": {
            "at": replacement_timestamp,
            "by": "replacement-owner",
        },
    }
    stale_dispatched = {
        **replacement_record,
        "fire_claim": {"at": original_timestamp, "by": "original-owner"},
        "next_run_at": original_timestamp,
    }

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([copy.deepcopy(replacement_record)])
        before = copy.deepcopy(jobs.get_job("stale-preflight"))

    side_effects = {"script": 0, "run_job": 0, "deliver": 0, "heartbeat": []}
    real_heartbeat = jobs.heartbeat_fire_claim
    real_run_job = scheduler.run_job

    def _observed_heartbeat(job_id: str, *, expected_owner: str) -> bool:
        updated = real_heartbeat(job_id, expected_owner=expected_owner)
        side_effects["heartbeat"].append(
            {"expected_owner": expected_owner, "updated": updated}
        )
        return updated

    def _script(*_a, **_k):
        side_effects["script"] += 1
        return True, "must not run"

    def _counting_run_job(*a, **k):
        side_effects["run_job"] += 1
        return real_run_job(*a, **k)

    def _deliver(*_a, **_k):
        side_effects["deliver"] += 1
        return None

    # Keep the first async heartbeat delay large so a missing preflight would
    # still execute side effects before any loop heartbeat — proving preflight
    # is synchronous and not delayed by the 60s wait.
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 60.0)
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _observed_heartbeat)
    monkeypatch.setattr(scheduler, "_run_job_script", _script)
    monkeypatch.setattr(scheduler, "run_job", _counting_run_job)
    monkeypatch.setattr(scheduler, "_deliver_result", _deliver)

    with jobs.use_cron_store(profile_home):
        ok = scheduler.run_one_job(stale_dispatched)
        after = jobs.get_job("stale-preflight")

    assert ok is False
    assert side_effects["script"] == 0
    assert side_effects["run_job"] == 0
    assert side_effects["deliver"] == 0
    assert side_effects["heartbeat"]
    assert all(
        h["expected_owner"] == "original-owner" and h["updated"] is False
        for h in side_effects["heartbeat"]
    )
    assert after == before


def test_stale_run_completion_does_not_mutate_replacement_owner_record(
    tmp_path, monkeypatch
):
    """Stale owner must abort preflight — no script, no mutation of owner B.

    Stored fire_claim.by is replacement-owner while the dispatched job still
    carries original-owner. Synchronous preflight must refuse execution; the
    replacement record remains byte-for-byte unchanged.
    """
    import copy
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    original_timestamp = "2026-07-12T12:00:00+00:00"
    replacement_timestamp = "2026-07-12T12:00:30+00:00"
    next_run = "2026-07-12T12:10:00+00:00"
    replacement_record = {
        "id": "stale-finish",
        "name": "stale finish",
        "prompt": "x",
        "script": "work.py",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": next_run,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": 3, "completed": 1},
        "last_status": None,
        "last_error": None,
        "last_run_at": None,
        "fire_claim": {
            "at": replacement_timestamp,
            "by": "replacement-owner",
        },
    }
    stale_dispatched = {
        **replacement_record,
        "fire_claim": {"at": original_timestamp, "by": "original-owner"},
        "next_run_at": original_timestamp,
    }

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([copy.deepcopy(replacement_record)])
        before = copy.deepcopy(jobs.get_job("stale-finish"))

    side_effects = {"script": 0}

    def _script(*_a, **_k):
        side_effects["script"] += 1
        return True, "stale done"

    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 60.0)
    monkeypatch.setattr(scheduler, "_run_job_script", _script)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **k: None)

    with jobs.use_cron_store(profile_home):
        ok = scheduler.run_one_job(stale_dispatched)
        after = jobs.get_job("stale-finish")

    assert ok is False
    assert side_effects["script"] == 0
    assert after is not None
    assert after == before
    assert after["fire_claim"] == {
        "at": replacement_timestamp,
        "by": "replacement-owner",
    }
    assert after["next_run_at"] == before["next_run_at"] == next_run
    assert after.get("last_status") == before.get("last_status")
    assert after.get("last_error") == before.get("last_error")
    assert after.get("last_run_at") == before.get("last_run_at")
    assert after.get("repeat") == before.get("repeat") == {
        "times": 3,
        "completed": 1,
    }
    assert after.get("enabled") is True
    assert after.get("state") == "scheduled"


def test_fire_claim_preflight_exception_fail_closed_no_side_effect(
    tmp_path, monkeypatch
):
    """heartbeat_fire_claim raising in preflight must fail closed before fn.

    No thread start, no script/agent side effects, return False. When this
    runner still owns the durable claim, owner-conditional mark may record the
    failure; otherwise durable state is left alone.
    """
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    claim_at = "2026-07-12T12:00:00+00:00"
    job = {
        "id": "hb-preflight-exc",
        "name": "hb preflight exc",
        "prompt": "must not run",
        "script": "work.py",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": "2026-07-12T12:05:00+00:00",
        "enabled": True,
        "fire_claim": {"at": claim_at, "by": "manual-owner"},
    }

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([job])
        claimed_job = jobs.get_job("hb-preflight-exc")

    side_effects = {"script": 0, "run_job": 0, "thread_start": 0}
    real_run_job = scheduler.run_job
    real_thread_start = threading.Thread.start

    def _boom_heartbeat(job_id: str, *, expected_owner: str) -> bool:
        raise RuntimeError("durable store unavailable for preflight")

    def _script(*_a, **_k):
        side_effects["script"] += 1
        return True, "should not run"

    def _counting_run_job(*a, **k):
        side_effects["run_job"] += 1
        return real_run_job(*a, **k)

    def _counting_start(self):
        side_effects["thread_start"] += 1
        return real_thread_start(self)

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _boom_heartbeat)
    monkeypatch.setattr(threading.Thread, "start", _counting_start)
    monkeypatch.setattr(scheduler, "_run_job_script", _script)
    monkeypatch.setattr(scheduler, "run_job", _counting_run_job)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **k: None)

    with jobs.use_cron_store(profile_home):
        ok = scheduler.run_one_job(claimed_job)
        finished = jobs.get_job("hb-preflight-exc")

    assert ok is False
    assert side_effects["script"] == 0
    assert side_effects["run_job"] == 0
    assert side_effects["thread_start"] == 0
    assert finished is not None
    # Still owns the claim → owner-conditional failure finalization is safe.
    assert finished.get("fire_claim") is None
    assert finished.get("last_status") == "error"
    assert finished.get("last_error")


def test_fire_claim_preflight_exception_does_not_mutate_replacement(
    tmp_path, monkeypatch
):
    """Preflight exception must not mutate a replacement owner's durable claim."""
    import copy
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    replacement_timestamp = "2026-07-12T12:00:30+00:00"
    next_run = "2026-07-12T12:10:00+00:00"
    replacement_record = {
        "id": "hb-preflight-replace",
        "name": "hb preflight replace",
        "prompt": "must not run",
        "script": "work.py",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": next_run,
        "enabled": True,
        "last_status": None,
        "last_error": None,
        "last_run_at": None,
        "fire_claim": {
            "at": replacement_timestamp,
            "by": "replacement-owner",
        },
    }
    stale_dispatched = {
        **replacement_record,
        "fire_claim": {
            "at": "2026-07-12T12:00:00+00:00",
            "by": "original-owner",
        },
    }

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([copy.deepcopy(replacement_record)])
        before = copy.deepcopy(jobs.get_job("hb-preflight-replace"))

    side_effects = {"script": 0, "thread_start": 0}
    real_thread_start = threading.Thread.start

    def _boom_heartbeat(job_id: str, *, expected_owner: str) -> bool:
        raise RuntimeError("durable store unavailable for preflight")

    def _script(*_a, **_k):
        side_effects["script"] += 1
        return True, "should not run"

    def _counting_start(self):
        side_effects["thread_start"] += 1
        return real_thread_start(self)

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _boom_heartbeat)
    monkeypatch.setattr(threading.Thread, "start", _counting_start)
    monkeypatch.setattr(scheduler, "_run_job_script", _script)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **k: None)

    with jobs.use_cron_store(profile_home):
        ok = scheduler.run_one_job(stale_dispatched)
        after = jobs.get_job("hb-preflight-replace")

    assert ok is False
    assert side_effects["script"] == 0
    assert side_effects["thread_start"] == 0
    assert after == before


def test_fire_claim_heartbeat_start_failure_fail_closed_no_side_effect(
    tmp_path, monkeypatch
):
    """If the fire_claim heartbeat thread cannot start, refuse to run unprotected.

    Thread.start() raising must fail closed before any job side effect. The
    shared run_one_job path returns a normal failed result, owner-conditionally
    finalizes only its own claim, and never clears a replacement owner's claim.
    """
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    claim_at = "2026-07-12T12:00:00+00:00"
    job = {
        "id": "hb-start-fail",
        "name": "hb start fail",
        "prompt": "must not run",
        "script": "work.py",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": "2026-07-12T12:05:00+00:00",
        "enabled": True,
        "fire_claim": {"at": claim_at, "by": "manual-owner"},
    }

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([job])
        claimed_job = jobs.get_job("hb-start-fail")

    side_effects = {"script": 0, "run_job": 0}

    def _boom_start(self):
        raise RuntimeError("thread start refused")

    def _script(*_a, **_k):
        side_effects["script"] += 1
        return True, "should not run"

    real_run_job = scheduler.run_job

    def _counting_run_job(*a, **k):
        side_effects["run_job"] += 1
        return real_run_job(*a, **k)

    monkeypatch.setattr(threading.Thread, "start", _boom_start)
    monkeypatch.setattr(scheduler, "_run_job_script", _script)
    monkeypatch.setattr(scheduler, "run_job", _counting_run_job)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **k: None)

    with jobs.use_cron_store(profile_home):
        ok = scheduler.run_one_job(claimed_job)
        finished = jobs.get_job("hb-start-fail")

    assert ok is False  # pre-execution abort is not a successful process
    assert side_effects["script"] == 0
    assert side_effects["run_job"] == 0
    assert finished is not None
    # Still owns the claim → owner-conditional failure finalization clears it
    # and records the owned failure.
    assert finished.get("fire_claim") is None
    assert finished.get("last_status") == "error"
    assert finished.get("last_error")


def test_fire_claim_heartbeat_start_failure_does_not_touch_replacement(
    tmp_path, monkeypatch
):
    """Heartbeat start failure must not finalize a replacement owner's claim.

    Preflight ownership for original-owner already fails closed before Thread
    start when the store holds replacement-owner. Either path must return
    False with zero side effects and leave the replacement record intact.
    """
    import copy
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    replacement_timestamp = "2026-07-12T12:00:30+00:00"
    next_run = "2026-07-12T12:10:00+00:00"
    replacement_record = {
        "id": "hb-start-replace",
        "name": "hb start replace",
        "prompt": "must not run",
        "script": "work.py",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": next_run,
        "enabled": True,
        "last_status": None,
        "last_error": None,
        "last_run_at": None,
        "fire_claim": {
            "at": replacement_timestamp,
            "by": "replacement-owner",
        },
    }
    stale_dispatched = {
        **replacement_record,
        "fire_claim": {
            "at": "2026-07-12T12:00:00+00:00",
            "by": "original-owner",
        },
    }

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([copy.deepcopy(replacement_record)])
        before = copy.deepcopy(jobs.get_job("hb-start-replace"))

    side_effects = {"script": 0}

    def _boom_start(self):
        raise RuntimeError("thread start refused")

    def _script(*_a, **_k):
        side_effects["script"] += 1
        return True, "should not run"

    monkeypatch.setattr(threading.Thread, "start", _boom_start)
    monkeypatch.setattr(scheduler, "_run_job_script", _script)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **k: None)

    with jobs.use_cron_store(profile_home):
        ok = scheduler.run_one_job(stale_dispatched)
        after = jobs.get_job("hb-start-replace")

    assert ok is False
    assert side_effects["script"] == 0
    assert after == before
    assert after["fire_claim"] == {
        "at": replacement_timestamp,
        "by": "replacement-owner",
    }
    assert after.get("next_run_at") == next_run
    assert after.get("last_status") is None
    assert after.get("last_error") is None
    assert after.get("last_run_at") is None


def test_fire_claim_start_failure_after_owned_preflight_returns_false(
    tmp_path, monkeypatch
):
    """Owned preflight + Thread.start failure still returns False, not success."""
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    claim_at = "2026-07-12T12:00:00+00:00"
    job = {
        "id": "hb-start-owned",
        "name": "hb start owned",
        "prompt": "must not run",
        "script": "work.py",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": "2026-07-12T12:05:00+00:00",
        "enabled": True,
        # Prior status must not be misreported as this run's success.
        "last_status": "ok",
        "last_error": None,
        "fire_claim": {"at": claim_at, "by": "manual-owner"},
    }

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([job])
        claimed_job = jobs.get_job("hb-start-owned")

    side_effects = {"script": 0}

    def _boom_start(self):
        raise RuntimeError("thread start refused")

    def _script(*_a, **_k):
        side_effects["script"] += 1
        return True, "should not run"

    monkeypatch.setattr(threading.Thread, "start", _boom_start)
    monkeypatch.setattr(scheduler, "_run_job_script", _script)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **k: None)

    with jobs.use_cron_store(profile_home):
        ok = scheduler.run_one_job(claimed_job)
        finished = jobs.get_job("hb-start-owned")

    assert ok is False
    assert side_effects["script"] == 0
    assert finished is not None
    assert finished.get("fire_claim") is None
    assert finished.get("last_status") == "error"


def test_fire_claim_heartbeat_covers_blocked_delivery_past_ttl(tmp_path, monkeypatch):
    """Heartbeat must cover post-run delivery until mark_job_run clears the claim.

    ``run_job`` can finish quickly while ``_deliver_result`` is still blocked.
    Simulated wall time then advances past the original 300s claim TTL. Without
    a lifecycle-wide heartbeat the claim looks dead and ``get_due_jobs`` can
    re-dispatch a concurrent copy of the same fire.
    """
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)

    original_timestamp = "2026-07-12T12:00:00+00:00"
    original_time = datetime.fromisoformat(original_timestamp)
    # Start near the end of the original claim TTL so a post-run refresh is
    # required once wall time advances past the original stamp.
    current_time = [original_time + timedelta(seconds=FIRE_CLAIM_TTL_SECONDS - 60)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: current_time[0])

    job = {
        "id": "delivery-gap",
        "name": "delivery gap",
        "prompt": "deliver me",
        "script": "work.py",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 5},
        # Due relative to the advanced clock without a fresh fire_claim.
        "next_run_at": original_timestamp,
        "enabled": True,
        "fire_claim": {
            "at": original_timestamp,
            "by": "manual-owner",
        },
    }

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([job])
        claimed_job = jobs.get_job("delivery-gap")

    run_job_finished = threading.Event()
    delivery_entered = threading.Event()
    release_delivery = threading.Event()
    heartbeat_during_delivery = threading.Event()
    mid_scan = {}

    real_run_job = scheduler.run_job

    def _tracking_run_job(job_arg, *, defer_agent_teardown=None):
        result = real_run_job(job_arg, defer_agent_teardown=defer_agent_teardown)
        run_job_finished.set()
        return result

    real_heartbeat = jobs.heartbeat_fire_claim

    def _observed_heartbeat(job_id: str, *, expected_owner: str) -> bool:
        updated = real_heartbeat(job_id, expected_owner=expected_owner)
        # Only assert singleflight after run_job returned and delivery is the
        # phase still holding the fire_claim open.
        if run_job_finished.is_set() and delivery_entered.is_set():
            current_time[0] = original_time + timedelta(
                seconds=FIRE_CLAIM_TTL_SECONDS + 10
            )
            mid_scan["due"] = jobs.get_due_jobs()
            mid_scan["claim"] = (jobs.get_job(job_id) or {}).get("fire_claim")
            mid_scan["refreshed"] = updated
            heartbeat_during_delivery.set()
        return updated

    def _blocking_deliver(job_arg, content, adapters=None, loop=None):
        delivery_entered.set()
        assert run_job_finished.wait(timeout=2), "run_job never finished before delivery"
        assert heartbeat_during_delivery.wait(timeout=2), (
            "fire_claim was not refreshed while post-run delivery blocked"
        )
        assert release_delivery.wait(timeout=2), "delivery was never released"
        return None

    monkeypatch.setattr(scheduler, "run_job", _tracking_run_job)
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _observed_heartbeat)
    monkeypatch.setattr(scheduler, "_deliver_result", _blocking_deliver)
    monkeypatch.setattr(
        scheduler, "_run_job_script", lambda *_a, **_k: (True, "delivery payload")
    )

    def _runner():
        with (
            jobs.use_cron_store(profile_home),
            patch("hermes_state.SessionDB", return_value=MagicMock()),
        ):
            return scheduler.run_one_job(claimed_job)

    worker = threading.Thread(target=lambda: mid_scan.setdefault("ok", _runner()))
    worker.start()

    assert delivery_entered.wait(timeout=2), "delivery phase never started"
    assert run_job_finished.is_set()
    # Heartbeat must keep the claim fresh while delivery is still blocked.
    assert heartbeat_during_delivery.wait(timeout=2)
    assert mid_scan["due"] == []
    assert mid_scan["refreshed"] is True
    mid_claim = mid_scan["claim"]
    assert isinstance(mid_claim, dict)
    assert mid_claim["by"] == "manual-owner"
    assert mid_claim["at"] != original_timestamp

    release_delivery.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert mid_scan.get("ok") is True

    with jobs.use_cron_store(profile_home):
        finished = jobs.get_job("delivery-gap")
        assert finished is not None
        # mark_job_run clears the durable claim once the full lifecycle ends.
        assert finished.get("fire_claim") is None
        assert finished.get("last_status") == "ok"


def test_get_due_jobs_skips_fresh_fire_claim(tmp_path, monkeypatch):
    """Scheduled ticker must not treat a freshly claimed fire as due.

    Fresh claims remain untouched in both the returned scan and durable store.
    """
    import cron.jobs as jobs

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    now = datetime.fromisoformat("2026-07-12T12:00:00+00:00")
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)

    fresh_claim = {
        "at": "2026-07-12T11:59:30+00:00",
        "by": "owner",
    }
    with jobs.use_cron_store(profile_home):
        jobs.save_jobs(
            [
                {
                    "id": "claimed-due",
                    "name": "claimed due",
                    "prompt": "x",
                    "schedule": {"kind": "interval", "minutes": 5},
                    "next_run_at": "2026-07-12T11:59:00+00:00",
                    "enabled": True,
                    "fire_claim": dict(fresh_claim),
                }
            ]
        )
        assert jobs.get_due_jobs() == []
        stored = jobs.get_job("claimed-due")
        assert stored is not None
        assert stored.get("fire_claim") == fresh_claim


def test_get_due_jobs_allows_expired_fire_claim(tmp_path, monkeypatch):
    """Expired fire_claim recovery: due once, claim cleared before dispatch.

    Returning the job *with* the old token would let the built-in ticker
    capture and heartbeat that stale generation, defeating the fence.
    """
    import cron.jobs as jobs

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    now = datetime.fromisoformat("2026-07-12T12:10:00+00:00")
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs(
            [
                {
                    "id": "stale-claim",
                    "name": "stale claim",
                    "prompt": "x",
                    "schedule": {"kind": "interval", "minutes": 5},
                    "next_run_at": "2026-07-12T12:00:00+00:00",
                    "enabled": True,
                    "fire_claim": {
                        "at": "2026-07-12T12:00:00+00:00",
                        "by": "dead-owner",
                    },
                }
            ]
        )
        due = jobs.get_due_jobs()
        assert len(due) == 1
        assert due[0]["id"] == "stale-claim"
        # Returned/dispatched job must not carry the expired generation token.
        assert due[0].get("fire_claim") in (None, {})
        # Durable store must clear the claim before get_due_jobs returns.
        stored = jobs.get_job("stale-claim")
        assert stored is not None
        assert stored.get("fire_claim") in (None, {})


def test_get_due_jobs_clears_future_dated_fire_claim(tmp_path, monkeypatch):
    """Future-dated fire_claim is not eternally fresh — clear before due return."""
    import cron.jobs as jobs

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    now = datetime.fromisoformat("2026-07-12T12:00:00+00:00")
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs(
            [
                {
                    "id": "future-claim",
                    "name": "future claim",
                    "prompt": "x",
                    "schedule": {"kind": "interval", "minutes": 5},
                    "next_run_at": "2026-07-12T11:59:00+00:00",
                    "enabled": True,
                    "fire_claim": {
                        "at": "2026-07-12T18:00:00+00:00",
                        "by": "skewed-owner",
                    },
                }
            ]
        )
        due = jobs.get_due_jobs()
        assert len(due) == 1
        assert due[0]["id"] == "future-claim"
        assert due[0].get("fire_claim") in (None, {})
        stored = jobs.get_job("future-claim")
        assert stored is not None
        assert stored.get("fire_claim") in (None, {})


def test_get_due_jobs_clears_malformed_fire_claim(tmp_path, monkeypatch):
    """Malformed fire_claim must not wedge forever or be adopted on dispatch."""
    import cron.jobs as jobs

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    now = datetime.fromisoformat("2026-07-12T12:00:00+00:00")
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs(
            [
                {
                    "id": "bad-claim",
                    "name": "bad claim",
                    "prompt": "x",
                    "schedule": {"kind": "interval", "minutes": 5},
                    "next_run_at": "2026-07-12T11:59:00+00:00",
                    "enabled": True,
                    "fire_claim": {"at": "not-a-timestamp", "by": "broken"},
                }
            ]
        )
        due = jobs.get_due_jobs()
        assert len(due) == 1
        assert due[0]["id"] == "bad-claim"
        assert due[0].get("fire_claim") in (None, {})
        stored = jobs.get_job("bad-claim")
        assert stored is not None
        assert stored.get("fire_claim") in (None, {})


def test_stale_runner_after_due_scan_recovery_cannot_adopt_cleared_claim(
    tmp_path, monkeypatch
):
    """After due-scan clears expired token A, a stale runner still holding A aborts.

    Durable store has claim cleared by due scan. The old dispatched job still
    carries token A; synchronous preflight must abort before side effects and
    owner-conditional mark is a no-op. The scheduled due record has no token
    and can run via the normal ticker path.
    """
    import copy
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    token_a = "old-runner:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    original_timestamp = "2026-07-12T12:00:00+00:00"
    now = datetime.fromisoformat("2026-07-12T12:10:00+00:00")
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)

    base_job = {
        "id": "recover-then-stale",
        "name": "recover then stale",
        "prompt": "x",
        "script": "work.py",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": original_timestamp,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": 3, "completed": 0},
        "last_status": None,
        "last_error": None,
        "last_run_at": None,
        "fire_claim": {
            "at": original_timestamp,
            "by": token_a,
        },
    }

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([copy.deepcopy(base_job)])
        due = jobs.get_due_jobs()
        assert len(due) == 1
        assert due[0].get("fire_claim") in (None, {})
        stored_after_scan = jobs.get_job("recover-then-stale")
        assert stored_after_scan is not None
        assert stored_after_scan.get("fire_claim") in (None, {})

    # Stale runner still holds the pre-scan dispatched snapshot with token A.
    stale_dispatched = {
        **base_job,
        "fire_claim": {"at": original_timestamp, "by": token_a},
    }

    side_effects = {"script": 0, "run_job": 0, "deliver": 0, "heartbeat": []}
    real_heartbeat = jobs.heartbeat_fire_claim
    real_run_job = scheduler.run_job

    def _observed_heartbeat(job_id: str, *, expected_owner: str) -> bool:
        updated = real_heartbeat(job_id, expected_owner=expected_owner)
        side_effects["heartbeat"].append(
            {"expected_owner": expected_owner, "updated": updated}
        )
        return updated

    def _script(*_a, **_k):
        side_effects["script"] += 1
        return True, "must not run on stale path"

    def _counting_run_job(*a, **k):
        side_effects["run_job"] += 1
        return real_run_job(*a, **k)

    def _deliver(*_a, **_k):
        side_effects["deliver"] += 1
        return None

    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 60.0)
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _observed_heartbeat)
    monkeypatch.setattr(scheduler, "_run_job_script", _script)
    monkeypatch.setattr(scheduler, "run_job", _counting_run_job)
    monkeypatch.setattr(scheduler, "_deliver_result", _deliver)

    with jobs.use_cron_store(profile_home):
        before_stale = copy.deepcopy(jobs.get_job("recover-then-stale"))
        ok_stale = scheduler.run_one_job(stale_dispatched)
        mark_applied = jobs.mark_job_run(
            "recover-then-stale",
            True,
            expected_fire_claim_owner=token_a,
        )
        after_stale = jobs.get_job("recover-then-stale")

    assert ok_stale is False
    assert mark_applied is False
    assert side_effects["script"] == 0
    assert side_effects["run_job"] == 0
    assert side_effects["deliver"] == 0
    assert side_effects["heartbeat"]
    assert all(
        h["expected_owner"] == token_a and h["updated"] is False
        for h in side_effects["heartbeat"]
    )
    assert after_stale == before_stale
    assert after_stale.get("fire_claim") in (None, {})

    # Cleared due record (no token) can run via the normal ticker path.
    ticker_job = copy.deepcopy(after_stale)
    assert ticker_job.get("fire_claim") in (None, {})

    side_effects_ok = {"script": 0}

    def _ok_script(*_a, **_k):
        side_effects_ok["script"] += 1
        return True, "ticker ok"

    monkeypatch.setattr(scheduler, "_run_job_script", _ok_script)
    # No fire_claim → no heartbeat preflight; run proceeds as scheduled path.
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", real_heartbeat)

    with jobs.use_cron_store(profile_home):
        ok_ticker = scheduler.run_one_job(ticker_job)
        finished = jobs.get_job("recover-then-stale")

    assert ok_ticker is True
    assert side_effects_ok["script"] == 1
    assert finished is not None
    assert finished.get("last_status") == "ok"
    assert finished.get("fire_claim") in (None, {})


def test_mark_job_run_matching_expected_fire_claim_owner_finalizes_and_clears(
    tmp_path,
):
    """Matching expected_fire_claim_owner may finalize the run and clear claim."""
    import cron.jobs as jobs

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    with jobs.use_cron_store(profile_home):
        jobs.save_jobs(
            [
                {
                    "id": "mark-match",
                    "name": "mark match",
                    "prompt": "x",
                    "schedule": {"kind": "interval", "minutes": 5},
                    "next_run_at": "2026-07-12T12:00:00+00:00",
                    "enabled": True,
                    "fire_claim": {
                        "at": "2026-07-12T12:00:00+00:00",
                        "by": "owner-a",
                    },
                }
            ]
        )
        applied = jobs.mark_job_run(
            "mark-match",
            True,
            expected_fire_claim_owner="owner-a",
        )
        record = jobs.get_job("mark-match")

    assert applied is True
    assert record is not None
    assert record.get("fire_claim") is None
    assert record.get("last_status") == "ok"
    assert record.get("last_run_at") is not None
    assert record.get("next_run_at") is not None
    assert record["next_run_at"] != "2026-07-12T12:00:00+00:00"


def test_mark_job_run_mismatched_expected_fire_claim_owner_is_full_noop(tmp_path):
    """Mismatched expected_fire_claim_owner is a full no-op and reports not-applied."""
    import copy
    import cron.jobs as jobs

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    stored = {
        "id": "mark-mismatch",
        "name": "mark mismatch",
        "prompt": "x",
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": "2026-07-12T12:10:00+00:00",
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": 5, "completed": 2},
        "last_status": "ok",
        "last_error": None,
        "last_run_at": "2026-07-12T11:55:00+00:00",
        "fire_claim": {
            "at": "2026-07-12T12:00:30+00:00",
            "by": "replacement-owner",
        },
    }
    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([copy.deepcopy(stored)])
        before = copy.deepcopy(jobs.get_job("mark-mismatch"))
        applied = jobs.mark_job_run(
            "mark-mismatch",
            True,
            expected_fire_claim_owner="original-owner",
        )
        after = jobs.get_job("mark-mismatch")

    assert applied is False
    assert after == before
    assert after["fire_claim"]["by"] == "replacement-owner"
    assert after["next_run_at"] == "2026-07-12T12:10:00+00:00"
    assert after["repeat"]["completed"] == 2
    assert after["last_status"] == "ok"
    assert after["last_run_at"] == "2026-07-12T11:55:00+00:00"


def test_mark_job_run_missing_claim_with_expected_owner_is_noop(tmp_path):
    """Expected owner with missing/absent fire_claim must not mutate the job."""
    import copy
    import cron.jobs as jobs

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    stored = {
        "id": "mark-missing-claim",
        "name": "mark missing",
        "prompt": "x",
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": "2026-07-12T12:10:00+00:00",
        "enabled": True,
        "last_status": None,
        "fire_claim": None,
    }
    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([copy.deepcopy(stored)])
        before = copy.deepcopy(jobs.get_job("mark-missing-claim"))
        applied = jobs.mark_job_run(
            "mark-missing-claim",
            False,
            "stale failure",
            expected_fire_claim_owner="original-owner",
        )
        after = jobs.get_job("mark-missing-claim")

    assert applied is False
    assert after == before


def test_mark_job_run_without_expected_owner_preserves_legacy_clear(tmp_path):
    """Legacy scheduled callers (no expected owner) still clear fire_claim."""
    import cron.jobs as jobs

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    with jobs.use_cron_store(profile_home):
        jobs.save_jobs(
            [
                {
                    "id": "mark-legacy",
                    "name": "mark legacy",
                    "prompt": "x",
                    "schedule": {"kind": "interval", "minutes": 5},
                    "next_run_at": "2026-07-12T12:00:00+00:00",
                    "enabled": True,
                    "fire_claim": {
                        "at": "2026-07-12T12:00:00+00:00",
                        "by": "anyone",
                    },
                }
            ]
        )
        # No expected_fire_claim_owner → legacy unconditional finalize+clear.
        result = jobs.mark_job_run("mark-legacy", True)
        record = jobs.get_job("mark-legacy")

    assert result is True or result is None  # applied; legacy may return None/True
    assert record is not None
    assert record.get("fire_claim") is None
    assert record.get("last_status") == "ok"


def test_same_machine_stale_generation_preflight_and_mark_fence(tmp_path, monkeypatch):
    """Same machine id, different generation tokens: stale run cannot touch B.

    Old dispatched job has generation A; durable replacement has generation B.
    Both share the same machine attribution prefix. Preflight must abort before
    side effects and owner-conditional mark must leave B byte-identical.
    """
    import copy
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    machine = "gw-host:4242"
    gen_a = f"{machine}:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    gen_b = f"{machine}:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    original_timestamp = "2026-07-12T12:00:00+00:00"
    replacement_timestamp = "2026-07-12T12:00:30+00:00"
    next_run = "2026-07-12T12:10:00+00:00"
    replacement_record = {
        "id": "same-machine-gen",
        "name": "same machine gen",
        "prompt": "x",
        "script": "work.py",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": next_run,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": 3, "completed": 1},
        "last_status": None,
        "last_error": None,
        "last_run_at": None,
        "fire_claim": {
            "at": replacement_timestamp,
            "by": gen_b,
        },
    }
    stale_dispatched = {
        **replacement_record,
        "fire_claim": {"at": original_timestamp, "by": gen_a},
        "next_run_at": original_timestamp,
    }

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([copy.deepcopy(replacement_record)])
        before = copy.deepcopy(jobs.get_job("same-machine-gen"))

    side_effects = {"script": 0, "run_job": 0, "deliver": 0, "heartbeat": []}
    real_heartbeat = jobs.heartbeat_fire_claim
    real_run_job = scheduler.run_job

    def _observed_heartbeat(job_id: str, *, expected_owner: str) -> bool:
        updated = real_heartbeat(job_id, expected_owner=expected_owner)
        side_effects["heartbeat"].append(
            {"expected_owner": expected_owner, "updated": updated}
        )
        return updated

    def _script(*_a, **_k):
        side_effects["script"] += 1
        return True, "must not run"

    def _counting_run_job(*a, **k):
        side_effects["run_job"] += 1
        return real_run_job(*a, **k)

    def _deliver(*_a, **_k):
        side_effects["deliver"] += 1
        return None

    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 60.0)
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _observed_heartbeat)
    monkeypatch.setattr(scheduler, "_run_job_script", _script)
    monkeypatch.setattr(scheduler, "run_job", _counting_run_job)
    monkeypatch.setattr(scheduler, "_deliver_result", _deliver)

    with jobs.use_cron_store(profile_home):
        ok = scheduler.run_one_job(stale_dispatched)
        # Stale completion mark must also be a full no-op against generation B.
        mark_applied = jobs.mark_job_run(
            "same-machine-gen",
            True,
            expected_fire_claim_owner=gen_a,
        )
        after = jobs.get_job("same-machine-gen")

    assert ok is False
    assert mark_applied is False
    assert side_effects["script"] == 0
    assert side_effects["run_job"] == 0
    assert side_effects["deliver"] == 0
    assert side_effects["heartbeat"]
    assert all(
        h["expected_owner"] == gen_a and h["updated"] is False
        for h in side_effects["heartbeat"]
    )
    assert after == before
    assert after["fire_claim"] == {
        "at": replacement_timestamp,
        "by": gen_b,
    }


def test_mid_run_heartbeat_false_aborts_before_delivery_and_preserves_replacement(
    tmp_path, monkeypatch
):
    """Preflight True, later heartbeat False: fail closed before delivery/mark.

    Synchronous ownership preflight succeeds for owner A. While ``run_job`` is
    blocked beyond a heartbeat interval, durable ownership moves to replacement
    owner B so subsequent heartbeats return False. Once ``run_job`` returns,
    runner A must detect lease loss BEFORE save/delivery, must not call
    ``_deliver_result``, and must leave B's claim/state untouched. ``run_one_job``
    reports failure/aborted (False).
    """
    import copy
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)

    original_timestamp = "2026-07-12T12:00:00+00:00"
    replacement_timestamp = "2026-07-12T12:00:30+00:00"
    next_run = "2026-07-12T12:10:00+00:00"
    owner_a = "runner-a:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    owner_b = "runner-b:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    base_job = {
        "id": "mid-run-false",
        "name": "mid run false",
        "prompt": "long work",
        "script": "work.py",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": next_run,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": 3, "completed": 1},
        "last_status": None,
        "last_error": None,
        "last_run_at": None,
        "fire_claim": {"at": original_timestamp, "by": owner_a},
    }

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([copy.deepcopy(base_job)])
        claimed_job = jobs.get_job("mid-run-false")

    run_entered = threading.Event()
    release_run = threading.Event()
    lost_seen = threading.Event()
    side_effects = {
        "deliver": 0,
        "heartbeat": [],
        "preflight_true": False,
        "mid_false": 0,
    }
    real_heartbeat = jobs.heartbeat_fire_claim
    real_run_job = scheduler.run_job
    lock = threading.Lock()

    def _observed_heartbeat(job_id: str, *, expected_owner: str) -> bool:
        updated = real_heartbeat(job_id, expected_owner=expected_owner)
        with lock:
            side_effects["heartbeat"].append(
                {"expected_owner": expected_owner, "updated": updated}
            )
            if updated:
                side_effects["preflight_true"] = True
            else:
                side_effects["mid_false"] += 1
                lost_seen.set()
        return updated

    def _blocking_run_job(job_arg, *, defer_agent_teardown=None):
        # Prove preflight already succeeded and the heartbeat thread is live
        # before ownership is stolen mid-run.
        assert side_effects["preflight_true"] is True
        run_entered.set()
        assert release_run.wait(timeout=3), "run_job was never released"
        return True, "# output\n", "must not deliver this", None

    def _deliver(*_a, **_k):
        side_effects["deliver"] += 1
        return None

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _observed_heartbeat)
    monkeypatch.setattr(scheduler, "run_job", _blocking_run_job)
    monkeypatch.setattr(scheduler, "_deliver_result", _deliver)
    monkeypatch.setattr(
        scheduler, "save_job_output", lambda *a, **k: str(tmp_path / "out.md")
    )

    result_holder = {}

    def _runner():
        with jobs.use_cron_store(profile_home):
            result_holder["ok"] = scheduler.run_one_job(claimed_job)

    worker = threading.Thread(target=_runner, name="mid-run-false-worker")
    worker.start()

    assert run_entered.wait(timeout=2), "run_job never entered after preflight"
    # Steal ownership while A is still blocked in run_job.
    with jobs.use_cron_store(profile_home):
        jobs.save_jobs(
            [
                {
                    **base_job,
                    "fire_claim": {"at": replacement_timestamp, "by": owner_b},
                    "next_run_at": next_run,
                    "last_status": None,
                    "last_error": None,
                    "last_run_at": None,
                    "repeat": {"times": 3, "completed": 1},
                }
            ]
        )
        replacement_before = copy.deepcopy(jobs.get_job("mid-run-false"))

    assert lost_seen.wait(timeout=2), (
        "heartbeat never observed False after replacement claim"
    )
    release_run.set()
    worker.join(timeout=5)
    assert not worker.is_alive()

    with jobs.use_cron_store(profile_home):
        after = jobs.get_job("mid-run-false")

    assert result_holder.get("ok") is False
    assert side_effects["preflight_true"] is True
    assert side_effects["mid_false"] >= 1
    assert side_effects["deliver"] == 0
    assert after == replacement_before
    assert after["fire_claim"] == {
        "at": replacement_timestamp,
        "by": owner_b,
    }
    assert after.get("last_status") is None
    assert after.get("last_error") is None
    assert after.get("last_run_at") is None
    assert after.get("repeat") == {"times": 3, "completed": 1}
    assert after.get("next_run_at") == next_run


def test_mid_run_heartbeat_exception_fail_closed_before_delivery(
    tmp_path, monkeypatch
):
    """Subsequent heartbeat raise: fail closed before delivery/mark, no leak.

    Preflight succeeds. A later heartbeat raises while ``run_job`` is blocked.
    Runner must abort remaining post-run side effects without leaking the
    exception, leave replacement owner B untouched, and return False.
    """
    import copy
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)

    original_timestamp = "2026-07-12T12:00:00+00:00"
    replacement_timestamp = "2026-07-12T12:00:30+00:00"
    next_run = "2026-07-12T12:10:00+00:00"
    owner_a = "runner-a:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    owner_b = "runner-b:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    base_job = {
        "id": "mid-run-exc",
        "name": "mid run exc",
        "prompt": "long work",
        "script": "work.py",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": next_run,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": 2, "completed": 0},
        "last_status": None,
        "last_error": None,
        "last_run_at": None,
        "fire_claim": {"at": original_timestamp, "by": owner_a},
    }

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([copy.deepcopy(base_job)])
        claimed_job = jobs.get_job("mid-run-exc")

    run_entered = threading.Event()
    release_run = threading.Event()
    boom_seen = threading.Event()
    side_effects = {"deliver": 0, "heartbeat_ok": 0, "heartbeat_exc": 0}
    real_heartbeat = jobs.heartbeat_fire_claim
    lock = threading.Lock()
    fail_after_preflight = {"armed": False}

    def _observed_heartbeat(job_id: str, *, expected_owner: str) -> bool:
        with lock:
            if fail_after_preflight["armed"]:
                side_effects["heartbeat_exc"] += 1
                boom_seen.set()
                raise RuntimeError("durable store unavailable mid-run")
        updated = real_heartbeat(job_id, expected_owner=expected_owner)
        with lock:
            if updated:
                side_effects["heartbeat_ok"] += 1
                # After preflight success, arm mid-run failure mode.
                fail_after_preflight["armed"] = True
        return updated

    def _blocking_run_job(job_arg, *, defer_agent_teardown=None):
        assert side_effects["heartbeat_ok"] >= 1
        run_entered.set()
        assert release_run.wait(timeout=3), "run_job was never released"
        return True, "# output\n", "must not deliver", None

    def _deliver(*_a, **_k):
        side_effects["deliver"] += 1
        return None

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _observed_heartbeat)
    monkeypatch.setattr(scheduler, "run_job", _blocking_run_job)
    monkeypatch.setattr(scheduler, "_deliver_result", _deliver)
    monkeypatch.setattr(
        scheduler, "save_job_output", lambda *a, **k: str(tmp_path / "out.md")
    )

    result_holder = {}

    def _runner():
        with jobs.use_cron_store(profile_home):
            try:
                result_holder["ok"] = scheduler.run_one_job(claimed_job)
            except Exception as exc:  # pragma: no cover - must not leak
                result_holder["leaked"] = exc
            result_holder["after"] = jobs.get_job("mid-run-exc")

    worker = threading.Thread(target=_runner, name="mid-run-exc-worker")
    worker.start()

    assert run_entered.wait(timeout=2), "run_job never entered after preflight"
    # Install replacement owner B while A is still blocked so a stale
    # fail-open mark would corrupt B if finalization were not owner-gated.
    with jobs.use_cron_store(profile_home):
        jobs.save_jobs(
            [
                {
                    **base_job,
                    "fire_claim": {
                        "at": replacement_timestamp,
                        "by": owner_b,
                    },
                }
            ]
        )
        result_holder["before"] = copy.deepcopy(jobs.get_job("mid-run-exc"))

    assert boom_seen.wait(timeout=2), "heartbeat never raised mid-run"
    release_run.set()
    worker.join(timeout=5)
    assert not worker.is_alive()

    assert "leaked" not in result_holder
    assert result_holder.get("ok") is False
    assert side_effects["heartbeat_ok"] >= 1
    assert side_effects["heartbeat_exc"] >= 1
    assert side_effects["deliver"] == 0
    assert result_holder["after"] == result_holder["before"]
    assert result_holder["after"]["fire_claim"] == {
        "at": replacement_timestamp,
        "by": owner_b,
    }
    assert result_holder["after"].get("last_status") is None


def test_post_run_ownership_checkpoint_before_delivery_boundary(
    tmp_path, monkeypatch
):
    """Lease may be lost while approaching save/delivery — explicit sync gate.

    After ``run_job`` returns, the runner must perform an explicit synchronous
    exact-owner refresh/check before delivery and before owner-conditional mark.
    If that refresh is False (or would raise), remaining post-run side effects
    abort: no platform delivery, no schedule/state mutation by the stale owner.
    """
    import copy
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    # Large interval so the background loop never fires during this short run;
    # loss is detected only by the post-run synchronous checkpoint.
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 60.0)

    original_timestamp = "2026-07-12T12:00:00+00:00"
    replacement_timestamp = "2026-07-12T12:00:45+00:00"
    next_run = "2026-07-12T12:15:00+00:00"
    owner_a = "runner-a:cccccccccccccccccccccccccccccccc"
    owner_b = "runner-b:dddddddddddddddddddddddddddddddd"

    base_job = {
        "id": "post-run-checkpoint",
        "name": "post run checkpoint",
        "prompt": "x",
        "script": "work.py",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": next_run,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": 4, "completed": 2},
        "last_status": None,
        "last_error": None,
        "last_run_at": None,
        "fire_claim": {"at": original_timestamp, "by": owner_a},
    }

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([copy.deepcopy(base_job)])
        claimed_job = jobs.get_job("post-run-checkpoint")

    side_effects = {
        "deliver": 0,
        "save": 0,
        "heartbeat": [],
        "run_job": 0,
    }
    real_heartbeat = jobs.heartbeat_fire_claim
    real_run_job = scheduler.run_job
    steal_after_run = {"done": False}

    def _observed_heartbeat(job_id: str, *, expected_owner: str) -> bool:
        updated = real_heartbeat(job_id, expected_owner=expected_owner)
        side_effects["heartbeat"].append(
            {"expected_owner": expected_owner, "updated": updated}
        )
        return updated

    def _run_then_steal(job_arg, *, defer_agent_teardown=None):
        side_effects["run_job"] += 1
        result = (
            True,
            "# diagnostic output\n",
            "payload that must not be delivered",
            None,
        )
        # Steal ownership after agent/script work finishes but before the
        # runner's post-run delivery/mark path runs.
        with jobs.use_cron_store(profile_home):
            jobs.save_jobs(
                [
                    {
                        **base_job,
                        "fire_claim": {
                            "at": replacement_timestamp,
                            "by": owner_b,
                        },
                    }
                ]
            )
            steal_after_run["before"] = copy.deepcopy(
                jobs.get_job("post-run-checkpoint")
            )
            steal_after_run["done"] = True
        return result

    def _deliver(*_a, **_k):
        side_effects["deliver"] += 1
        return None

    def _save(*_a, **_k):
        side_effects["save"] += 1
        return str(tmp_path / "diag.md")

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _observed_heartbeat)
    monkeypatch.setattr(scheduler, "run_job", _run_then_steal)
    monkeypatch.setattr(scheduler, "_deliver_result", _deliver)
    monkeypatch.setattr(scheduler, "save_job_output", _save)

    with jobs.use_cron_store(profile_home):
        ok = scheduler.run_one_job(claimed_job)
        after = jobs.get_job("post-run-checkpoint")

    assert steal_after_run["done"] is True
    assert side_effects["run_job"] == 1
    assert ok is False
    assert side_effects["deliver"] == 0
    # Optional diagnostic save may or may not run; delivery/mark must not.
    assert after == steal_after_run["before"]
    assert after["fire_claim"] == {
        "at": replacement_timestamp,
        "by": owner_b,
    }
    assert after.get("last_status") is None
    assert after.get("last_error") is None
    assert after.get("repeat") == {"times": 4, "completed": 2}
    # At least preflight True, then a post-run checkpoint False.
    assert any(h["updated"] is True for h in side_effects["heartbeat"])
    assert any(h["updated"] is False for h in side_effects["heartbeat"])
    assert all(h["expected_owner"] == owner_a for h in side_effects["heartbeat"])


def test_happy_path_heartbeat_true_delivers_and_marks_once(tmp_path, monkeypatch):
    """Happy path: ownership held for the full lifecycle delivers/marks once."""
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)

    job = {
        "id": "happy-path",
        "name": "happy path",
        "prompt": "ok",
        "script": "work.py",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": "2026-07-12T12:05:00+00:00",
        "enabled": True,
        "fire_claim": {
            "at": "2026-07-12T12:00:00+00:00",
            "by": "happy-owner",
        },
    }

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([job])
        claimed_job = jobs.get_job("happy-path")

    side_effects = {"deliver": 0, "heartbeat_true": 0}
    real_heartbeat = jobs.heartbeat_fire_claim

    def _observed_heartbeat(job_id: str, *, expected_owner: str) -> bool:
        updated = real_heartbeat(job_id, expected_owner=expected_owner)
        if updated:
            side_effects["heartbeat_true"] += 1
        return updated

    def _deliver(job_arg, content, adapters=None, loop=None):
        side_effects["deliver"] += 1
        assert "happy" in content or content
        return None

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _observed_heartbeat)
    monkeypatch.setattr(scheduler, "_deliver_result", _deliver)
    monkeypatch.setattr(
        scheduler, "_run_job_script", lambda *_a, **_k: (True, "happy payload")
    )

    with (
        jobs.use_cron_store(profile_home),
        patch("hermes_state.SessionDB", return_value=MagicMock()),
    ):
        ok = scheduler.run_one_job(claimed_job)
        finished = jobs.get_job("happy-path")

    assert ok is True
    assert side_effects["deliver"] == 1
    assert side_effects["heartbeat_true"] >= 1
    assert finished is not None
    assert finished.get("fire_claim") is None
    assert finished.get("last_status") == "ok"


def test_mark_rejected_after_before_mark_checkpoint_aborts_and_preserves_b(
    tmp_path, monkeypatch
):
    """TOCTOU: before_mark confirms A, then mark CAS loses to B → abort False.

    Checkpoint stage ``before_mark`` synchronously confirms owner A. A
    replacement owner B wins immediately after that checkpoint and before
    ``mark_job_run`` applies under the jobs lock. Owner-conditional mark is a
    full no-op (returns False, B preserved). ``run_one_job`` must report
    False/aborted — not True from ignoring the rejected mark — with no
    exception leak and B's claim/schedule/status/repeat byte-equivalent.
    """
    import copy
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    # Large interval: only synchronous checkpoints observe ownership (not the
    # background heartbeat loop).
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 60.0)

    original_timestamp = "2026-07-12T12:00:00+00:00"
    replacement_timestamp = "2026-07-12T12:00:50+00:00"
    next_run = "2026-07-12T12:20:00+00:00"
    owner_a = "runner-a:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    owner_b = "runner-b:ffffffffffffffffffffffffffffffff"

    base_job = {
        "id": "mark-toctou",
        "name": "mark toctou",
        "prompt": "x",
        "script": "work.py",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": next_run,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": 5, "completed": 2},
        "last_status": None,
        "last_error": None,
        "last_run_at": None,
        "fire_claim": {"at": original_timestamp, "by": owner_a},
    }

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([copy.deepcopy(base_job)])
        claimed_job = jobs.get_job("mark-toctou")

    side_effects = {
        "deliver": 0,
        "heartbeat": [],
        "mark_results": [],
        "mark_expected_owners": [],
    }
    real_heartbeat = jobs.heartbeat_fire_claim
    real_mark = jobs.mark_job_run
    b_snapshot = {}

    def _observed_heartbeat(job_id: str, *, expected_owner: str) -> bool:
        updated = real_heartbeat(job_id, expected_owner=expected_owner)
        side_effects["heartbeat"].append(
            {"expected_owner": expected_owner, "updated": updated}
        )
        return updated

    def _steal_b_then_mark(
        job_id,
        success,
        error=None,
        delivery_error=None,
        *,
        expected_fire_claim_owner=None,
    ):
        # Inject B immediately before/inside the mark CAS so the durable store
        # no longer matches A when the owner-conditional gate runs.
        with jobs.use_cron_store(profile_home):
            jobs.save_jobs(
                [
                    {
                        **base_job,
                        "fire_claim": {
                            "at": replacement_timestamp,
                            "by": owner_b,
                        },
                    }
                ]
            )
            if "before" not in b_snapshot:
                b_snapshot["before"] = copy.deepcopy(jobs.get_job("mark-toctou"))
        applied = real_mark(
            job_id,
            success,
            error,
            delivery_error=delivery_error,
            expected_fire_claim_owner=expected_fire_claim_owner,
        )
        side_effects["mark_results"].append(applied)
        side_effects["mark_expected_owners"].append(expected_fire_claim_owner)
        return applied

    def _deliver(*_a, **_k):
        side_effects["deliver"] += 1
        return None

    def _silent_run_job(job_arg, *, defer_agent_teardown=None):
        # Local/diagnostic path only: SILENT suppresses platform delivery so the
        # run reaches the final mark with no deliver side effect.
        return True, "# local output\n", "[SILENT]", None

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _observed_heartbeat)
    monkeypatch.setattr(scheduler, "mark_job_run", _steal_b_then_mark)
    monkeypatch.setattr(scheduler, "run_job", _silent_run_job)
    monkeypatch.setattr(scheduler, "_deliver_result", _deliver)
    monkeypatch.setattr(
        scheduler, "save_job_output", lambda *a, **k: str(tmp_path / "local.md")
    )

    with jobs.use_cron_store(profile_home):
        try:
            ok = scheduler.run_one_job(claimed_job)
        except Exception as exc:  # pragma: no cover - must not leak
            pytest.fail(f"run_one_job leaked exception: {exc!r}")
        after = jobs.get_job("mark-toctou")

    assert ok is False
    assert side_effects["deliver"] == 0
    # before_mark (and earlier) checkpoints must have confirmed A.
    assert any(h["updated"] is True for h in side_effects["heartbeat"])
    assert all(h["expected_owner"] == owner_a for h in side_effects["heartbeat"])
    # At least one owner-conditional mark attempt was rejected.
    assert side_effects["mark_results"]
    assert all(r is False for r in side_effects["mark_results"])
    assert all(
        owner == owner_a for owner in side_effects["mark_expected_owners"]
    )
    assert "before" in b_snapshot
    assert after == b_snapshot["before"]
    assert after["fire_claim"] == {
        "at": replacement_timestamp,
        "by": owner_b,
    }
    assert after.get("last_status") is None
    assert after.get("last_error") is None
    assert after.get("last_run_at") is None
    assert after.get("repeat") == {"times": 5, "completed": 2}
    assert after.get("next_run_at") == next_run
    assert after.get("state") == "scheduled"


def test_owned_mark_true_keeps_run_one_job_true(tmp_path, monkeypatch):
    """Happy path: owner A still owns at mark → mark True, run_one_job True."""
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 60.0)

    owner_a = "runner-a:11111111111111111111111111111111"
    job = {
        "id": "mark-owned-true",
        "name": "mark owned true",
        "prompt": "ok",
        "script": "work.py",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": "2026-07-12T12:05:00+00:00",
        "enabled": True,
        "repeat": {"times": 2, "completed": 0},
        "fire_claim": {
            "at": "2026-07-12T12:00:00+00:00",
            "by": owner_a,
        },
    }

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([job])
        claimed_job = jobs.get_job("mark-owned-true")

    mark_results = []
    real_mark = jobs.mark_job_run

    def _observed_mark(
        job_id,
        success,
        error=None,
        delivery_error=None,
        *,
        expected_fire_claim_owner=None,
    ):
        applied = real_mark(
            job_id,
            success,
            error,
            delivery_error=delivery_error,
            expected_fire_claim_owner=expected_fire_claim_owner,
        )
        mark_results.append(
            {
                "applied": applied,
                "expected": expected_fire_claim_owner,
                "success": success,
            }
        )
        return applied

    def _silent_run_job(job_arg, *, defer_agent_teardown=None):
        return True, "# out\n", "[SILENT]", None

    monkeypatch.setattr(scheduler, "mark_job_run", _observed_mark)
    monkeypatch.setattr(scheduler, "run_job", _silent_run_job)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **k: None)
    monkeypatch.setattr(
        scheduler, "save_job_output", lambda *a, **k: str(tmp_path / "out.md")
    )

    with jobs.use_cron_store(profile_home):
        ok = scheduler.run_one_job(claimed_job)
        finished = jobs.get_job("mark-owned-true")

    assert ok is True
    assert mark_results
    assert mark_results[0]["applied"] is True
    assert mark_results[0]["expected"] == owner_a
    assert finished is not None
    assert finished.get("fire_claim") is None
    assert finished.get("last_status") == "ok"
