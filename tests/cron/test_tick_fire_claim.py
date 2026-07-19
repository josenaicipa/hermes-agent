"""Built-in tick must use the same per-fire CAS claim as manual/external.

Inverse cross-path race (proven):
  1. tick: due_jobs = get_due_jobs()  # releases jobs lock
  2. tick later: advance_next_run + submit no-claim snapshot
  3. manual/external: claim_job_for_fire wins between due scan and submit
  → both execute (double fire)

Required: after get_due_jobs and before pool submit, tick must mint a unique
token, CAS-claim, exact-token re-read, and only then dispatch the claimed
record. claim_job_for_fire already advances recurring next_run_at — no
separate advance_next_run for those jobs.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def temp_home(tmp_path):
    """Route cron storage explicitly so import order cannot touch the real store."""
    from cron.jobs import use_cron_store

    with use_cron_store(tmp_path):
        yield tmp_path


@pytest.fixture
def isolate_tick_lock(tmp_path, monkeypatch):
    """Point the tick file lock at a per-test temp dir (xdist-safe)."""
    lock_dir = tmp_path / "cron-lock"
    lock_dir.mkdir(exist_ok=True)
    lock_file = lock_dir / ".tick.lock"
    monkeypatch.setattr(
        "cron.scheduler._get_lock_paths",
        lambda: (lock_dir, lock_file),
    )
    yield


def _force_due(job_id: str, *, minutes_ago: int = 5) -> None:
    from cron.jobs import load_jobs, save_jobs

    jobs = load_jobs()
    for job in jobs:
        if job["id"] == job_id:
            job["next_run_at"] = (
                datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
            ).isoformat()
            job["fire_claim"] = None
    save_jobs(jobs)


def _reset_scheduler_runtime():
    import cron.scheduler as sched

    sched._parallel_pool = None
    sched._parallel_pool_max_workers = None
    sched._running_job_ids.clear()


# ---------------------------------------------------------------------------
# A. Scheduled tick claim: CAS + exact-token + claimed record + no advance
# ---------------------------------------------------------------------------


def test_tick_cas_claims_due_job_and_dispatches_claimed_record(
    temp_home, isolate_tick_lock, monkeypatch
):
    """A: due recurring job is CAS-claimed with unique token; run_one_job gets it.

    claim_job_for_fire already advances next_run_at; tick must NOT call
    advance_next_run for that job (no double-advance).
    """
    import cron.jobs as jobs_mod
    import cron.scheduler as sched
    from cron.jobs import create_job, get_job

    _reset_scheduler_runtime()
    job = create_job(prompt="tick claim", schedule="every 5m", name="tick-cas")
    jid = job["id"]
    _force_due(jid)
    before_next = get_job(jid)["next_run_at"]

    seen = []
    advance_calls = []
    minted = []

    real_new = jobs_mod.new_fire_claim_owner
    real_claim = jobs_mod.claim_job_for_fire
    real_advance = jobs_mod.advance_next_run

    def _mint():
        tok = real_new()
        minted.append(tok)
        return tok

    def _claim(job_id, *, claim_owner=None, **kw):
        return real_claim(job_id, claim_owner=claim_owner, **kw)

    def _advance(job_id):
        advance_calls.append(job_id)
        return real_advance(job_id)

    monkeypatch.setattr(sched, "new_fire_claim_owner", _mint, raising=False)
    monkeypatch.setattr(sched, "claim_job_for_fire", _claim, raising=False)
    monkeypatch.setattr(sched, "get_job", jobs_mod.get_job, raising=False)
    monkeypatch.setattr(sched, "advance_next_run", _advance)
    monkeypatch.setattr(
        sched,
        "run_one_job",
        lambda j, **kw: seen.append(j) or True,
    )

    n = sched.tick(verbose=False, sync=True)
    assert n == 1
    assert len(seen) == 1
    dispatched = seen[0]
    assert dispatched["id"] == jid
    claim = dispatched.get("fire_claim")
    assert isinstance(claim, dict)
    token = claim.get("by")
    assert token
    assert token in minted
    # Post-claim durable record must match the exact token generation.
    durable = get_job(jid)
    assert durable is not None
    assert durable.get("fire_claim", {}).get("by") == token
    # claim_job_for_fire advanced next_run; tick must not double-advance.
    assert advance_calls == [], (
        f"tick must not call advance_next_run after claim; got {advance_calls}"
    )
    assert durable["next_run_at"] != before_next

    _reset_scheduler_runtime()


# ---------------------------------------------------------------------------
# B. Manual/external wins race: claim lost → no submit/run/count
# ---------------------------------------------------------------------------


def test_tick_lost_claim_does_not_submit_run_or_count(
    temp_home, isolate_tick_lock, monkeypatch
):
    """B: get_due returns snapshot; claim returns False → no side effects."""
    import cron.scheduler as sched

    _reset_scheduler_runtime()
    due = {
        "id": "lost-claim-job",
        "name": "lost",
        "prompt": "x",
        "schedule": {"kind": "interval", "minutes": 5},
        "enabled": True,
        "next_run_at": "2020-01-01T00:00:00+00:00",
        "deliver": "local",
    }
    ran = []
    advance_calls = []

    monkeypatch.setattr(sched, "get_due_jobs", lambda: [due])
    monkeypatch.setattr(
        sched,
        "new_fire_claim_owner",
        lambda: "tick:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        raising=False,
    )
    monkeypatch.setattr(
        sched, "claim_job_for_fire", lambda jid, **kw: False, raising=False
    )
    monkeypatch.setattr(
        sched,
        "get_job",
        lambda jid: (_ for _ in ()).throw(
            AssertionError("get_job must not run after lost claim")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        sched, "advance_next_run", lambda jid: advance_calls.append(jid) or True
    )
    monkeypatch.setattr(
        sched, "run_one_job", lambda j, **kw: ran.append(j["id"]) or True
    )

    n = sched.tick(verbose=False, sync=True)
    assert n == 0
    assert ran == []
    assert advance_calls == []

    _reset_scheduler_runtime()


# ---------------------------------------------------------------------------
# C. Tick wins race: manual claim loses while scheduled run in flight
# ---------------------------------------------------------------------------


def test_tick_wins_claim_manual_loses_only_one_side_effect(
    temp_home, isolate_tick_lock, monkeypatch
):
    """C: tick claims token A; subsequent manual claim loses; one side effect."""
    import cron.jobs as jobs_mod
    import cron.scheduler as sched
    from cron.jobs import claim_job_for_fire, create_job, get_job, new_fire_claim_owner

    _reset_scheduler_runtime()
    job = create_job(prompt="race", schedule="every 5m", name="tick-wins")
    jid = job["id"]
    _force_due(jid)

    entered = threading.Event()
    release = threading.Event()
    side_effects = []

    def _slow_run(j, **kw):
        claim = j.get("fire_claim") if isinstance(j, dict) else None
        owner = claim.get("by") if isinstance(claim, dict) else None
        side_effects.append(("scheduled", owner))
        entered.set()
        assert release.wait(timeout=5)
        return True

    monkeypatch.setattr(sched, "run_one_job", _slow_run)

    # Async tick: claim + submit, return without waiting.
    n = sched.tick(verbose=False, sync=False)
    assert n == 1
    assert entered.wait(timeout=5)

    # While scheduled run is in flight / claim is fresh, manual claim loses.
    manual_token = new_fire_claim_owner()
    assert claim_job_for_fire(jid, claim_owner=manual_token) is False
    durable = get_job(jid)
    assert durable is not None
    durable_claim = durable.get("fire_claim")
    assert isinstance(durable_claim, dict)
    # Tick's token still owns the claim — not the manual token.
    assert durable_claim["by"] != manual_token
    assert side_effects == [
        ("scheduled", durable_claim["by"]),
    ]

    release.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and sched.get_running_job_ids():
        time.sleep(0.01)
    sched._shutdown_parallel_pool()
    _reset_scheduler_runtime()


# ---------------------------------------------------------------------------
# D. Post-claim re-read missing / wrong-token → fail closed
# ---------------------------------------------------------------------------


def test_tick_post_claim_missing_job_fail_closed(isolate_tick_lock, monkeypatch):
    """D1: claim wins but get_job is None → no submit/run; leave claim alone."""
    import cron.scheduler as sched

    _reset_scheduler_runtime()
    due = {
        "id": "missing-after-claim",
        "name": "gone",
        "prompt": "x",
        "enabled": True,
        "next_run_at": "2020-01-01T00:00:00+00:00",
        "deliver": "local",
    }
    token = "tick:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    ran = []
    claim_calls = []

    monkeypatch.setattr(sched, "get_due_jobs", lambda: [due])
    monkeypatch.setattr(sched, "new_fire_claim_owner", lambda: token, raising=False)
    monkeypatch.setattr(
        sched,
        "claim_job_for_fire",
        lambda jid, **kw: claim_calls.append((jid, kw.get("claim_owner"))) or True,
        raising=False,
    )
    monkeypatch.setattr(sched, "get_job", lambda jid: None, raising=False)
    monkeypatch.setattr(
        sched, "run_one_job", lambda j, **kw: ran.append(j) or True
    )
    # Must not fall back to clearing unknown claims.
    clear_calls = []
    monkeypatch.setattr(
        sched,
        "mark_job_run",
        lambda *a, **k: clear_calls.append((a, k)),
    )

    n = sched.tick(verbose=False, sync=True)
    assert n == 0
    assert ran == []
    assert claim_calls == [("missing-after-claim", token)]
    assert clear_calls == []

    _reset_scheduler_runtime()


def test_tick_post_claim_wrong_token_fail_closed(isolate_tick_lock, monkeypatch):
    """D2: re-read fire_claim.by != minted token → no submit; do not clear."""
    import cron.scheduler as sched

    _reset_scheduler_runtime()
    due = {
        "id": "wrong-token-job",
        "name": "wrong",
        "prompt": "x",
        "enabled": True,
        "next_run_at": "2020-01-01T00:00:00+00:00",
        "deliver": "local",
    }
    token_a = "tick:cccccccccccccccccccccccccccccccc"
    token_b = "tick:dddddddddddddddddddddddddddddddd"
    ran = []
    clear_calls = []

    monkeypatch.setattr(sched, "get_due_jobs", lambda: [due])
    monkeypatch.setattr(sched, "new_fire_claim_owner", lambda: token_a, raising=False)
    monkeypatch.setattr(
        sched, "claim_job_for_fire", lambda jid, **kw: True, raising=False
    )
    monkeypatch.setattr(
        sched,
        "get_job",
        lambda jid: {
            "id": jid,
            "name": "wrong",
            "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": token_b},
        },
        raising=False,
    )
    monkeypatch.setattr(
        sched, "run_one_job", lambda j, **kw: ran.append(j) or True
    )
    monkeypatch.setattr(
        sched,
        "mark_job_run",
        lambda *a, **k: clear_calls.append((a, k)),
    )

    n = sched.tick(verbose=False, sync=True)
    assert n == 0
    assert ran == []
    assert clear_calls == []

    _reset_scheduler_runtime()


# ---------------------------------------------------------------------------
# E. Multiple due jobs: independent claim/filter
# ---------------------------------------------------------------------------


def test_tick_multiple_due_jobs_independent_claim_filter(
    isolate_tick_lock, monkeypatch
):
    """E: one lost claim must not block other valid claimed jobs."""
    import cron.scheduler as sched

    _reset_scheduler_runtime()
    due = [
        {
            "id": "win-a",
            "name": "a",
            "prompt": "a",
            "enabled": True,
            "next_run_at": "2020-01-01T00:00:00+00:00",
            "deliver": "local",
        },
        {
            "id": "lose-b",
            "name": "b",
            "prompt": "b",
            "enabled": True,
            "next_run_at": "2020-01-01T00:00:00+00:00",
            "deliver": "local",
        },
        {
            "id": "win-c",
            "name": "c",
            "prompt": "c",
            "enabled": True,
            "next_run_at": "2020-01-01T00:00:00+00:00",
            "deliver": "local",
        },
    ]
    tokens = {
        "win-a": "tick:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "lose-b": "tick:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "win-c": "tick:cccccccccccccccccccccccccccccccc",
    }
    mint_order = []

    def _mint():
        # Order follows due list iteration.
        jid = due[len(mint_order)]["id"]
        tok = tokens[jid]
        mint_order.append(jid)
        return tok

    def _claim(jid, *, claim_owner=None, **kw):
        assert claim_owner == tokens[jid]
        return jid != "lose-b"

    def _get(jid):
        if jid == "lose-b":
            raise AssertionError("should not re-read after lost claim")
        return {
            "id": jid,
            "name": jid,
            "fire_claim": {
                "at": "2026-07-12T12:00:00+00:00",
                "by": tokens[jid],
            },
            "deliver": "local",
        }

    ran = []
    monkeypatch.setattr(sched, "get_due_jobs", lambda: list(due))
    monkeypatch.setattr(sched, "new_fire_claim_owner", _mint, raising=False)
    monkeypatch.setattr(sched, "claim_job_for_fire", _claim, raising=False)
    monkeypatch.setattr(sched, "get_job", _get, raising=False)
    monkeypatch.setattr(
        sched, "run_one_job", lambda j, **kw: ran.append(j["id"]) or True
    )

    n = sched.tick(verbose=False, sync=True)
    assert n == 2
    assert ran == ["win-a", "win-c"]
    assert "lose-b" not in ran

    _reset_scheduler_runtime()


# ---------------------------------------------------------------------------
# F. Async/sync tick + in-process running guard remain compatible
# ---------------------------------------------------------------------------


def test_tick_sync_async_and_running_guard_with_claims(
    isolate_tick_lock, monkeypatch
):
    """F: claim path keeps sync count, async optimistic count, running guard."""
    import cron.scheduler as sched

    _reset_scheduler_runtime()
    job = {
        "id": "guard-claimed",
        "name": "guard",
        "prompt": "x",
        "enabled": True,
        "next_run_at": "2020-01-01T00:00:00+00:00",
        "deliver": "local",
    }
    token = "tick:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    barrier = threading.Barrier(2, timeout=5)

    def _claim_ok(jid, *, claim_owner=None, **kw):
        return True

    def _get(jid):
        return {
            **job,
            "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": token},
        }

    def _slow_run(j, **kw):
        barrier.wait()
        return True

    monkeypatch.setattr(sched, "get_due_jobs", lambda: [job])
    monkeypatch.setattr(sched, "new_fire_claim_owner", lambda: token, raising=False)
    monkeypatch.setattr(sched, "claim_job_for_fire", _claim_ok, raising=False)
    monkeypatch.setattr(sched, "get_job", _get, raising=False)
    monkeypatch.setattr(sched, "run_one_job", _slow_run)

    # Async: optimistic count 1, returns before job finishes.
    start = time.monotonic()
    n_async = sched.tick(verbose=False, sync=False)
    elapsed = time.monotonic() - start
    assert n_async == 1
    assert elapsed < 1.0

    # While still running, next tick must not re-dispatch (in-process guard).
    # Claim may succeed or fail depending on store; guard is process-local.
    # Simulate another due scan + successful claim race against running set.
    n_guard = sched.tick(verbose=False, sync=True)
    assert n_guard == 0

    barrier.wait()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and sched.get_running_job_ids():
        time.sleep(0.01)

    # After release, sync tick dispatches once and waits.
    _reset_scheduler_runtime()
    monkeypatch.setattr(sched, "run_one_job", lambda j, **kw: True)
    n_sync = sched.tick(verbose=False, sync=True)
    assert n_sync == 1

    sched._shutdown_parallel_pool()
    _reset_scheduler_runtime()


def test_tick_does_not_use_due_snapshot_when_claim_verification_fails(
    isolate_tick_lock, monkeypatch
):
    """Never fall back to the pre-claim due snapshot for dispatch."""
    import cron.scheduler as sched

    _reset_scheduler_runtime()
    due_snapshot = {
        "id": "snap-job",
        "name": "snap",
        "prompt": "from-due",
        "enabled": True,
        "fire_claim": None,  # pre-claim snapshot has no token
        "next_run_at": "2020-01-01T00:00:00+00:00",
        "deliver": "local",
    }
    token = "tick:ffffffffffffffffffffffffffffffff"
    ran = []

    monkeypatch.setattr(sched, "get_due_jobs", lambda: [due_snapshot])
    monkeypatch.setattr(sched, "new_fire_claim_owner", lambda: token, raising=False)
    monkeypatch.setattr(
        sched, "claim_job_for_fire", lambda jid, **kw: True, raising=False
    )
    # Wrong token on re-read — must not dispatch due_snapshot either.
    monkeypatch.setattr(
        sched,
        "get_job",
        lambda jid: {
            **due_snapshot,
            "fire_claim": {
                "at": "2026-07-12T12:00:00+00:00",
                "by": "other:00000000000000000000000000000000",
            },
        },
        raising=False,
    )
    monkeypatch.setattr(
        sched, "run_one_job", lambda j, **kw: ran.append(j) or True
    )

    assert sched.tick(verbose=False, sync=True) == 0
    assert ran == []

    _reset_scheduler_runtime()
