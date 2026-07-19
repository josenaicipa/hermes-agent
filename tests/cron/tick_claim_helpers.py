"""Helpers so unit tests that mock ``get_due_jobs`` still pass tick's CAS fence.

The built-in ticker now mints a per-fire token, ``claim_job_for_fire``s, and
re-reads via ``get_job`` before every submit. Characterization tests that feed
synthetic due snapshots (no real jobs.json row) need a matching claim path.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import patch


DueJobs = Union[List[Dict[str, Any]], Callable[[], List[Dict[str, Any]]]]


def _resolve_due(due_jobs: DueJobs) -> List[Dict[str, Any]]:
    jobs = due_jobs() if callable(due_jobs) else due_jobs
    return list(jobs or [])


@contextmanager
def successful_tick_fire_claims(due_jobs: DueJobs):
    """Context manager: tick CAS claim always wins for the given due snapshots.

    Also stubs ``heartbeat_fire_claim`` so ``run_one_job`` preflight succeeds
    when the claimed record is not present in a real durable store.
    """
    tokens: Dict[str, str] = {}

    def _mint() -> str:
        return f"test-tick:{uuid.uuid4().hex}"

    def _claim(jid: str, *, claim_owner: Optional[str] = None, **_kw) -> bool:
        by_id = {j["id"]: j for j in _resolve_due(due_jobs) if j.get("id")}
        if jid not in by_id:
            return False
        tokens[jid] = claim_owner or _mint()
        return True

    def _get(jid: str) -> Optional[Dict[str, Any]]:
        by_id = {j["id"]: j for j in _resolve_due(due_jobs) if j.get("id")}
        base = by_id.get(jid)
        if base is None:
            return None
        out = dict(base)
        tok = tokens.get(jid)
        if tok:
            out["fire_claim"] = {
                "at": "2026-07-12T12:00:00+00:00",
                "by": tok,
            }
        return out

    with patch("cron.scheduler.new_fire_claim_owner", side_effect=_mint), \
         patch("cron.scheduler.claim_job_for_fire", side_effect=_claim), \
         patch("cron.scheduler.get_job", side_effect=_get), \
         patch("cron.scheduler.heartbeat_fire_claim", return_value=True):
        yield


def install_successful_tick_fire_claims(monkeypatch, due_jobs: DueJobs, sched=None):
    """monkeypatch variant of :func:`successful_tick_fire_claims`."""
    import cron.scheduler as default_sched

    target = sched if sched is not None else default_sched
    tokens: Dict[str, str] = {}

    def _mint() -> str:
        return f"test-tick:{uuid.uuid4().hex}"

    def _claim(jid: str, *, claim_owner: Optional[str] = None, **_kw) -> bool:
        by_id = {j["id"]: j for j in _resolve_due(due_jobs) if j.get("id")}
        if jid not in by_id:
            return False
        tokens[jid] = claim_owner or _mint()
        return True

    def _get(jid: str) -> Optional[Dict[str, Any]]:
        by_id = {j["id"]: j for j in _resolve_due(due_jobs) if j.get("id")}
        base = by_id.get(jid)
        if base is None:
            return None
        out = dict(base)
        tok = tokens.get(jid)
        if tok:
            out["fire_claim"] = {
                "at": "2026-07-12T12:00:00+00:00",
                "by": tok,
            }
        return out

    monkeypatch.setattr(target, "new_fire_claim_owner", _mint, raising=False)
    monkeypatch.setattr(target, "claim_job_for_fire", _claim, raising=False)
    monkeypatch.setattr(target, "get_job", _get, raising=False)
    monkeypatch.setattr(
        target, "heartbeat_fire_claim", lambda *a, **k: True, raising=False
    )
