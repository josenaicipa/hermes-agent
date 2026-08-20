"""Long state.db maintenance must not starve interactive writers.

2026-08-20 vpsclone: ``maybe_auto_prune_and_vacuum`` → ``vacuum()`` held the
per-user ``global.write.lock`` around VACUUM + TRUNCATE of a 12–30 GB DB
for >20 min. Gateway routing saves expired at 20 s and ``append_message``
at 60 s (``session storage was busy`` / ``database is locked``) while
concurrent ACPs and other profiles' ``state.db`` files sat behind the same
flock.

These tests reproduce that class with a slow VACUUM and a concurrent
critical writer. They use temporary databases only — never the live
profile store.
"""

from __future__ import annotations

import threading
import time

import pytest

from hermes_state import SessionDB


def _slow_vacuum_execute(conn, hold_s: float, started: threading.Event):
    """Wrap ``conn.execute`` so VACUUM blocks for *hold_s* seconds."""
    real_execute = conn.execute

    def execute(sql, *args, **kwargs):
        text = str(sql).strip().upper()
        if text == "VACUUM" or text.startswith("VACUUM "):
            started.set()
            time.sleep(hold_s)
        return real_execute(sql, *args, **kwargs)

    conn.execute = execute  # type: ignore[method-assign]
    return real_execute


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    d.close()


class TestVacuumDoesNotHoldGlobalFlock:
    def test_critical_writer_on_other_db_succeeds_during_slow_vacuum(
        self, tmp_path
    ):
        """VACUUM must not pin the per-user admission token.

        A writer on a *different* Hermes state database (ACP / other
        profile) has to land while this file is mid-rewrite. Holding the
        global flock for the rewrite is exactly the 2026-08-20 outage.
        """
        maint = SessionDB(db_path=tmp_path / "maint.db")
        live = SessionDB(db_path=tmp_path / "live.db")
        try:
            live.create_session("s-live", "cli")
            started = threading.Event()
            hold_s = 3.0
            _slow_vacuum_execute(maint._conn, hold_s, started)
            errors: list[BaseException] = []

            def _run_vacuum() -> None:
                try:
                    maint.vacuum()
                except BaseException as exc:  # pragma: no cover - surfaced below
                    errors.append(exc)

            t = threading.Thread(target=_run_vacuum)
            t.start()
            assert started.wait(5.0), "VACUUM never started"
            t0 = time.monotonic()
            msg_id = live.append_message(
                session_id="s-live",
                role="user",
                content="critical-during-vacuum",
            )
            elapsed = time.monotonic() - t0
            t.join(timeout=hold_s + 5.0)
            assert not t.is_alive()
            assert errors == []
            assert isinstance(msg_id, int)
            assert elapsed < 1.5, (
                f"critical writer blocked {elapsed:.2f}s behind VACUUM; "
                "the per-user flock is still held across the rewrite"
            )
            msgs = live.get_messages("s-live")
            assert any(m["content"] == "critical-during-vacuum" for m in msgs)
        finally:
            maint.close()
            live.close()

    def test_vacuum_still_runs_when_uncontended(self, db):
        db.create_session("s1", "cli")
        db.append_message(session_id="s1", role="user", content="hi")
        db.vacuum()  # must not raise
        assert len(db.get_messages("s1")) == 1


class TestAutoMaintenanceDoesNotBlockCriticalWriter:
    def _make_old_ended(self, db: SessionDB, sid: str, days_old: int = 100) -> None:
        db.create_session(session_id=sid, source="cli")
        db.end_session(sid, end_reason="done")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (time.time() - days_old * 86400, sid),
        )
        db._conn.commit()

    def test_auto_prune_skips_vacuum_so_writer_is_not_blocked(
        self, db, monkeypatch
    ):
        self._make_old_ended(db, "old1")
        db.create_session("s-live", "cli")

        def _forbidden_vacuum() -> int:
            raise AssertionError(
                "maybe_auto_prune_and_vacuum must not call vacuum(); "
                "automatic VACUUM is what monopolized global.write.lock"
            )

        monkeypatch.setattr(db, "vacuum", _forbidden_vacuum)

        started = threading.Event()
        result_box: dict[str, object] = {}
        errors: list[BaseException] = []

        def _maint() -> None:
            started.set()
            try:
                result_box["result"] = db.maybe_auto_prune_and_vacuum(
                    retention_days=90, min_interval_hours=0
                )
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        t = threading.Thread(target=_maint)
        t.start()
        assert started.wait(5.0)
        t0 = time.monotonic()
        msg_id = db.append_message(
            session_id="s-live",
            role="user",
            content="critical-during-auto-maint",
        )
        elapsed = time.monotonic() - t0
        t.join(timeout=10.0)
        assert not t.is_alive()
        assert errors == []
        result = result_box["result"]
        assert isinstance(result, dict)
        assert result["pruned"] == 1
        assert result["vacuumed"] is False
        assert result.get("vacuum_deferred") is True
        assert isinstance(msg_id, int)
        assert elapsed < 1.5
        msgs = db.get_messages("s-live")
        assert any(m["content"] == "critical-during-auto-maint" for m in msgs)

    def test_auto_maintenance_does_not_call_vacuum_even_if_due(
        self, db, monkeypatch
    ):
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 5)
        called = []
        monkeypatch.setattr(db, "vacuum", lambda: called.append(True))
        result = db.maybe_auto_prune_and_vacuum(min_interval_hours=0)
        assert called == []
        assert result["vacuumed"] is False
        assert result.get("vacuum_deferred") is True
