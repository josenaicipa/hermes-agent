"""Cross-process write admission for the shared state.db.

Gateway, Dashboard and ACP share one SQLite file.  SQLite's busy handler is
not a queue, so a sibling's schema init can starve create_session /
append_message until their patience budget expires (2026-08-19 vpsclone:
Dashboard ``_run_init_schema_with_wide_busy_timeout`` vs Gateway 60 s
watchdog).  These tests use real child processes and a temporary database
— never the live profile store.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sqlite3
import sys
import threading
import time
import traceback
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_lock import acquire_state_write_lock

REPO_ROOT = str(Path(__file__).resolve().parents[2])


def _prepare_child_env(hermes_home: str) -> None:
    os.environ["HERMES_HOME"] = hermes_home
    os.environ["HOME"] = hermes_home
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


def _child_init_and_append(
    db_path: str,
    hermes_home: str,
    session_id: str,
    n: int,
    result_path: str,
    err_path: str,
) -> None:
    _prepare_child_env(hermes_home)
    try:
        from hermes_state import SessionDB as ChildDB

        db = ChildDB(db_path=Path(db_path))
        db.create_session(session_id, "cli")
        for i in range(n):
            db.append_message(
                session_id=session_id,
                role="user",
                content=f"{session_id}-{i}",
            )
        db.close()
        Path(result_path).write_text("ok", encoding="utf-8")
    except Exception:
        Path(err_path).write_text(traceback.format_exc(), encoding="utf-8")


def _child_hold_flock(
    db_path: str,
    ready_path: str,
    release_path: str,
    err_path: str,
) -> None:
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    try:
        from hermes_state_lock import acquire_state_write_lock as _acquire

        with _acquire(db_path, timeout_s=5.0) as admitted:
            if not admitted:
                Path(err_path).write_text("not admitted", encoding="utf-8")
                return
            Path(ready_path).touch()
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                if Path(release_path).exists():
                    break
                time.sleep(0.05)
    except Exception:
        Path(err_path).write_text(traceback.format_exc(), encoding="utf-8")


def _spawn(ctx, target, args):
    proc = ctx.Process(target=target, args=args)
    proc.start()
    return proc


def _join_ok(proc, timeout: float, err_path: Path) -> None:
    proc.join(timeout)
    if proc.is_alive():
        proc.kill()
        proc.join(5.0)
        extra = err_path.read_text(encoding="utf-8") if err_path.exists() else ""
        pytest.fail(f"child {proc.pid} still alive after {timeout}s\n{extra}")
    if err_path.exists() and err_path.stat().st_size:
        pytest.fail(err_path.read_text(encoding="utf-8"))
    assert proc.exitcode == 0, f"child exit {proc.exitcode}"


class TestWriteLockPrimitive:
    def test_uncontended_acquire_is_immediate(self, tmp_path):
        db_path = tmp_path / "state.db"
        t0 = time.monotonic()
        with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
            assert admitted is True
        assert time.monotonic() - t0 < 1.0
        # Sidecar may remain; it is not a lock. Re-acquire must succeed.
        with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
            assert admitted is True

    def test_reentrant_on_same_thread(self, tmp_path):
        db_path = tmp_path / "state.db"
        with acquire_state_write_lock(db_path, timeout_s=1.0) as outer:
            assert outer is True
            with acquire_state_write_lock(db_path, timeout_s=0.2) as inner:
                assert inner is True

    def test_timeout_yields_false_not_orphan(self, tmp_path):
        db_path = tmp_path / "state.db"
        started = threading.Event()
        release = threading.Event()

        def _holder():
            with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
                assert admitted is True
                started.set()
                release.wait(5.0)

        holder = threading.Thread(target=_holder)
        holder.start()
        try:
            assert started.wait(2.0)
            t0 = time.monotonic()
            with acquire_state_write_lock(db_path, timeout_s=0.25) as waiter:
                assert waiter is False
            assert time.monotonic() - t0 < 1.5
        finally:
            release.set()
            holder.join(5.0)
        with acquire_state_write_lock(db_path, timeout_s=1.0) as after:
            assert after is True


class TestTwoProcessWriters:
    def test_schema_init_and_critical_append_from_two_processes(self, tmp_path):
        """Two real writer processes: each inits schema and appends."""
        db_path = tmp_path / "state.db"
        home = str(tmp_path / "home")
        Path(home).mkdir()
        ctx = mp.get_context("spawn")
        procs = []
        err_paths = []
        for i in range(2):
            result = tmp_path / f"ok-{i}"
            err = tmp_path / f"err-{i}"
            err_paths.append(err)
            procs.append(
                _spawn(
                    ctx,
                    _child_init_and_append,
                    (str(db_path), home, f"s{i}", 5, str(result), str(err)),
                )
            )
        for proc, err in zip(procs, err_paths):
            _join_ok(proc, 30.0, err)
        for i in range(2):
            assert (tmp_path / f"ok-{i}").read_text(encoding="utf-8") == "ok"

        db = SessionDB(db_path=db_path)
        try:
            assert len(db.get_messages("s0")) == 5
            assert len(db.get_messages("s1")) == 5
        finally:
            db.close()

    def test_bounded_load_no_starvation(self, tmp_path):
        db_path = tmp_path / "state.db"
        home = str(tmp_path / "home")
        Path(home).mkdir()
        ctx = mp.get_context("spawn")
        n_workers = 3
        n_msgs = 8
        procs = []
        err_paths = []
        t0 = time.monotonic()
        for i in range(n_workers):
            result = tmp_path / f"ok-{i}"
            err = tmp_path / f"err-{i}"
            err_paths.append(err)
            procs.append(
                _spawn(
                    ctx,
                    _child_init_and_append,
                    (
                        str(db_path),
                        home,
                        f"w{i}",
                        n_msgs,
                        str(result),
                        str(err),
                    ),
                )
            )
        for proc, err in zip(procs, err_paths):
            _join_ok(proc, 40.0, err)
        elapsed = time.monotonic() - t0
        # Loose bound: coordination must not turn 24 short writes into a
        # patience-length stall.
        assert elapsed < 30.0
        db = SessionDB(db_path=db_path)
        try:
            total = sum(len(db.get_messages(f"w{i}")) for i in range(n_workers))
            assert total == n_workers * n_msgs
        finally:
            db.close()

    def test_waiter_proceeds_after_holder_releases(self, tmp_path):
        db_path = tmp_path / "state.db"
        home = str(tmp_path / "home")
        Path(home).mkdir()
        SessionDB(db_path=db_path).close()

        ctx = mp.get_context("spawn")
        ready = tmp_path / "ready"
        release = tmp_path / "release"
        hold_err = tmp_path / "hold-err"
        holder = _spawn(
            ctx,
            _child_hold_flock,
            (str(db_path), str(ready), str(release), str(hold_err)),
        )
        try:
            deadline = time.monotonic() + 10.0
            while not ready.exists():
                if time.monotonic() > deadline:
                    pytest.fail("holder never acquired the write lock")
                if hold_err.exists() and hold_err.stat().st_size:
                    pytest.fail(hold_err.read_text(encoding="utf-8"))
                time.sleep(0.05)

            def _release_after_pause():
                time.sleep(0.3)
                release.touch()

            threading.Thread(target=_release_after_pause, daemon=True).start()
            started = time.monotonic()
            db = SessionDB(db_path=db_path)
            db.create_session("s-wait", "cli")
            msg_id = db.append_message(
                session_id="s-wait", role="user", content="after-release"
            )
            waited = time.monotonic() - started
            db.close()
            assert isinstance(msg_id, int)
            assert waited < 10.0
        finally:
            release.touch()
            holder.join(10.0)
            if holder.is_alive():
                holder.kill()
                holder.join(5.0)

    def test_holder_death_releases_lock_no_orphan(self, tmp_path):
        db_path = tmp_path / "state.db"
        home = str(tmp_path / "home")
        Path(home).mkdir()
        SessionDB(db_path=db_path).close()

        ctx = mp.get_context("spawn")
        ready = tmp_path / "ready"
        release = tmp_path / "never-release"
        hold_err = tmp_path / "hold-err"
        holder = _spawn(
            ctx,
            _child_hold_flock,
            (str(db_path), str(ready), str(release), str(hold_err)),
        )
        deadline = time.monotonic() + 10.0
        while not ready.exists():
            if time.monotonic() > deadline:
                if holder.is_alive():
                    holder.kill()
                    holder.join(5.0)
                pytest.fail("holder never acquired the write lock")
            time.sleep(0.05)

        holder.kill()
        holder.join(5.0)
        assert not holder.is_alive()

        # The sidecar file may remain; flock is not a pidfile. Acquire
        # succeeding quickly is the proof there is no orphan lock.
        t0 = time.monotonic()
        with acquire_state_write_lock(db_path, timeout_s=3.0) as admitted:
            assert admitted is True
        assert time.monotonic() - t0 < 2.0
        db = SessionDB(db_path=db_path)
        try:
            db.create_session("s-after-kill", "cli")
            db.append_message(
                session_id="s-after-kill",
                role="user",
                content="holder died",
            )
            assert len(db.get_messages("s-after-kill")) == 1
        finally:
            db.close()

    def test_readers_stay_concurrent_with_writer(self, tmp_path):
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        db.create_session("s-read", "cli")
        db.append_message(session_id="s-read", role="user", content="seed")
        db.close()

        ctx = mp.get_context("spawn")
        ready = tmp_path / "ready"
        release = tmp_path / "release"
        hold_err = tmp_path / "hold-err"
        holder = _spawn(
            ctx,
            _child_hold_flock,
            (str(db_path), str(ready), str(release), str(hold_err)),
        )
        try:
            deadline = time.monotonic() + 10.0
            while not ready.exists():
                if time.monotonic() > deadline:
                    pytest.fail("holder never acquired the write lock")
                time.sleep(0.05)
            reader = SessionDB(db_path=db_path, read_only=True)
            try:
                t0 = time.monotonic()
                msgs = reader.get_messages("s-read")
                assert any(m["content"] == "seed" for m in msgs)
                assert time.monotonic() - t0 < 2.0
            finally:
                reader.close()
        finally:
            release.touch()
            holder.join(10.0)
            if holder.is_alive():
                holder.kill()
                holder.join(5.0)


class TestNullActiveHealSkipsNoopWrite:
    def test_reopen_does_not_need_null_active_rows(self, tmp_path):
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        db.create_session("s1", "cli")
        db.append_message(session_id="s1", role="user", content="x")
        db.close()
        conn = sqlite3.connect(str(db_path))
        try:
            n = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE active IS NULL"
            ).fetchone()[0]
        finally:
            conn.close()
        assert n == 0
        t0 = time.monotonic()
        db2 = SessionDB(db_path=db_path)
        try:
            assert len(db2.get_messages("s1")) == 1
        finally:
            db2.close()
        assert time.monotonic() - t0 < 5.0
