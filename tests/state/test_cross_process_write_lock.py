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
from hermes_state_lock import (
    _lock_open_flags,
    acquire_state_write_lock,
    canonical_db_key,
    write_lock_path,
)

REPO_ROOT = str(Path(__file__).resolve().parents[2])

_HAS_SYMLINK = hasattr(os, "symlink")


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


@pytest.mark.skipif(not _HAS_SYMLINK, reason="platform has no os.symlink")
class TestSymlinkAliasSharesLock:
    """Nemo REQUEST_CHANGES (20260820T042052Z): write_lock_path() derived the
    sidecar from the raw path while canonical_db_key() resolved symlinks, so
    ``state.db`` and ``alias.db -> state.db`` locked different sidecars and
    two processes could hold admission on the same SQLite file at once."""

    def test_write_lock_path_matches_for_symlink_alias(self, tmp_path):
        real_db = tmp_path / "state.db"
        real_db.touch()
        alias_db = tmp_path / "alias.db"
        os.symlink(real_db, alias_db)

        assert write_lock_path(real_db) == write_lock_path(alias_db)
        assert canonical_db_key(real_db) == canonical_db_key(alias_db)

    def test_alias_writer_is_blocked_by_real_path_holder(self, tmp_path):
        real_db = tmp_path / "state.db"
        real_db.touch()
        alias_db = tmp_path / "alias.db"
        os.symlink(real_db, alias_db)

        started = threading.Event()
        release = threading.Event()

        def _hold_real():
            with acquire_state_write_lock(real_db, timeout_s=2.0) as admitted:
                assert admitted is True
                started.set()
                release.wait(5.0)

        holder = threading.Thread(target=_hold_real)
        holder.start()
        try:
            assert started.wait(2.0)
            # A different thread going through the alias must see the same
            # admission token as the real path and time out while it is held.
            t0 = time.monotonic()
            with acquire_state_write_lock(alias_db, timeout_s=0.25) as admitted:
                assert admitted is False
            assert time.monotonic() - t0 < 1.5
        finally:
            release.set()
            holder.join(5.0)

        with acquire_state_write_lock(alias_db, timeout_s=1.0) as admitted:
            assert admitted is True


@pytest.mark.skipif(not _HAS_SYMLINK, reason="platform has no os.symlink")
class TestSidecarSymlinkSubstitution:
    """A sidecar swapped for a symlink (e.g. by an untrusted co-tenant of a
    writable directory) must not be followed: the acquire should degrade to
    admitted rather than silently flock whatever the symlink points at."""

    def test_symlinked_sidecar_is_refused_not_followed(self, tmp_path):
        db_path = tmp_path / "state.db"
        target = tmp_path / "other-process.lock"
        target.write_bytes(b"")
        lock_path = write_lock_path(db_path)
        os.symlink(target, lock_path)

        with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
            # Degrades to admitted (same policy as a read-only directory);
            # the important property is it never opens `target` through the
            # symlink.
            assert admitted is True

        # The symlink itself is untouched — proof the module didn't unlink
        # or rewrite it while degrading.
        assert lock_path.is_symlink()
        assert os.readlink(lock_path) == str(target)


@pytest.mark.skipif(not _HAS_SYMLINK, reason="platform has no os.link")
class TestSidecarInodeSubstitutionBypass:
    """Nemo REQUEST_CHANGES (20260820T114302Z): the sidecar was opened
    without protecting its identity against unlink+recreate or a hardlink
    swap. A co-tenant with write access to the directory could replace
    ``state.db.write.lock`` while a holder still held the flock on the old
    inode; a second acquirer opened the *new* inode and locked it too,
    breaking the mutual exclusion this module exists to provide."""

    def test_hardlink_substitution_does_not_grant_concurrent_admission(
        self, tmp_path
    ):
        db_path = tmp_path / "state.db"
        lock_path = write_lock_path(db_path)
        started = threading.Event()
        release = threading.Event()

        def _holder():
            with acquire_state_write_lock(db_path, timeout_s=5.0) as admitted:
                assert admitted is True
                started.set()
                release.wait(5.0)

        holder = threading.Thread(target=_holder)
        holder.start()
        try:
            assert started.wait(2.0)

            # Simulate a co-tenant with write access to the directory
            # swapping the sidecar for a hardlink to an unrelated file
            # while the holder still owns the flock on the original inode.
            victim = tmp_path / "victim.lock"
            victim.write_bytes(b"")
            os.unlink(lock_path)
            os.link(victim, lock_path)

            t0 = time.monotonic()
            with acquire_state_write_lock(db_path, timeout_s=0.5) as admitted:
                # Must NOT admit on the substituted inode while the real
                # holder is still active — admitting here is the exact
                # exclusion break Nemo flagged.
                assert admitted is False
            assert time.monotonic() - t0 < 1.5
        finally:
            release.set()
            holder.join(5.0)

        # The module must never itself unlink the tampered sidecar (that
        # would just be the same attack performed by trusted code); prove
        # it left the hardlink alone.
        assert os.path.samefile(lock_path, victim)

        # Once the tampering is cleaned up (an operator removing the rogue
        # entry, as would happen operationally), normal admission resumes.
        os.unlink(lock_path)
        with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
            assert admitted is True


class TestLockOpenFlagsPortability:
    def test_posix_adds_nofollow_against_symlink_swap(self):
        flags = _lock_open_flags(is_windows=False)
        assert flags & os.O_CREAT
        assert flags & os.O_RDWR
        assert hasattr(os, "O_NOFOLLOW")
        assert flags & os.O_NOFOLLOW

    def test_windows_has_no_nofollow_equivalent(self):
        # msvcrt has no O_NOFOLLOW; asserting its absence here pins the
        # platform split so it can't silently regress into OSError on
        # Windows (O_NOFOLLOW doesn't exist in the os module there either).
        flags = _lock_open_flags(is_windows=True)
        assert flags == (os.O_RDWR | os.O_CREAT)

    def test_missing_o_nofollow_attribute_degrades_instead_of_crashing(
        self, monkeypatch
    ):
        # os.O_NOFOLLOW is POSIX-only and not guaranteed present on every
        # POSIX platform; a direct `os.O_NOFOLLOW` reference would raise
        # AttributeError outside the caller's `except OSError`. Simulate a
        # platform lacking it and confirm the flag is just omitted.
        monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
        flags = _lock_open_flags(is_windows=False)
        assert flags == (os.O_RDWR | os.O_CREAT)


class TestSustainedContentionNoStarvation:
    def test_many_threads_all_make_progress_under_sustained_contention(
        self, tmp_path
    ):
        """Sustained contention with more concurrent writers than the
        earlier two/three-process tests: every writer must complete within a
        bounded window and none may be starved outright."""
        db_path = tmp_path / "state.db"
        n_workers = 8
        n_rounds = 15
        completions: dict[int, float] = {}
        errors: list[str] = []
        lock = threading.Lock()

        def _worker(idx: int) -> None:
            try:
                for _ in range(n_rounds):
                    with acquire_state_write_lock(
                        db_path, timeout_s=10.0
                    ) as admitted:
                        if not admitted:
                            with lock:
                                errors.append(f"worker {idx} starved")
                            return
                        time.sleep(0.001)
                with lock:
                    completions[idx] = time.monotonic()
            except Exception as exc:  # pragma: no cover - defensive
                with lock:
                    errors.append(f"worker {idx}: {exc!r}")

        threads = [
            threading.Thread(target=_worker, args=(i,)) for i in range(n_workers)
        ]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(30.0)
        for t in threads:
            assert not t.is_alive(), "worker thread hung under contention"

        assert errors == []
        assert len(completions) == n_workers
        assert time.monotonic() - t0 < 25.0


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
