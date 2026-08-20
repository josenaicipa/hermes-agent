"""Cross-process write admission for Hermes state databases.

Gateway, Dashboard and ACP share SQLite files.  SQLite's busy handler is
not a queue, so a sibling's schema init can starve create_session /
append_message until their patience budget expires (2026-08-19 vpsclone:
Dashboard ``_run_init_schema_with_wide_busy_timeout`` vs Gateway 60 s
watchdog).  These tests use real child processes and temporary databases
— never the live profile store.

Every test in this module gets its own isolated private lock root under
``tmp_path`` (see the ``_isolated_lock_root`` autouse fixture below) — never
the real per-account root (``/tmp/hermes-state-locks-<uid>`` on POSIX) that
any live Hermes process on the same machine, under the same OS account,
might already be relying on.

Fase C7: ``hermes_state_lock`` used to name its sidecar after a stable hash
of ``canonical_db_key(db_path)`` — symlink-resolved, and case/Unicode-folded
where the underlying filesystem actually folded those. That kept growing to
cover one more alias class and was broken outright by a simpler one: an
arbitrary *hardlink* to a database's inode, created under any name,
resolves through ``realpath`` to the identical canonical path the real
database has — there is no alias to detect, the hardlink's identity *is*
the real identity. The fix removed per-database identity entirely: there is
now exactly one sidecar (``global.write.lock``) per OS account, shared by
every Hermes state database that account writes to. ``write_lock_path()``
takes no database argument at all, so most of the alias-specific test
classes this module used to need (symlink, hardlink, case-insensitive
volume, Unicode normalization/casefold) no longer test anything meaningful
— convergence for those is now a structural property of the function
signature, not a per-alias behavior to verify. What still needs runtime
coverage is the actual admission behavior: two genuinely distinct
databases, and every alias class at once, must now serialize against the
same lock — see ``TestGlobalLockUnifiesEveryAliasAndDatabase``.
"""

from __future__ import annotations

import inspect
import json
import multiprocessing as mp
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
import threading
import time
import traceback
import unicodedata
from pathlib import Path

import pytest

import hermes_state_lock
from hermes_state import SessionDB
from hermes_state_lock import (
    _lock_open_flags,
    _lock_root,
    _posix_lock_root,
    _windows_lock_root,
    acquire_state_write_lock,
    write_lock_path,
)

REPO_ROOT = str(Path(__file__).resolve().parents[2])

_HAS_SYMLINK = hasattr(os, "symlink")
_HAS_HARDLINK = hasattr(os, "link")
_IS_WINDOWS = sys.platform == "win32"


@pytest.fixture(autouse=True)
def _isolated_lock_root(tmp_path, monkeypatch):
    """Point every acquire in this test at a private root under tmp_path.

    Without this, ``hermes_state_lock`` would resolve its default private
    root — a real, shared-by-uid location — and these tests would create,
    chmod, symlink and hardlink-swap files there. On this machine that
    directory can be the one a live Hermes gateway process (same OS
    account) is actively using for its own state.db admission.
    """
    root = tmp_path / "lock-root"
    monkeypatch.setenv("HERMES_STATE_LOCK_ROOT", str(root))
    return root


def _prepare_child_env(hermes_home: str, lock_root: str) -> None:
    os.environ["HERMES_HOME"] = hermes_home
    os.environ["HOME"] = hermes_home
    os.environ["HERMES_STATE_LOCK_ROOT"] = lock_root
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


def _child_init_and_append(
    db_path: str,
    hermes_home: str,
    lock_root: str,
    session_id: str,
    n: int,
    result_path: str,
    err_path: str,
) -> None:
    _prepare_child_env(hermes_home, lock_root)
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
    lock_root: str,
    ready_path: str,
    release_path: str,
    err_path: str,
) -> None:
    os.environ["HERMES_STATE_LOCK_ROOT"] = lock_root
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


def _child_cooperative_cycles(
    db_path: str,
    lock_root: str,
    n: int,
    identity_path: str,
    err_path: str,
) -> None:
    """Cooperative worker: N plain acquire/release cycles, no tampering.

    Records the sidecar's (st_dev, st_ino) after each cycle so the parent
    can assert every cooperative process — and every cycle within each —
    observed exactly one, unchanging identity. Never unlinks the sidecar
    itself; that is the property under test.
    """
    os.environ["HERMES_STATE_LOCK_ROOT"] = lock_root
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    try:
        from hermes_state_lock import acquire_state_write_lock as _acquire
        from hermes_state_lock import write_lock_path as _write_lock_path

        identities = []
        for _ in range(n):
            with _acquire(db_path, timeout_s=5.0) as admitted:
                if not admitted:
                    Path(err_path).write_text("not admitted", encoding="utf-8")
                    return
                st = os.stat(_write_lock_path())
                identities.append([st.st_dev, st.st_ino])
        Path(identity_path).write_text(json.dumps(identities), encoding="utf-8")
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
    def test_schema_init_and_critical_append_from_two_processes(
        self, tmp_path, _isolated_lock_root
    ):
        """Two real writer processes: each inits schema and appends."""
        db_path = tmp_path / "state.db"
        home = str(tmp_path / "home")
        Path(home).mkdir()
        lock_root = str(_isolated_lock_root)
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
                    (
                        str(db_path),
                        home,
                        lock_root,
                        f"s{i}",
                        5,
                        str(result),
                        str(err),
                    ),
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

    def test_bounded_load_no_starvation(self, tmp_path, _isolated_lock_root):
        db_path = tmp_path / "state.db"
        home = str(tmp_path / "home")
        Path(home).mkdir()
        lock_root = str(_isolated_lock_root)
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
                        lock_root,
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

    def test_waiter_proceeds_after_holder_releases(
        self, tmp_path, _isolated_lock_root
    ):
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
            (
                str(db_path),
                str(_isolated_lock_root),
                str(ready),
                str(release),
                str(hold_err),
            ),
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

    def test_holder_death_releases_lock_no_orphan(
        self, tmp_path, _isolated_lock_root
    ):
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
            (
                str(db_path),
                str(_isolated_lock_root),
                str(ready),
                str(release),
                str(hold_err),
            ),
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

    def test_readers_stay_concurrent_with_writer(
        self, tmp_path, _isolated_lock_root
    ):
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
            (
                str(db_path),
                str(_isolated_lock_root),
                str(ready),
                str(release),
                str(hold_err),
            ),
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


class TestAcquireSignatureCompatibility:
    """Requirement 2 (Fase C7): ``acquire_state_write_lock(db_path, ...)``
    keeps its signature even though ``db_path`` no longer derives the
    lock's identity — every call site in this codebase (``hermes_state.py``)
    still passes one, and it remains useful for logging/diagnostics."""

    def test_db_path_remains_the_first_positional_parameter(self):
        sig = inspect.signature(acquire_state_write_lock)
        params = list(sig.parameters.values())
        assert params[0].name == "db_path"
        assert params[0].kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        assert "timeout_s" in sig.parameters
        assert sig.parameters["timeout_s"].default == 1.0

    def test_write_lock_path_takes_no_database_argument(self):
        # Requirement 3: canonicalization/hashing of db_path is gone from
        # the admission path entirely — there is nothing left to pass.
        assert inspect.signature(write_lock_path).parameters == {}


class TestGlobalLockUnifiesEveryAliasAndDatabase:
    """Fase C7 (gate 20260820T134628Z): a hardlink to a database's inode,
    created under an arbitrary name, resolves through ``realpath`` to that
    database's own canonical path — the old per-database identity
    (``canonical_db_key`` + case/Unicode folding + hashing) had no way to
    treat that as anything but the same database, because it *is* the same
    database by every filesystem measure. The fix is not a better identity
    check: it is one sidecar per OS account, full stop. These tests would
    have failed against the pre-C7 design for ``other_db`` specifically —
    that design deliberately kept genuinely distinct databases on separate
    locks; unifying them onto one lock is exactly what changed.
    """

    def test_two_genuinely_distinct_databases_serialize_on_one_lock(
        self, tmp_path
    ):
        """The core behavior change: db_a and db_b share no inode, no
        directory, no name — nothing but the OS account. Under the old
        per-database design these would never have blocked each other."""
        db_a = tmp_path / "a" / "state.db"
        db_b = tmp_path / "b" / "state.db"
        db_a.parent.mkdir()
        db_b.parent.mkdir()
        db_a.touch()
        db_b.touch()

        started = threading.Event()
        release = threading.Event()

        def _hold_a():
            with acquire_state_write_lock(db_a, timeout_s=2.0) as admitted:
                assert admitted is True
                started.set()
                release.wait(5.0)

        holder = threading.Thread(target=_hold_a)
        holder.start()
        try:
            assert started.wait(2.0)
            t0 = time.monotonic()
            with acquire_state_write_lock(db_b, timeout_s=0.25) as admitted:
                assert admitted is False
            assert time.monotonic() - t0 < 1.5
        finally:
            release.set()
            holder.join(5.0)

        with acquire_state_write_lock(db_b, timeout_s=1.0) as admitted:
            assert admitted is True

    def test_distinct_db_symlink_hardlink_case_unicode_and_nonexistent_aliases_all_serialize(
        self, tmp_path
    ):
        """One holder on the real database; every other spelling/alias
        class this module used to canonicalize individually — plus a
        wholly unrelated second database and a path that does not exist at
        all — must all observe the same admission token."""
        real_db = tmp_path / "state.db"
        real_db.touch()

        candidates = []
        if _HAS_SYMLINK:
            symlink_alias = tmp_path / "alias.db"
            os.symlink(real_db, symlink_alias)
            candidates.append(symlink_alias)
        if _HAS_HARDLINK:
            hardlink_alias = tmp_path / "State.DB"
            os.link(real_db, hardlink_alias)
            candidates.append(hardlink_alias)

        other_db = tmp_path / "other" / "state.db"
        other_db.parent.mkdir()
        other_db.touch()
        candidates.append(other_db)

        unicode_db = tmp_path / unicodedata.normalize("NFC", "café.db")
        candidates.append(unicode_db)  # does not exist

        nonexistent_db = tmp_path / "does-not-exist-yet.db"
        candidates.append(nonexistent_db)

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
            for candidate in candidates:
                t0 = time.monotonic()
                with acquire_state_write_lock(candidate, timeout_s=0.25) as admitted:
                    assert admitted is False, (
                        f"{candidate} was not serialized behind the global lock"
                    )
                assert time.monotonic() - t0 < 1.5
        finally:
            release.set()
            holder.join(5.0)

        for candidate in candidates:
            with acquire_state_write_lock(candidate, timeout_s=1.0) as admitted:
                assert admitted is True

    def test_db_path_is_never_touched_on_the_filesystem(self, tmp_path):
        """A db_path whose parent directories do not even exist must still
        admit — proof nothing in the admission path stats, resolves, or
        otherwise touches db_path itself."""
        bogus = tmp_path / "does" / "not" / "exist" / "at" / "all.db"
        assert not bogus.parent.exists()
        with acquire_state_write_lock(bogus, timeout_s=1.0) as admitted:
            assert admitted is True
        assert not bogus.parent.exists()

    def test_write_lock_path_is_a_pure_function_of_the_lock_root_only(
        self, tmp_path
    ):
        assert write_lock_path() == write_lock_path()
        assert write_lock_path().parent == _lock_root()
        assert write_lock_path().name == "global.write.lock"


class TestLegacyColocationNoLongerMatters:
    """Nemo REQUEST_CHANGES (20260820T115805Z): a co-tenant with write
    access to the database's own directory could ``unlink`` the old sibling
    sidecar (``state.db.write.lock``) and drop in a brand-new, ordinary
    regular file under the same name between two acquires. Nothing in a
    fresh file's metadata differs from a sidecar this module would have
    created itself, so no ``fstat``-only check can distinguish them — a
    holder still flock'd on the old inode and a second acquirer that locks
    the new one would both believe they hold admission.

    The fix does not try to detect that swap; it removes the shared
    directory. These tests reproduce the exact swap against the DB's own
    (co-tenant-writable) directory and show it has no bearing on admission
    at all, because the real lock never lives there anymore."""

    def test_unlink_and_recreate_in_the_db_directory_does_not_bypass_the_lock(
        self, tmp_path
    ):
        db_path = tmp_path / "state.db"
        # Where the sidecar used to live, sibling to the database.
        legacy_sidecar = tmp_path / "state.db.write.lock"
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

            # A co-tenant with write access to tmp_path (the DB's own
            # directory) performs the exact bypass Nemo found: unlink
            # whatever is at the legacy sidecar name and drop in a fresh,
            # indistinguishable regular file under the same name — more
            # than once, to model repeated attempts across the hold.
            for _ in range(3):
                if legacy_sidecar.exists():
                    legacy_sidecar.unlink()
                legacy_sidecar.write_bytes(b"")

            # No effect: admission was never derived from this directory.
            t0 = time.monotonic()
            with acquire_state_write_lock(db_path, timeout_s=0.5) as admitted:
                assert admitted is False
            assert time.monotonic() - t0 < 1.5
        finally:
            release.set()
            holder.join(5.0)

        with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
            assert admitted is True

    def test_write_lock_path_never_lives_next_to_the_database(
        self, tmp_path, monkeypatch
    ):
        # Exercise the real default resolution, not the isolated per-test
        # root the autouse fixture points at for every other test here.
        monkeypatch.delenv("HERMES_STATE_LOCK_ROOT", raising=False)
        db_path = tmp_path / "state.db"
        lock_path = write_lock_path()
        assert tmp_path not in lock_path.parents
        assert lock_path.parent != db_path.parent


@pytest.mark.skipif(not _HAS_SYMLINK, reason="platform has no os.symlink")
class TestSidecarSymlinkSubstitution:
    """Now that the containing directory is private per-account, a co-tenant
    can no longer reach this path to swap it — this is defense-in-depth
    against a same-account race or bug: a sidecar swapped for a symlink must
    not be followed, only degrade to admitted."""

    def test_symlinked_sidecar_is_refused_not_followed(self, tmp_path):
        db_path = tmp_path / "state.db"
        target = tmp_path / "other-process.lock"
        target.write_bytes(b"")
        lock_path = write_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
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
    """Now that the containing directory is private per-account, a co-tenant
    can no longer reach this path to swap it — this is defense-in-depth
    against a same-account race or bug (a rotation script, a retry that
    reopens mid-hold): a hardlink swap onto the sidecar's name must still
    not grant concurrent admission while a holder is active, and — per the
    C5 remediation for 20260820T125928Z — a subsequent plain unlink+recreate
    of the same name must not let this same process quietly resume trusting
    it either, since that pattern is exactly what a same-account attacker
    would perform after the hardlink swap is noticed."""

    def test_hardlink_substitution_does_not_grant_concurrent_admission(
        self, tmp_path
    ):
        db_path = tmp_path / "state.db"
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
            lock_path = write_lock_path()

            # Simulate a same-account race swapping the sidecar for a
            # hardlink to an unrelated file while the holder still owns the
            # flock on the original inode.
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
        # would just be the same swap performed by trusted code); prove it
        # left the hardlink alone.
        lock_path = write_lock_path()
        victim = tmp_path / "victim.lock"
        assert os.path.samefile(lock_path, victim)

        # Cleaning up the tampering does NOT silently resume admission in
        # THIS process: this process's own memory of the sidecar's identity
        # (recorded when the holder thread first acquired it, above) now
        # disagrees with the fresh file that replaces the hardlinked one —
        # see _check_sidecar_continuity. That disagreement is direct
        # evidence of a same-account swap this process itself witnessed, so
        # it fails closed rather than re-trusting whatever now has the
        # name; converging back to admission requires a process restart
        # (an unpoisoned bootstrap), not just removing the rogue entry.
        os.unlink(lock_path)
        with acquire_state_write_lock(db_path, timeout_s=0.5) as admitted:
            assert admitted is False

        # A process that never held the original sidecar has no baseline to
        # disagree with, so it is unaffected by what this process witnessed
        # — bootstrap must keep working for everyone else.
        with pytest.MonkeyPatch.context() as mpatch:
            mpatch.setattr(hermes_state_lock, "_known_sidecar_identity", None)
            with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
                assert admitted is True


class TestSidecarIdentityContinuity:
    """C5 remediation for REQUEST_CHANGES 20260820T125928Z: cover accidental
    sidecar recreation without claiming a boundary this module cannot
    actually hold against malicious same-account code.

    Requirement 1 of that gate was an audit, not a test: neither this
    module nor ``hermes_state.py`` ever unlinks or recreates the sidecar on
    release or cleanup — ``acquire_state_write_lock`` releases by
    ``flock(LOCK_UN)`` + ``close()`` only (see the ``finally`` block), and
    ``_open_sidecar`` opens with ``O_CREAT`` (create-if-absent), never
    truncates or replaces an existing file. There was no cooperative
    delete/recreate path to remove. These tests pin that down as a
    regression guard, then exercise the one gap that *did* need a fix: a
    plain unlink+recreate (no hardlink, so ``st_nlink`` never moves off
    ``1``) is invisible to ``_sidecar_identity_is_trustworthy``'s
    single-attempt check, because a freshly-opened replacement is
    self-consistent from that one check's point of view. Only a second,
    independent memory — this process's own record of what it last
    trusted, added in ``_check_sidecar_continuity`` — can catch that."""

    def test_cooperative_release_reacquire_cycles_never_unlink_and_keep_one_inode(
        self, tmp_path
    ):
        """Purely cooperative use (no tampering): many sequential
        release/reacquire cycles on the same path must observe exactly one
        ``(st_dev, st_ino)`` throughout, and this module must never call
        ``os.unlink``/``os.remove`` on the sidecar itself."""
        db_path = tmp_path / "state.db"
        real_unlink = os.unlink
        unlinked_paths = []

        def _spying_unlink(path, *a, **kw):
            unlinked_paths.append(os.fspath(path))
            return real_unlink(path, *a, **kw)

        with pytest.MonkeyPatch.context() as mpatch:
            mpatch.setattr(os, "unlink", _spying_unlink)

            identities = set()
            for _ in range(25):
                with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
                    assert admitted is True
                    st = os.stat(write_lock_path())
                    identities.add((st.st_dev, st.st_ino))

        assert len(identities) == 1, (
            "the sidecar's identity must not change across cooperative "
            f"release/reacquire cycles, observed: {identities}"
        )
        lock_path = str(write_lock_path())
        assert lock_path not in unlinked_paths, (
            "acquire_state_write_lock must never unlink its own sidecar"
        )

    def test_cooperative_multi_process_cycles_never_unlink_and_keep_one_inode(
        self, tmp_path, _isolated_lock_root
    ):
        """Same invariant, but across real cooperative processes rather than
        one process's threads — the actual Gateway/Dashboard/ACP shape."""
        db_path = tmp_path / "state.db"
        lock_root = str(_isolated_lock_root)
        # Establish the sidecar before spawning so both children observe
        # the same starting identity, not two independent bootstraps.
        with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
            assert admitted is True
        baseline = os.stat(write_lock_path())
        baseline_identity = [baseline.st_dev, baseline.st_ino]

        ctx = mp.get_context("spawn")
        procs = []
        identity_paths = []
        err_paths = []
        for i in range(2):
            identity_path = tmp_path / f"identities-{i}.json"
            err = tmp_path / f"err-{i}"
            identity_paths.append(identity_path)
            err_paths.append(err)
            procs.append(
                _spawn(
                    ctx,
                    _child_cooperative_cycles,
                    (str(db_path), lock_root, 10, str(identity_path), str(err)),
                )
            )
        for proc, err in zip(procs, err_paths):
            _join_ok(proc, 30.0, err)

        all_identities = [baseline_identity]
        for identity_path in identity_paths:
            all_identities.extend(json.loads(identity_path.read_text(encoding="utf-8")))

        unique = {tuple(identity) for identity in all_identities}
        assert unique == {tuple(baseline_identity)}, (
            "two cooperative processes cycling the same lock must observe "
            f"exactly the pre-spawn identity throughout, observed: {unique}"
        )

    def test_bootstrap_first_ever_acquire_has_no_baseline_to_reject(
        self, tmp_path
    ):
        """Requirement 3's condition: the new check must not break the very
        first acquire, where there is nothing yet to compare against."""
        db_path = tmp_path / "state.db"
        assert hermes_state_lock._known_sidecar_identity is None
        with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
            assert admitted is True
        assert hermes_state_lock._known_sidecar_identity is not None

    def test_plain_unlink_and_recreate_between_cycles_is_rejected(self, tmp_path):
        """RED before the continuity check existed: a plain unlink+recreate
        (no hardlink — ``st_nlink`` stays ``1`` throughout) between two
        non-overlapping acquires, with no concurrent holder at all, is
        exactly the accidental-recreation shape (a cleanup script, a
        ``tmpfiles.d``-style sweep, a bug) this gate exists to cover.
        ``_sidecar_identity_is_trustworthy`` alone cannot see this — the
        replacement is internally self-consistent — so this is GREEN only
        because ``_check_sidecar_continuity`` now compares against this
        process's own prior observation."""
        db_path = tmp_path / "state.db"
        with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
            assert admitted is True

        lock_path = write_lock_path()
        original = os.stat(lock_path)
        os.unlink(lock_path)
        lock_path.write_bytes(b"")  # ordinary regular file, nlink == 1
        replacement = os.stat(lock_path)
        assert replacement.st_ino != original.st_ino or (
            replacement.st_dev != original.st_dev
        ), "the replacement must actually be a different inode for this test to mean anything"
        assert replacement.st_nlink == 1, (
            "no hardlink involved — the nlink-based check must not be what "
            "catches this"
        )

        with acquire_state_write_lock(db_path, timeout_s=0.5) as admitted:
            assert admitted is False

    def test_matching_identity_on_reacquire_does_not_raise(self):
        """Sanity check on the mechanism itself, independent of the public
        context-manager API: an unchanged identity across two calls must
        not raise."""
        identity = (1, 42)
        hermes_state_lock._check_sidecar_continuity(identity)
        hermes_state_lock._check_sidecar_continuity(identity)  # no raise

    def test_differing_identity_on_recheck_raises_and_keeps_original_baseline(
        self,
    ):
        """Sanity check on the mechanism itself: a disagreeing identity
        raises, and the *original* identity remains the recorded baseline
        afterward — the mismatch is never "healed" onto the new value,
        which is what keeps a still-live swap from being re-trusted on the
        very next attempt."""
        original = (1, 42)
        swapped = (1, 99)
        hermes_state_lock._check_sidecar_continuity(original)
        with pytest.raises(hermes_state_lock._SidecarIdentitySwapped):
            hermes_state_lock._check_sidecar_continuity(swapped)
        assert hermes_state_lock._known_sidecar_identity == original
        # And the disagreement keeps being reported, not just the first time.
        with pytest.raises(hermes_state_lock._SidecarIdentitySwapped):
            hermes_state_lock._check_sidecar_continuity(swapped)


class TestPrivateLockRootInvariants:
    """POSIX: the private lock root must be created 0700, owned by the
    current account, and never a symlink — re-verified on every acquire.
    Failing any of those is fail-closed (admitted is False), not a silent
    degrade: an unverified root is exactly the co-tenant-writable-directory
    condition this design exists to remove."""

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX-only invariants")
    def test_root_created_fresh_is_0700_and_owned_by_current_user(
        self, tmp_path
    ):
        db_path = tmp_path / "state.db"
        with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
            assert admitted is True
        root = write_lock_path().parent
        st = os.lstat(root)
        assert not stat.S_ISLNK(st.st_mode)
        assert stat.S_ISDIR(st.st_mode)
        assert stat.S_IMODE(st.st_mode) == 0o700
        assert st.st_uid == os.getuid()

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX-only invariants")
    def test_root_owned_by_a_different_uid_fails_closed(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "state.db"
        # Establish the root for real first, owned by us.
        with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
            assert admitted is True

        real_getuid = os.getuid
        monkeypatch.setattr(os, "getuid", lambda: real_getuid() + 1)
        with acquire_state_write_lock(db_path, timeout_s=0.5) as admitted:
            assert admitted is False

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX-only invariants")
    def test_root_with_uncorrectable_permissions_fails_closed(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "state.db"
        with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
            assert admitted is True
        root = write_lock_path().parent
        os.chmod(root, 0o755)  # loosen it, simulating drift or tampering

        real_chmod = os.chmod

        def _refuse_chmod(path, mode, *a, **kw):
            if Path(path) == root:
                raise PermissionError("simulated: cannot tighten")
            return real_chmod(path, mode, *a, **kw)

        monkeypatch.setattr(os, "chmod", _refuse_chmod)
        with acquire_state_write_lock(db_path, timeout_s=0.5) as admitted:
            assert admitted is False

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX-only invariants")
    def test_root_with_correctable_permissions_self_heals(self, tmp_path):
        db_path = tmp_path / "state.db"
        with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
            assert admitted is True
        root = write_lock_path().parent
        os.chmod(root, 0o755)

        with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
            assert admitted is True
        st = os.lstat(root)
        assert stat.S_IMODE(st.st_mode) == 0o700

    @pytest.mark.skipif(not _HAS_SYMLINK, reason="platform has no os.symlink")
    def test_root_replaced_by_a_symlink_fails_closed_and_is_left_alone(
        self, tmp_path
    ):
        db_path = tmp_path / "state.db"
        with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
            assert admitted is True
        root = write_lock_path().parent
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        shutil.rmtree(root)
        os.symlink(elsewhere, root)

        with acquire_state_write_lock(db_path, timeout_s=0.5) as admitted:
            assert admitted is False
        # Never followed, never replaced.
        assert root.is_symlink()
        assert os.readlink(root) == str(elsewhere)

    def test_cannot_create_root_at_all_degrades_to_admitted(
        self, tmp_path, monkeypatch
    ):
        """Distinct from the invariant failures above: a plain inability to
        create the root (parent missing, not a directory, read-only fs) is
        operational, not a tampering signal, so it degrades the way an
        unopenable lock file always has."""
        db_path = tmp_path / "state.db"
        blocker = tmp_path / "blocker"
        blocker.write_bytes(b"")  # a file, not a directory
        monkeypatch.setenv("HERMES_STATE_LOCK_ROOT", str(blocker / "root"))
        with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
            assert admitted is True


class TestLockRootPlatformSplit:
    def test_posix_root_is_uid_scoped_under_tempdir(self, monkeypatch):
        monkeypatch.delenv("HERMES_STATE_LOCK_ROOT", raising=False)
        root = _posix_lock_root()
        assert str(os.getuid()) in root.name
        assert root.parent == Path(tempfile.gettempdir())

    def test_windows_root_is_under_localappdata(self, monkeypatch):
        monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\jose\\AppData\\Local")
        root = _windows_lock_root()
        assert root == Path("C:\\Users\\jose\\AppData\\Local") / "hermes" / "state-locks"

    def test_windows_root_falls_back_without_localappdata(self, monkeypatch):
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        root = _windows_lock_root()
        assert root == Path.home() / "AppData" / "Local" / "hermes" / "state-locks"

    def test_lock_root_override_env_var_takes_precedence(
        self, tmp_path, monkeypatch
    ):
        override = tmp_path / "custom-root"
        monkeypatch.setenv("HERMES_STATE_LOCK_ROOT", str(override))
        assert _lock_root(is_windows=False) == override
        assert _lock_root(is_windows=True) == override


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
