"""Cross-process write admission for the shared state.db.

Gateway, Dashboard and ACP share one SQLite file.  SQLite's busy handler is
not a queue, so a sibling's schema init can starve create_session /
append_message until their patience budget expires (2026-08-19 vpsclone:
Dashboard ``_run_init_schema_with_wide_busy_timeout`` vs Gateway 60 s
watchdog).  These tests use real child processes and a temporary database
— never the live profile store.

Every test in this module gets its own isolated private lock root under
``tmp_path`` (see the ``_isolated_lock_root`` autouse fixture below) — never
the real per-account root (``/tmp/hermes-state-locks-<uid>`` on POSIX) that
any live Hermes process on the same machine, under the same OS account,
might already be relying on.
"""

from __future__ import annotations

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
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_lock import (
    _CASE_PROBE_PREFIX,
    _is_case_insensitive_fs,
    _lock_open_flags,
    _lock_root,
    _posix_lock_root,
    _windows_lock_root,
    acquire_state_write_lock,
    canonical_db_key,
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


@pytest.mark.skipif(not _HAS_SYMLINK, reason="platform has no os.symlink")
class TestSymlinkAliasSharesLock:
    """Nemo REQUEST_CHANGES (20260820T042052Z): write_lock_path() derived the
    sidecar from the raw path while canonical_db_key() resolved symlinks, so
    ``state.db`` and ``alias.db -> state.db`` locked different sidecars and
    two processes could hold admission on the same underlying SQLite file at
    once."""

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


@pytest.mark.skipif(_IS_WINDOWS, reason="normcase already folds case on Windows")
class TestCaseInsensitiveAliasSharesLock:
    """Gate 20260820T124010Z: ``canonical_db_key()`` folded case with
    ``os.path.normcase``, but ``normcase`` is the identity function on every
    POSIX platform, including macOS — regardless of whether the underlying
    volume is case-insensitive. macOS's default APFS/HFS+ volumes *are*
    case-insensitive (case-preserving), so ``State.db`` and ``state.db`` can
    be the very same on-disk file there, yet the old ``canonical_db_key``
    hashed them to two different keys and split admission across two
    sidecars — the same class of bug ``TestSymlinkAliasSharesLock`` covers
    for symlinks, just reached through case instead of a link.

    A genuinely case-insensitive volume isn't available on every runner this
    suite executes on, but the property under test — two spellings that name
    the same file must canonicalize identically — doesn't require one: a
    hardlink from a differently-cased name onto the same inode reproduces
    exactly what a case-insensitive lookup resolves to, without depending on
    the host filesystem's own case sensitivity. The alias is built with
    ``str.swapcase()`` (not a single flipped letter) because that is exactly
    what :func:`hermes_state_lock._is_case_insensitive_fs` probes with —
    matching it exactly is what lets the hardlink stand in for a real
    case-insensitive volume, where *every* case spelling of the name
    (single-letter or fully swapped) would land on the one directory entry.
    """

    @pytest.mark.skipif(not _HAS_HARDLINK, reason="platform has no os.link")
    def test_write_lock_path_matches_for_differently_cased_alias(self, tmp_path):
        real_db = tmp_path / "state.db"
        real_db.touch()
        cased_alias = tmp_path / real_db.name.swapcase()
        os.link(real_db, cased_alias)

        assert write_lock_path(real_db) == write_lock_path(cased_alias)
        assert canonical_db_key(real_db) == canonical_db_key(cased_alias)

    @pytest.mark.skipif(not _HAS_HARDLINK, reason="platform has no os.link")
    def test_alias_writer_is_blocked_by_differently_cased_holder(self, tmp_path):
        real_db = tmp_path / "state.db"
        real_db.touch()
        cased_alias = tmp_path / real_db.name.swapcase()
        os.link(real_db, cased_alias)

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
            # A different thread going through the differently-cased alias
            # must see the same admission token as the real path and time
            # out while it is held.
            t0 = time.monotonic()
            with acquire_state_write_lock(cased_alias, timeout_s=0.25) as admitted:
                assert admitted is False
            assert time.monotonic() - t0 < 1.5
        finally:
            release.set()
            holder.join(5.0)

        with acquire_state_write_lock(cased_alias, timeout_s=1.0) as admitted:
            assert admitted is True

    def test_genuinely_distinct_files_differing_only_in_case_are_not_merged(
        self, tmp_path
    ):
        """The probe must not report false positives: two *separate* files
        that happen to differ only in case (no hardlink, no symlink between
        them) must keep separate keys — merging them would let a writer on
        one file believe it holds admission over the other."""
        lower_db = tmp_path / "state.db"
        lower_db.write_bytes(b"lower")
        upper_db = tmp_path / "State.db"
        upper_db.write_bytes(b"upper")

        assert not _is_case_insensitive_fs(str(lower_db))
        assert canonical_db_key(lower_db) != canonical_db_key(upper_db)
        assert write_lock_path(lower_db) != write_lock_path(upper_db)

    def test_basename_with_no_cased_characters_is_not_probed_as_insensitive(
        self, tmp_path
    ):
        digits_db = tmp_path / "12345.db"
        digits_db.touch()
        assert _is_case_insensitive_fs(str(digits_db)) is False

    def test_missing_path_probes_the_parent_directory_instead_of_giving_up(
        self, tmp_path
    ):
        """On this (case-sensitive) test filesystem, probing the existing
        parent directory of a not-yet-created database still correctly
        answers "not case-insensitive" — but it now gets there by actually
        probing ``tmp_path``, not by short-circuiting on ENOENT. See
        ``TestNewDatabaseCaseInsensitiveAliasConverges`` for the case this
        distinction exists to fix."""
        missing = tmp_path / "does-not-exist.db"
        assert _is_case_insensitive_fs(str(missing)) is False


def _make_case_insensitive_lookup(monkeypatch, directory):
    """Simulate a case-insensitive/case-preserving directory (e.g. default
    APFS) for *directory* without needing one actually mounted.

    Wraps ``os.lstat`` so that a lookup for a name that doesn't exist
    verbatim in *directory* falls back to a case-insensitive match among
    that directory's real entries — exactly what a case-folding filesystem's
    own lookup does. Every other path (including any name that already
    exists verbatim, and every path outside *directory*) goes straight to
    the real ``os.lstat`` untouched, so this cannot affect unrelated tests
    or unrelated directories running in the same process.

    This targets ``os.lstat`` specifically because that is the only syscall
    ``_probe_directory_case_folds`` uses to resolve the swapped-case name —
    the probe file itself is created with its one real, randomly generated
    name via ``os.open``, so there is exactly one genuine directory entry
    for the shim to fold onto.
    """
    real_lstat = os.lstat
    directory = os.path.realpath(str(directory))

    def fake_lstat(path, *args, **kwargs):
        fspath = os.fspath(path)
        try:
            return real_lstat(fspath, *args, **kwargs)
        except FileNotFoundError:
            parent, name = os.path.split(fspath)
            if os.path.realpath(parent or ".") != directory:
                raise
            try:
                entries = os.listdir(directory)
            except OSError:
                raise
            for entry in entries:
                if entry.lower() == name.lower():
                    return real_lstat(os.path.join(directory, entry), *args, **kwargs)
            raise

    monkeypatch.setattr(os, "lstat", fake_lstat)


@pytest.mark.skipif(_IS_WINDOWS, reason="normcase already folds case on Windows")
class TestNewDatabaseCaseInsensitiveAliasConverges:
    """Gate 20260820T125109Z REQUEST_CHANGES: ``_is_case_insensitive_fs``
    returned ``False`` unconditionally whenever the target database did not
    exist yet (``ENOENT``). On a genuinely case-insensitive volume that is
    exactly backwards for the case that matters most — schema init on a
    brand-new ``state.db`` — because two sibling processes racing to create
    it through differently-cased spellings (``state.db`` vs ``State.db``)
    would each probe ENOENT, each conclude "not proven case-insensitive",
    and each hash to a *different* lock key, so both would believe they hold
    exclusive admission over what the filesystem resolves to one file.

    A real case-insensitive volume isn't available on every runner this
    suite executes on, so ``_make_case_insensitive_lookup`` reproduces what
    such a volume's own lookup does — case-insensitive fallback resolution
    within one directory — without depending on the host filesystem.
    """

    def test_canonical_db_key_converges_for_case_alias_of_nonexistent_db(
        self, tmp_path, monkeypatch
    ):
        _make_case_insensitive_lookup(monkeypatch, tmp_path)
        real_db = tmp_path / "state.db"
        cased_alias = tmp_path / "State.db"

        assert not real_db.exists()
        assert not cased_alias.exists()
        assert canonical_db_key(real_db) == canonical_db_key(cased_alias)

    def test_write_lock_path_converges_for_case_alias_of_nonexistent_db(
        self, tmp_path, monkeypatch
    ):
        _make_case_insensitive_lookup(monkeypatch, tmp_path)
        real_db = tmp_path / "state.db"
        cased_alias = tmp_path / "State.db"

        assert write_lock_path(real_db) == write_lock_path(cased_alias)

    def test_concurrent_schema_init_via_case_alias_is_serialized_by_the_lock(
        self, tmp_path, monkeypatch
    ):
        """The actual failure mode this gate exists to close: two "sibling
        processes" (here, threads) racing schema init on a not-yet-created
        database through differently-cased spellings must be admitted as one
        writer at a time, not two — mirrors
        ``TestCaseInsensitiveAliasSharesLock`` but for a database neither
        side has created yet."""
        _make_case_insensitive_lookup(monkeypatch, tmp_path)
        real_db = tmp_path / "state.db"
        cased_alias = tmp_path / "State.db"
        assert not real_db.exists()
        assert not cased_alias.exists()

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
            t0 = time.monotonic()
            with acquire_state_write_lock(cased_alias, timeout_s=0.25) as admitted:
                assert admitted is False
            assert time.monotonic() - t0 < 1.5
        finally:
            release.set()
            holder.join(5.0)

        with acquire_state_write_lock(cased_alias, timeout_s=1.0) as admitted:
            assert admitted is True

    def test_case_sensitive_directory_keeps_nonexistent_aliases_separate(
        self, tmp_path
    ):
        """No shim here: on this (real, case-sensitive) test filesystem, two
        differently-cased spellings of a database neither side has created
        yet must keep separate keys and must NOT block each other — folding
        case on a genuinely case-sensitive filesystem would be the exact
        false-merge ``test_genuinely_distinct_files_differing_only_in_case_
        are_not_merged`` guards against, just reached via ENOENT instead of
        two already-existing files."""
        real_db = tmp_path / "state.db"
        cased_alias = tmp_path / "State.db"
        assert not real_db.exists()
        assert not cased_alias.exists()

        assert canonical_db_key(real_db) != canonical_db_key(cased_alias)
        assert write_lock_path(real_db) != write_lock_path(cased_alias)

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
            # Separate keys: the alias must be admitted immediately, not
            # time out waiting on the real path's holder.
            with acquire_state_write_lock(cased_alias, timeout_s=1.0) as admitted:
                assert admitted is True
        finally:
            release.set()
            holder.join(5.0)


@pytest.mark.skipif(_IS_WINDOWS, reason="normcase already folds case on Windows")
class TestDirectoryCaseProbeHygiene:
    """``_probe_directory_case_folds`` (the fallback ``_is_case_insensitive_fs``
    takes when the target database doesn't exist yet) creates a real,
    if short-lived, file on disk. These tests pin down the safety properties
    that make that acceptable: it never leaks the probe file, it never
    touches a name it didn't create itself, and it degrades to "cannot
    prove" rather than raising or hanging when it can't complete."""

    def test_probe_leaves_no_file_behind_on_a_case_sensitive_directory(
        self, tmp_path
    ):
        before = set(os.listdir(tmp_path))
        missing = tmp_path / "does-not-exist.db"

        assert _is_case_insensitive_fs(str(missing)) is False
        assert set(os.listdir(tmp_path)) == before

    def test_probe_leaves_no_file_behind_on_a_case_insensitive_directory(
        self, tmp_path, monkeypatch
    ):
        _make_case_insensitive_lookup(monkeypatch, tmp_path)
        before = set(os.listdir(tmp_path))
        missing = tmp_path / "does-not-exist.db"

        assert _is_case_insensitive_fs(str(missing)) is True
        assert set(os.listdir(tmp_path)) == before

    def test_probe_never_deletes_a_preexisting_swapped_case_file(self, tmp_path):
        """If a file that happens to collide with a probe's swapped-case
        name already exists for unrelated reasons, the probe must leave it
        untouched — it may only ever unlink the exact name it created."""
        real_lstat = os.lstat
        swapped_prefix = _CASE_PROBE_PREFIX.swapcase()
        probe_swapped_names = []

        def spying_lstat(path, *args, **kwargs):
            fspath = os.fspath(path)
            name = os.path.basename(fspath)
            if name.startswith(swapped_prefix):
                probe_swapped_names.append(fspath)
            return real_lstat(fspath, *args, **kwargs)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "lstat", spying_lstat)
            missing = tmp_path / "does-not-exist.db"
            assert _is_case_insensitive_fs(str(missing)) is False

        # The probe's swapped-name candidate was looked up (lstat'd)...
        assert probe_swapped_names
        # ...but since nothing was ever created under that name, nothing
        # exists there afterward either.
        for candidate in probe_swapped_names:
            assert not os.path.lexists(candidate)

    def test_probe_degrades_to_false_when_every_candidate_name_collides(
        self, tmp_path, monkeypatch
    ):
        """A pathological (astronomically unlikely in practice) run of
        random-name collisions must not hang or crash the caller — it must
        give up after a bounded number of attempts and report "cannot
        prove"."""

        def always_collides(*_args, **_kwargs):
            raise FileExistsError("simulated collision")

        monkeypatch.setattr(os, "open", always_collides)
        missing = tmp_path / "does-not-exist.db"

        assert _is_case_insensitive_fs(str(missing)) is False

    def test_probe_degrades_to_false_when_directory_is_unwritable(self, tmp_path):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            pytest.skip("root bypasses directory write permission checks")

        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o500)
        try:
            missing = readonly_dir / "does-not-exist.db"
            assert _is_case_insensitive_fs(str(missing)) is False
        finally:
            readonly_dir.chmod(0o700)

    def test_probe_degrades_to_false_when_parent_directory_is_missing(
        self, tmp_path
    ):
        missing = tmp_path / "no-such-parent" / "does-not-exist.db"
        assert _is_case_insensitive_fs(str(missing)) is False

    def test_probe_uses_swapcase_not_a_fixed_case_conversion(self, tmp_path):
        """Sanity check on the probe's own naming: ``_CASE_PROBE_PREFIX``
        must actually contain cased characters, or every probe attempt would
        degenerate into the swapped==original no-op guard and this whole
        fallback would be dead code."""
        assert _CASE_PROBE_PREFIX.swapcase() != _CASE_PROBE_PREFIX


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
        lock_path = write_lock_path(db_path)
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
        lock_path = write_lock_path(db_path)
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
    not grant concurrent admission while a holder is active."""

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
            lock_path = write_lock_path(db_path)

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
        lock_path = write_lock_path(db_path)
        victim = tmp_path / "victim.lock"
        assert os.path.samefile(lock_path, victim)

        # Once the tampering is cleaned up (an operator removing the rogue
        # entry, as would happen operationally), normal admission resumes.
        os.unlink(lock_path)
        with acquire_state_write_lock(db_path, timeout_s=1.0) as admitted:
            assert admitted is True


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
        root = write_lock_path(db_path).parent
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
        root = write_lock_path(db_path).parent
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
        root = write_lock_path(db_path).parent
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
        root = write_lock_path(db_path).parent
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
