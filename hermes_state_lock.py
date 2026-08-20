"""Cross-process write admission for the shared ``state.db``.

SQLite WAL allows many readers and exactly one writer.  Its busy handler is
not a queue: blocked writers re-probe, and whoever probes at the right
instant wins.  Gateway, Dashboard and ACP are separate processes against the
same file, so that unfairness is visible as ``database is locked`` on
``create_session`` / ``append_message`` while a sibling's schema init (or
another writer) holds the lock — the 2026-08-19 vpsclone incident.

An in-process gate cannot fix that class.  This module is the cross-process
half: one ``flock``/``msvcrt.locking`` admission token per canonical database
path, acquired around schema init and around each ``BEGIN IMMEDIATE``.

Why ``flock`` (and not a pidfile):

* The kernel drops the lock when the holding process dies, so a crash cannot
  leave an orphan that wedges every later writer.
* Readers never take it, so WAL concurrent reads stay intact.
* The lock file is a sibling sidecar (``state.db.write.lock``); the database
  file, WAL and SHM are untouched.

The lock is **not** a substitute for SQLite's own locking.  Callers still
``BEGIN IMMEDIATE`` and still retry on ``SQLITE_BUSY`` from holders that do
not go through this module (``sqlite3`` CLI, mixed-version processes).
Admission is released before those retries so we never hold the token while
waiting on an ungated writer.

Per-thread re-entrant: a nested write on the same path (reconnect during a
failed write, a helper that writes again) must not block on its own flock.
Other threads and other processes wait.

If the lock file cannot be created (read-only directory, exhausted fds) the
acquire degrades to admitted — the in-process ``SessionDB._lock`` plus SQLite
busy handling remain, which is what shipped before this module existed.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("hermes_state")

_IS_WINDOWS = sys.platform == "win32"
_POLL_S = 0.05
_WRITE_LOCK_SUFFIX = ".write.lock"

# thread ident -> {canonical key: _Hold}.  Depth is per-thread so a nested
# acquire on the same path does not open a second fd and self-deadlock
# (Linux flock is per open-file-description; two fds of the same path block).
_tls = threading.local()


class _Hold:
    __slots__ = ("handle", "depth")

    def __init__(self, handle) -> None:
        self.handle = handle
        self.depth = 1


def canonical_db_key(db_path: os.PathLike | str) -> str:
    """Canonicalize a database path so aliases share one lock.

    ``realpath`` collapses symlinks and relative spellings; ``normcase``
    folds case on filesystems where the OS does (macOS, Windows).
    """
    text = str(db_path)
    try:
        return os.path.normcase(os.path.realpath(text))
    except OSError:
        return os.path.normcase(text)


def write_lock_path(db_path: os.PathLike | str) -> Path:
    """Return the sidecar lock path for *db_path* (``<name>.write.lock``)."""
    path = Path(db_path)
    return path.with_name(path.name + _WRITE_LOCK_SUFFIX)


def _holds() -> dict:
    holds = getattr(_tls, "holds", None)
    if holds is None:
        holds = {}
        _tls.holds = holds
    return holds


def _try_exclusive(handle) -> bool:
    """Non-blocking exclusive lock.  True if this handle now holds it."""
    try:
        if _IS_WINDOWS:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def _unlock(handle) -> None:
    try:
        if _IS_WINDOWS:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def _reset_after_fork() -> None:
    """Drop inherited holds in a forked child without unlocking the parent.

    ``fork`` duplicates fds onto the same open-file-description, so
    ``LOCK_UN`` here would release the parent's admission.  Closing the
    inherited fd is enough: the parent still holds the description.
    """
    holds = getattr(_tls, "holds", None) or {}
    _tls.holds = {}
    for hold in holds.values():
        try:
            hold.handle.close()
        except Exception:
            pass


if hasattr(os, "register_at_fork"):  # pragma: no branch - POSIX only
    os.register_at_fork(after_in_child=_reset_after_fork)


@contextlib.contextmanager
def acquire_state_write_lock(
    db_path: os.PathLike | str,
    *,
    timeout_s: float = 1.0,
) -> Iterator[bool]:
    """Admit this thread as the Hermes writer for *db_path*.

    Yields ``True`` when this thread holds admission (including a nested
    re-acquire, and including the degrade-open path).  Yields ``False`` when
    the bounded wait expired; the caller has not touched SQLite and must
    treat it as contention.

    A non-positive *timeout_s* still performs one non-blocking attempt so a
    caller whose budget is already spent can make one last honest try.
    """
    key = canonical_db_key(db_path)
    holds = _holds()
    existing: Optional[_Hold] = holds.get(key)
    if existing is not None:
        existing.depth += 1
        try:
            yield True
        finally:
            existing.depth -= 1
        return

    lock_path = write_lock_path(db_path)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
    except OSError as exc:
        logger.warning(
            "Could not open state.db write lock %s (%s) — proceeding with "
            "SQLite busy handling only.",
            lock_path,
            exc,
        )
        yield True
        return

    deadline = time.monotonic() + max(0.0, timeout_s)
    acquired = False
    try:
        while True:
            if _try_exclusive(handle):
                acquired = True
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(_POLL_S, remaining))

        if not acquired:
            yield False
            return

        holds[key] = _Hold(handle)
        released = False
        try:
            yield True
        finally:
            hold = holds.get(key)
            if hold is not None:
                hold.depth -= 1
                if hold.depth <= 0:
                    holds.pop(key, None)
                    _unlock(handle)
                    try:
                        handle.close()
                    except OSError:
                        pass
                    released = True
        if released:
            handle = None
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass