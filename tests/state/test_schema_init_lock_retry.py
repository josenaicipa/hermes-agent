"""``SessionDB()`` construction must survive writer-lock contention.

Context (vpsclone, 2026-08-03 13:51): a window of SQLite single-writer
contention on a 13.4 GB ``state.db`` produced ``database is locked`` across
unrelated sessions.  The visible main-turn failure was an ``append_message``
on an ALREADY-OPEN handle giving up after ~16.2 s — that path is not what
this module changes, and the original lock holder was never forensically
logged.

What these tests do pin is a separate, demonstrated defect found while
diagnosing it: the two write paths in :class:`SessionDB` had very different
contention budgets.

  * every write via ``_execute_write``: BEGIN IMMEDIATE + 15 jittered
    retries of the lock-contention class => measured ~16 s of patience;
  * ``_init_schema``: writes directly on the connection, bypassing
    ``_execute_write``, so its only tolerance was the connection's 1 s
    ``busy_timeout`` => measured ~2 s before ``SessionDB()`` raised.

So a short contention window that every other writer rode out could still
fail construction, and callers that treat a construction failure as "session
store unavailable" lose the features backed by it.  A second, nested Hermes
process running a different release made such concurrent opens more likely
during the incident window; it is correlated, not proven to be the original
writer.

The fix shares one bounded, jittered lock-retry budget between both paths,
retries ONLY the lock-contention class, and still fails closed once that
budget is exhausted.
"""

import sqlite3
import threading
import time
from unittest.mock import patch

import pytest

import hermes_state
from hermes_state import SessionDB, is_lock_contention_error


@pytest.fixture(autouse=True)
def _reset_last_init_error():
    hermes_state._set_last_init_error(None)
    yield
    hermes_state._set_last_init_error(None)


# ---------------------------------------------------------------------------
# The retryable error class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc, expected",
    [
        (sqlite3.OperationalError("database is locked"), True),
        (sqlite3.OperationalError("database table is locked: messages"), True),
        (sqlite3.OperationalError("database is busy"), True),
        # Everything else means the statement may have done work, or that
        # retrying cannot help.  These must propagate on the first attempt.
        (sqlite3.OperationalError("no such table: messages"), False),
        (sqlite3.OperationalError("attempt to write a readonly database"), False),
        (sqlite3.DatabaseError("database disk image is malformed"), False),
        (sqlite3.IntegrityError("UNIQUE constraint failed"), False),
        (ValueError("not a sqlite error"), False),
    ],
)
def test_lock_contention_error_class(exc, expected):
    assert is_lock_contention_error(exc) is expected


# ---------------------------------------------------------------------------
# Schema init retries the lock class
# ---------------------------------------------------------------------------


def test_schema_init_retries_until_the_lock_clears(tmp_path):
    """A busy window that clears within budget must NOT fail construction.

    Before the fix the first lock error aborted ``SessionDB()``.
    """
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()  # materialise the schema once

    calls = {"n": 0}
    real_init = SessionDB._init_schema

    def _flaky_init(self):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise sqlite3.OperationalError("database is locked")
        return real_init(self)

    with patch.object(SessionDB, "_init_schema", _flaky_init):
        db = SessionDB(db_path=db_path)

    assert calls["n"] == 4, "must retry the lock class, not give up on it"
    # The store is fully usable — not a degraded/None fallback.
    db.create_session("S1", "cli")
    assert isinstance(db.append_message("S1", "user", "hello"), int)
    db.close()


def test_schema_init_does_not_retry_other_errors(tmp_path):
    """Non-lock errors keep today's fail-fast behaviour exactly."""
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()

    calls = {"n": 0}

    def _broken_init(self):
        calls["n"] += 1
        raise sqlite3.OperationalError("attempt to write a readonly database")

    with patch.object(SessionDB, "_init_schema", _broken_init):
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            SessionDB(db_path=db_path)

    assert calls["n"] == 1, "a non-lock error must not be retried"


def test_schema_init_still_fails_closed_when_budget_is_exhausted(tmp_path):
    """Fail-closed is preserved: a permanent lock still raises, bounded."""
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()

    calls = {"n": 0}

    def _always_locked(self):
        calls["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    # Neutralise the sleeps so the bound is asserted, not waited out.
    with patch.object(SessionDB, "_init_schema", _always_locked), \
         patch.object(hermes_state.time, "sleep", lambda _s: None):
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            SessionDB(db_path=db_path)

    assert calls["n"] == SessionDB._WRITE_MAX_RETRIES, (
        "retries must be bounded by the same budget as _execute_write"
    )
    # The cause is still recorded for /resume-style error strings.
    assert "locked" in (hermes_state.get_last_init_error() or "")


def test_schema_init_retry_sleeps_are_jittered_and_bounded(tmp_path):
    """Backoff reuses _execute_write's jitter window (no new tuning knob)."""
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()

    sleeps: list[float] = []

    def _always_locked(self):
        raise sqlite3.OperationalError("database is locked")

    with patch.object(SessionDB, "_init_schema", _always_locked), \
         patch.object(hermes_state.time, "sleep", sleeps.append):
        with pytest.raises(sqlite3.OperationalError):
            SessionDB(db_path=db_path)

    assert len(sleeps) == SessionDB._WRITE_MAX_RETRIES - 1
    assert all(
        SessionDB._WRITE_RETRY_MIN_S <= s <= SessionDB._WRITE_RETRY_MAX_S
        for s in sleeps
    )
    assert len(set(sleeps)) > 1, "jitter must stagger competing openers"


# ---------------------------------------------------------------------------
# End to end against a real competing writer
# ---------------------------------------------------------------------------


def test_open_survives_a_real_concurrent_write_transaction(tmp_path):
    """Another connection holds BEGIN IMMEDIATE while we construct.

    Before the fix this raised ``database is locked`` out of ``__init__``
    after ~2 s, so the caller got no session store at all; an already-open
    handle would have ridden out the same window.
    """
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()

    holder_ready = threading.Event()
    release = threading.Event()
    hold_seconds = 3.0

    def _holder():
        conn = sqlite3.connect(str(db_path), timeout=1.0, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES ('probe', '1') "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
            )
            holder_ready.set()
            release.wait(30)
            conn.rollback()
        finally:
            conn.close()

    thread = threading.Thread(target=_holder, daemon=True)
    thread.start()
    try:
        assert holder_ready.wait(10), "holder never acquired the write lock"
        timer = threading.Timer(hold_seconds, release.set)
        timer.start()
        try:
            started = time.monotonic()
            db = SessionDB(db_path=db_path)
        finally:
            timer.cancel()
            release.set()
        waited = time.monotonic() - started
    finally:
        release.set()
        thread.join(10)

    # It waited out the holder rather than dying at the 1 s busy timeout...
    assert waited >= hold_seconds - 0.5, (
        f"expected to wait out the {hold_seconds}s holder, waited {waited:.2f}s"
    )
    # ...and produced a working store, so durability is never silently lost.
    db.create_session("S2", "cli")
    assert isinstance(db.append_message("S2", "user", "persisted"), int)
    assert [m["content"] for m in db.get_messages("S2")] == ["persisted"]
    db.close()


def test_execute_write_budget_is_unchanged(tmp_path):
    """The normal write path keeps its exact retry contract."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("S3", "cli")

    attempts = {"n": 0}
    sleeps: list[float] = []

    def _always_locked(_conn):
        attempts["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    with patch.object(hermes_state.time, "sleep", sleeps.append):
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            db._execute_write(_always_locked)

    assert attempts["n"] == SessionDB._WRITE_MAX_RETRIES
    assert len(sleeps) == SessionDB._WRITE_MAX_RETRIES - 1
    db.close()
