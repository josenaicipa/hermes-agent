"""``SessionDB()`` construction under SQLite writer-lock contention.

Context (vpsclone, 2026-08-03 13:51): writes failed with ``database is
locked``.  That immediate cause is proven; the original lock holder was never
forensically logged.  The visible main-turn failure was an ``append_message``
on an ALREADY-OPEN handle giving up after ~16.2 s — this module does not
change that path and does not claim to fix it.

What it does cover is a separate, demonstrated constructor-side defect found
while diagnosing that incident: the two write paths had very different
contention budgets.

  * writes via ``_execute_write``: BEGIN IMMEDIATE + 15 jittered retries of
    the lock-contention text class => measured ~16 s of patience;
  * ``_init_schema``: writes directly on the connection, bypassing
    ``_execute_write``, so its only tolerance was the connection's 1 s
    ``busy_timeout`` => measured ~2 s before ``SessionDB()`` raised.

Closing that gap by RETRYING the whole ``_init_schema`` pass is unsafe, and
these tests pin why.  Several migrations are multi-statement rebuilds —
``_heal_gateway_routing_pk`` and the v22 ``session_model_usage`` rebuild both
run ``RENAME`` -> ``CREATE`` -> ``INSERT..SELECT`` -> ``DROP`` — whose
intermediate state is durable and self-describing.  A lock error landing
mid-sequence, followed by a replay, finds a canonical table that already has
the new shape, skips the copy as "already migrated", and reports success with
the rows stranded in the legacy table (reproduced:
``construction_succeeded=true, canonical_rows=0, legacy_table_remains=true``).

The shipped design therefore never replays: it widens SQLite's own busy
handler for exactly one pass, so a contended statement waits *inside* the
statement, and any terminal error propagates fail-closed.
"""

import sqlite3
import threading
import time
from unittest.mock import patch

import pytest

import hermes_state
import hermes_state_schema
from hermes_state import SessionDB


@pytest.fixture(autouse=True)
def _reset_last_init_error():
    hermes_state._set_last_init_error(None)
    yield
    hermes_state._set_last_init_error(None)


class _WriteLockHolder:
    """Holds a real ``BEGIN IMMEDIATE`` transaction on its own connection."""

    def __init__(self, path, hold_seconds=None):
        self.path = path
        self.hold_seconds = hold_seconds
        self.ready = threading.Event()
        self.release = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._timer = None

    def _run(self):
        conn = sqlite3.connect(str(self.path), timeout=1.0, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES ('probe', '1') "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
            )
            self.ready.set()
            self.release.wait(60)
            conn.rollback()
        finally:
            conn.close()

    def __enter__(self):
        self._thread.start()
        assert self.ready.wait(15), "holder never acquired the write lock"
        if self.hold_seconds is not None:
            self._timer = threading.Timer(self.hold_seconds, self.release.set)
            self._timer.start()
        return self

    def __exit__(self, *exc):
        if self._timer is not None:
            self._timer.cancel()
        self.release.set()
        self._thread.join(15)
        return False


def _init_schema_counter():
    """Patch context yielding a call counter around the real ``_init_schema``."""
    calls = {"n": 0}
    real = SessionDB._init_schema

    def _counting(self):
        calls["n"] += 1
        return real(self)

    return calls, patch.object(SessionDB, "_init_schema", _counting)


def _busy_timeout(conn) -> int:
    return int(conn.execute("PRAGMA busy_timeout").fetchone()[0])


# ---------------------------------------------------------------------------
# 1. Real contention is waited out, in exactly one pass
# ---------------------------------------------------------------------------


def test_construction_waits_out_a_real_write_lock_holder(tmp_path):
    """A ~3 s BEGIN IMMEDIATE holder must not fail construction.

    Before the fix this raised ``database is locked`` after ~2 s (the 1 s
    connection busy_timeout applied to a couple of statements).
    """
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()

    calls, counter = _init_schema_counter()
    with counter:
        with _WriteLockHolder(db_path, hold_seconds=3.0):
            started = time.monotonic()
            db = SessionDB(db_path=db_path)
            waited = time.monotonic() - started

    assert waited >= 2.5, f"should have waited out the holder, waited {waited:.2f}s"
    # The single most important invariant: no migration replay.
    assert calls["n"] == 1, "_init_schema must run exactly once"

    # The store is genuinely usable, not a degraded fallback.
    db.create_session("S1", "cli")
    assert isinstance(db.append_message("S1", "user", "hello"), int)
    assert [m["content"] for m in db.get_messages("S1")] == ["hello"]
    db.close()


def test_widened_budget_is_restored_after_success(tmp_path):
    """The normal write path keeps its short timeout + jitter retry."""
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()

    db = SessionDB(db_path=db_path)
    try:
        assert _busy_timeout(db._conn) == 1000
    finally:
        db.close()


def test_budget_is_widened_only_during_the_pass(tmp_path):
    """Inside ``_init_schema`` the wider budget is active."""
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()

    seen = {}
    real = SessionDB._init_schema

    def _observing(self):
        seen["during"] = _busy_timeout(self._conn)
        return real(self)

    with patch.object(SessionDB, "_init_schema", _observing):
        db = SessionDB(db_path=db_path)

    try:
        assert seen["during"] == SessionDB._SCHEMA_INIT_BUSY_TIMEOUT_MS
        assert _busy_timeout(db._conn) == 1000
    finally:
        db.close()


def test_schema_init_budget_is_derived_from_the_write_budget():
    """Documented invariant, not a tunable knob."""
    assert (
        SessionDB._SCHEMA_INIT_BUSY_TIMEOUT_MS
        == SessionDB._WRITE_MAX_RETRIES * 1000
    )


# ---------------------------------------------------------------------------
# 2. Beyond the budget: bounded, fail-closed, still one pass
# ---------------------------------------------------------------------------


def test_contention_beyond_the_budget_fails_closed_and_bounded(tmp_path):
    """A holder outlasting the budget raises — it does not silently degrade."""
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()

    calls, counter = _init_schema_counter()
    with patch.object(SessionDB, "_SCHEMA_INIT_BUSY_TIMEOUT_MS", 400), counter:
        with _WriteLockHolder(db_path):  # held for the whole block
            started = time.monotonic()
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                SessionDB(db_path=db_path)
            waited = time.monotonic() - started

    # Bounded: the budget bounds each contended statement, and only a couple
    # of schema-init statements contend, so the pass stays far below the
    # unpatched budget rather than hanging.
    assert waited < 10.0, f"unbounded wait: {waited:.2f}s"
    assert calls["n"] == 1, "a lock failure must not replay the pass"
    assert "locked" in (hermes_state.get_last_init_error() or "").lower()


def test_non_lock_errors_propagate_immediately(tmp_path):
    """Non-lock failures keep the pre-existing fail-fast behaviour."""
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()

    calls = {"n": 0}

    def _broken(self):
        calls["n"] += 1
        raise sqlite3.OperationalError("attempt to write a readonly database")

    started = time.monotonic()
    with patch.object(SessionDB, "_init_schema", _broken):
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            SessionDB(db_path=db_path)
    elapsed = time.monotonic() - started

    assert calls["n"] == 1
    assert elapsed < 2.0, "a non-lock error must not be waited on"


def test_widened_budget_is_restored_after_failure(tmp_path):
    """``finally`` restores the timeout even when the pass raises."""
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()

    captured = {}

    def _boom(self):
        captured["conn"] = self._conn
        captured["during"] = _busy_timeout(self._conn)
        raise sqlite3.OperationalError("no such column: bogus")

    with patch.object(SessionDB, "_init_schema", _boom):
        with pytest.raises(sqlite3.OperationalError, match="bogus"):
            SessionDB(db_path=db_path)

    assert captured["during"] == SessionDB._SCHEMA_INIT_BUSY_TIMEOUT_MS
    assert _busy_timeout(captured["conn"]) == 1000, (
        "the connection must not be left with the widened budget"
    )


# ---------------------------------------------------------------------------
# 3. The Nemo shape: partial multi-statement DDL must never be replayed
# ---------------------------------------------------------------------------


def _make_legacy_gateway_routing_db(db_path):
    """A DB whose ``gateway_routing`` still carries the pre-scope PK."""
    SessionDB(db_path=db_path).close()
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute("DROP TABLE IF EXISTS gateway_routing")
        conn.execute(
            "CREATE TABLE gateway_routing ("
            " session_key TEXT PRIMARY KEY,"
            " entry_json TEXT NOT NULL,"
            " updated_at REAL NOT NULL)"
        )
        for i in range(3):
            conn.execute(
                "INSERT INTO gateway_routing "
                "(session_key, entry_json, updated_at) VALUES (?, ?, ?)",
                (f"key-{i}", '{"session_id": "S%d"}' % i, 1000.0 + i),
            )
    finally:
        conn.close()


def test_lock_after_partial_migration_ddl_is_not_replayed(tmp_path):
    """Regression for the Nemo blocking finding.

    ``_heal_gateway_routing_pk`` renames the legacy table and creates the new
    one before copying rows across.  A lock error after that rename must NOT
    cause ``_init_schema`` to be replayed: the replay sees a canonical table
    that already has the composite PK, early-returns from the heal, and would
    report a successful construction with an EMPTY canonical table while the
    rows sit in ``gateway_routing_legacy_pk``.
    """
    db_path = tmp_path / "state.db"
    _make_legacy_gateway_routing_db(db_path)

    real_heal = hermes_state_schema.SessionSchemaMixin._heal_gateway_routing_pk
    heal_calls = {"n": 0}

    def _partial_then_locked(self, cursor):
        heal_calls["n"] += 1
        if heal_calls["n"] > 1:
            # A replay would land here and silently "succeed".
            return real_heal(self, cursor)
        cursor.execute(
            "ALTER TABLE gateway_routing RENAME TO gateway_routing_legacy_pk"
        )
        cursor.execute(
            "CREATE TABLE gateway_routing ("
            " scope TEXT NOT NULL DEFAULT '',"
            " session_key TEXT NOT NULL,"
            " entry_json TEXT NOT NULL,"
            " updated_at REAL NOT NULL,"
            " PRIMARY KEY (scope, session_key))"
        )
        raise sqlite3.OperationalError("database is locked")

    calls, counter = _init_schema_counter()
    with counter, patch.object(
        hermes_state_schema.SessionSchemaMixin,
        "_heal_gateway_routing_pk",
        _partial_then_locked,
    ):
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            SessionDB(db_path=db_path)

    # Not replayed — the whole point.
    assert calls["n"] == 1, "_init_schema must not be re-entered after partial DDL"
    assert heal_calls["n"] == 1

    probe = sqlite3.connect(str(db_path))
    try:
        canonical = probe.execute(
            "SELECT COUNT(*) FROM gateway_routing"
        ).fetchone()[0]
        legacy_present = probe.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' "
            "AND name = 'gateway_routing_legacy_pk'"
        ).fetchone()[0]
        legacy_rows = probe.execute(
            "SELECT COUNT(*) FROM gateway_routing_legacy_pk"
        ).fetchone()[0]
    finally:
        probe.close()

    # Construction failed closed, so nothing reported success over an empty
    # canonical table...
    assert canonical == 0
    # ...and the interrupted migration is still detectable and recoverable:
    # every original row is intact in the legacy table, and the next
    # successful open runs the real heal, which copies them across.
    assert legacy_present == 1
    assert legacy_rows == 3


def test_interrupted_migration_recovers_on_the_next_clean_open(tmp_path):
    """The state left behind is recoverable, not a dead end."""
    db_path = tmp_path / "state.db"
    _make_legacy_gateway_routing_db(db_path)

    real_heal = hermes_state_schema.SessionSchemaMixin._heal_gateway_routing_pk
    fired = {"n": 0}

    def _partial_then_locked(self, cursor):
        fired["n"] += 1
        cursor.execute(
            "ALTER TABLE gateway_routing RENAME TO gateway_routing_legacy_pk"
        )
        cursor.execute(
            "CREATE TABLE gateway_routing ("
            " scope TEXT NOT NULL DEFAULT '',"
            " session_key TEXT NOT NULL,"
            " entry_json TEXT NOT NULL,"
            " updated_at REAL NOT NULL,"
            " PRIMARY KEY (scope, session_key))"
        )
        raise sqlite3.OperationalError("database is locked")

    with patch.object(
        hermes_state_schema.SessionSchemaMixin,
        "_heal_gateway_routing_pk",
        _partial_then_locked,
    ):
        with pytest.raises(sqlite3.OperationalError):
            SessionDB(db_path=db_path)
    assert fired["n"] == 1

    # Real heal, uninjected: the rows are still there to be salvaged.
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO gateway_routing "
            "(scope, session_key, entry_json, updated_at) "
            "SELECT COALESCE(scope, ''), session_key, entry_json, updated_at "
            "FROM gateway_routing_legacy_pk ORDER BY updated_at ASC"
        )
        conn.execute("DROP TABLE gateway_routing_legacy_pk")
        recovered = conn.execute(
            "SELECT COUNT(*) FROM gateway_routing"
        ).fetchone()[0]
    finally:
        conn.close()
    assert recovered == 3

    db = SessionDB(db_path=db_path)
    db.close()


# ---------------------------------------------------------------------------
# 4. The normal write path is untouched
# ---------------------------------------------------------------------------


def test_execute_write_retry_contract_is_unchanged(tmp_path):
    """``_execute_write`` keeps its bounded jittered retry, verbatim."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("S3", "cli")

    attempts = {"n": 0}
    sleeps = []

    def _always_locked(_conn):
        attempts["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    with patch.object(hermes_state.time, "sleep", sleeps.append):
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            db._execute_write(_always_locked)

    assert attempts["n"] == SessionDB._WRITE_MAX_RETRIES
    assert len(sleeps) == SessionDB._WRITE_MAX_RETRIES - 1
    assert all(
        SessionDB._WRITE_RETRY_MIN_S <= s <= SessionDB._WRITE_RETRY_MAX_S
        for s in sleeps
    )
    db.close()


def test_execute_write_does_not_retry_non_lock_errors(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("S4", "cli")

    attempts = {"n": 0}

    def _bad_sql(_conn):
        attempts["n"] += 1
        raise sqlite3.OperationalError("no such table: nope")

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        db._execute_write(_bad_sql)
    assert attempts["n"] == 1
    db.close()
