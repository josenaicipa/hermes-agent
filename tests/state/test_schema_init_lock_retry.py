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

Two follow-on properties are covered here as well:

  * **Recovery.** Failing closed is not enough on its own — the leftover
    ``gateway_routing_legacy_pk`` has to be merged back automatically.  The
    heal used to return as soon as the canonical table had the composite PK,
    which declared an interrupted rebuild "healthy" forever
    (``second_open_succeeded=True, canonical_rows=0, legacy_rows=3``).  It
    now detects the leftover table first and merges it, newest
    ``updated_at`` per ``(scope, session_key)`` winning and ties keeping the
    canonical row, dropping the leftover only after the merge commits.
  * **Atomicity.** The rebuild/recovery runs inside one SAVEPOINT, so a new
    interruption rolls back to the original table instead of creating the
    stranded shape at all.
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


def _make_legacy_gateway_routing_db(db_path, *, wal: bool = False):
    """A DB whose ``gateway_routing`` still carries the pre-scope PK.

    ``wal=True`` pre-creates the file in WAL so ``apply_wal_with_fallback``
    keeps WAL (it refuses to live-downgrade an on-disk WAL database), which
    lets the recovery flow be exercised in the journal mode production runs.
    """
    if wal:
        boot = sqlite3.connect(str(db_path), isolation_level=None)
        boot.execute("PRAGMA journal_mode=WAL")
        boot.close()
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


# ---------------------------------------------------------------------------
# 4. Recovery: a plain second SessionDB() must heal the stranded rows
# ---------------------------------------------------------------------------
#
# Nemo evidence at b9b7: {'second_open_succeeded': True, 'canonical_rows': 0,
# 'legacy_rows': 3}.  `_heal_gateway_routing_pk` returned as soon as the
# canonical table had the composite PK, without ever looking for the leftover
# `gateway_routing_legacy_pk` — so the interrupted rebuild was declared
# healthy forever and the routing rows stayed stranded.
#
# These tests use NO manual recovery SQL.  They damage the database the way
# the pre-fix code did, then open SessionDB() and assert the rows are back.


def _damage_with_partial_rebuild(db_path):
    """Leave the exact stranded-rows shape an interrupted rebuild produced.

    Performed WITHOUT the savepoint the shipped code now uses, because that
    is precisely the state of a database damaged by the older build — the
    shipped path can no longer create it (see
    ``test_shipped_rebuild_is_atomic_on_failure``).
    """
    def _partial_then_locked(self, cursor):
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

    assert _routing_state(db_path) == (0, True, 3), (
        "fixture did not produce the stranded-rows shape"
    )


def _routing_state(db_path):
    """``(canonical_rows, legacy_present, legacy_rows)``."""
    conn = sqlite3.connect(str(db_path))
    try:
        canonical = conn.execute(
            "SELECT COUNT(*) FROM gateway_routing"
        ).fetchone()[0]
        legacy_present = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' "
            "AND name = 'gateway_routing_legacy_pk'"
        ).fetchone()[0] > 0
        legacy_rows = (
            conn.execute(
                "SELECT COUNT(*) FROM gateway_routing_legacy_pk"
            ).fetchone()[0]
            if legacy_present else 0
        )
        return canonical, legacy_present, legacy_rows
    finally:
        conn.close()


def _routing_entries(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        return dict(
            conn.execute("SELECT session_key, entry_json FROM gateway_routing")
        )
    finally:
        conn.close()


@pytest.mark.parametrize("wal", [False, True], ids=["delete", "wal"])
def test_second_open_recovers_stranded_rows_without_manual_sql(tmp_path, wal):
    """THE requirement: plain second ``SessionDB()``, zero operator SQL."""
    db_path = tmp_path / "state.db"
    _make_legacy_gateway_routing_db(db_path, wal=wal)
    _damage_with_partial_rebuild(db_path)

    # No recovery SQL of any kind — just open the database.
    SessionDB(db_path=db_path).close()

    canonical, legacy_present, _ = _routing_state(db_path)
    assert canonical == 3, "stranded rows were not merged back"
    assert legacy_present is False, "legacy table dropped only after the merge"
    assert _routing_entries(db_path) == {
        f"key-{i}": '{"session_id": "S%d"}' % i for i in range(3)
    }


@pytest.mark.parametrize("wal", [False, True], ids=["delete", "wal"])
def test_recovery_is_idempotent_across_repeat_opens(tmp_path, wal):
    db_path = tmp_path / "state.db"
    _make_legacy_gateway_routing_db(db_path, wal=wal)
    _damage_with_partial_rebuild(db_path)

    for _ in range(3):
        SessionDB(db_path=db_path).close()
        assert _routing_state(db_path) == (3, False, 0)


@pytest.mark.parametrize("wal", [False, True], ids=["delete", "wal"])
def test_recovery_never_clobbers_a_newer_canonical_row(tmp_path, wal):
    """Newest ``updated_at`` wins; an exact tie keeps the canonical row."""
    db_path = tmp_path / "state.db"
    _make_legacy_gateway_routing_db(db_path, wal=wal)
    _damage_with_partial_rebuild(db_path)

    # A live gateway kept writing routing entries into the new table while the
    # rebuild was stuck: one strictly newer, one on an exact timestamp tie.
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute(
            "INSERT INTO gateway_routing "
            "(scope, session_key, entry_json, updated_at) "
            "VALUES ('', 'key-1', '{\"session_id\": \"NEWER\"}', 9999.0)"
        )
        conn.execute(
            "INSERT INTO gateway_routing "
            "(scope, session_key, entry_json, updated_at) "
            "VALUES ('', 'key-2', '{\"session_id\": \"TIE\"}', 1002.0)"
        )
    finally:
        conn.close()

    SessionDB(db_path=db_path).close()

    entries = _routing_entries(db_path)
    assert _routing_state(db_path) == (3, False, 0)
    assert entries["key-1"] == '{"session_id": "NEWER"}', "newer row clobbered"
    assert entries["key-2"] == '{"session_id": "TIE"}', "tie must keep canonical"
    # ...and the legacy-only row is still merged in.
    assert entries["key-0"] == '{"session_id": "S0"}'


@pytest.mark.parametrize("wal", [False, True], ids=["delete", "wal"])
def test_recovers_when_interruption_landed_before_the_create(tmp_path, wal):
    """RENAME committed but CREATE did not: canonical missing entirely."""
    db_path = tmp_path / "state.db"
    _make_legacy_gateway_routing_db(db_path, wal=wal)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute(
            "ALTER TABLE gateway_routing RENAME TO gateway_routing_legacy_pk"
        )
    finally:
        conn.close()

    SessionDB(db_path=db_path).close()
    assert _routing_state(db_path) == (3, False, 0)


@pytest.mark.parametrize("wal", [False, True], ids=["delete", "wal"])
def test_shipped_rebuild_is_atomic_on_failure(tmp_path, wal):
    """A failure inside the shipped rebuild rolls back to the original table.

    This is what stops a NEW interruption from ever producing the stranded
    shape: SQLite DDL is transactional, and the whole
    RENAME/CREATE/COPY/DROP sequence runs in one savepoint.
    """
    db_path = tmp_path / "state.db"
    _make_legacy_gateway_routing_db(db_path, wal=wal)

    def _boom(self, cursor, legacy_columns):
        raise sqlite3.OperationalError("database is locked")

    with patch.object(
        hermes_state_schema.SessionSchemaMixin,
        "_merge_gateway_routing_legacy",
        _boom,
    ):
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            SessionDB(db_path=db_path)

    # Fully rolled back: no leftover table, original rows and original PK.
    canonical, legacy_present, _ = _routing_state(db_path)
    assert legacy_present is False, "the RENAME was not rolled back"
    assert canonical == 3
    conn = sqlite3.connect(str(db_path))
    try:
        pk = [
            r[1] for r in sorted(
                (r for r in conn.execute('PRAGMA table_info("gateway_routing")')
                 if r[5]),
                key=lambda r: r[5],
            )
        ]
    finally:
        conn.close()
    assert pk == ["session_key"], "original legacy table was not preserved"

    # And a clean open still performs the real migration afterwards.
    SessionDB(db_path=db_path).close()
    assert _routing_state(db_path) == (3, False, 0)


def test_malformed_leftover_table_fails_closed(tmp_path):
    """A leftover we cannot merge must not be dropped or ignored."""
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute(
            "CREATE TABLE gateway_routing_legacy_pk "
            "(session_key TEXT, junk TEXT)"
        )
        conn.execute(
            "INSERT INTO gateway_routing_legacy_pk VALUES ('k', 'v')"
        )
    finally:
        conn.close()

    with pytest.raises(sqlite3.DatabaseError, match="missing column"):
        SessionDB(db_path=db_path)

    # Nothing dropped, nothing emptied.
    _canonical, legacy_present, legacy_rows = _routing_state(db_path)
    assert (legacy_present, legacy_rows) == (True, 1)


def test_ambiguous_double_legacy_shape_fails_closed(tmp_path):
    """Leftover next to a still-legacy canonical table is not guessable."""
    db_path = tmp_path / "state.db"
    _make_legacy_gateway_routing_db(db_path)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute(
            "CREATE TABLE gateway_routing_legacy_pk ("
            " session_key TEXT PRIMARY KEY,"
            " entry_json TEXT NOT NULL,"
            " updated_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO gateway_routing_legacy_pk VALUES ('x', '{}', 1.0)"
        )
    finally:
        conn.close()

    with pytest.raises(sqlite3.DatabaseError, match="ambiguous"):
        SessionDB(db_path=db_path)

    assert _routing_state(db_path) == (3, True, 1)


# ---------------------------------------------------------------------------
# 5. busy_timeout restoration semantics
# ---------------------------------------------------------------------------


def _break_busy_timeout_restore(db):
    """Make only ``PRAGMA busy_timeout = N`` fail on this connection."""
    original = db._conn.execute

    def _execute(sql, *args, **kwargs):
        if isinstance(sql, str) and sql.startswith("PRAGMA busy_timeout ="):
            raise sqlite3.OperationalError("pragma refused")
        return original(sql, *args, **kwargs)

    db._conn.execute = _execute


def test_restore_failure_after_successful_init_fails_construction(tmp_path):
    """Never hand back a connection still carrying the widened handler."""
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()

    real_init = SessionDB._init_schema
    ran = {"init": False}

    def _init_then_break_restore(self):
        real_init(self)
        ran["init"] = True
        _break_busy_timeout_restore(self)

    with patch.object(SessionDB, "_init_schema", _init_then_break_restore):
        with pytest.raises(
            sqlite3.OperationalError, match="busy_timeout could not be restored"
        ):
            SessionDB(db_path=db_path)

    assert ran["init"] is True, "the failure must come from the restore, not init"


def test_restore_failure_after_failed_init_preserves_original_error(tmp_path):
    """The actionable error is the schema failure, not the restore failure."""
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()

    def _init_fails_and_breaks_restore(self):
        _break_busy_timeout_restore(self)
        raise sqlite3.OperationalError("ORIGINAL schema failure")

    with patch.object(SessionDB, "_init_schema", _init_fails_and_breaks_restore):
        with pytest.raises(
            sqlite3.OperationalError, match="ORIGINAL schema failure"
        ):
            SessionDB(db_path=db_path)


# ---------------------------------------------------------------------------
# 6. The normal write path is untouched
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
