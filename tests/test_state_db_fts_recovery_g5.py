"""G5 regression: atomic base-FTS rebuild + fuse-aware state.db recovery.

Two confirmed root causes in Hermes state recovery, both driven out under
strict RED-GREEN TDD against a real SQLite file (no mocks of the DB engine):

1. **Non-atomic base FTS rebuild (durable history loss).** ``SessionDB`` opens
   its connection with ``isolation_level=None`` (autocommit — see the comment
   at ``hermes_state.SessionDB.__init__``), and ``_rebuild_fts_indexes`` issued
   ``DELETE FROM messages_fts`` and the large ``INSERT ... SELECT`` repopulation
   as *separate* autocommitted statements. An interruption/failure after the
   DELETE committed but before the INSERT durably emptied the historical base
   FTS index — full-text search then silently returned nothing for all prior
   history. The reset+repopulate must be atomic: a failure after the delete
   boundary must roll back and leave the pre-existing rows searchable.

2. **Fuse-unaware recovery.** ``repair_state_db_schema`` rebuilt or preserved
   the optional ``messages_fts_trigram`` index during recovery even when the
   profile-local ``.disable-trigram-fts`` fuse was installed next to state.db.
   Recovery must honor the fuse: no trigram objects or triggers may survive it.

The base FTS is an *inline* FTS5 table (``USING fts5(content)``) whose content
is derived from ``messages`` (``content || tool_name || tool_calls``); the fix
must repopulate it with an explicit ``INSERT ... SELECT FROM messages``, never
the external-content ``'rebuild'`` command.
"""
import sqlite3
import uuid
from pathlib import Path

import pytest

import hermes_state
from hermes_state import (
    SessionDB,
    repair_state_db_schema,
    _TRIGRAM_FTS_DISABLE_MARKER,
)


def _build_healthy_db(db_path: Path, *, n_pairs: int = 5) -> str:
    """One session with ``n_pairs`` user/assistant message pairs.

    User turns carry the token ``pizza``, assistant turns ``pasta`` — distinct
    tokens let a MATCH prove *which* rows survived, not just a row count.
    """
    db = SessionDB(db_path=db_path)
    sid = db.create_session(session_id=str(uuid.uuid4()), source="cli")
    for i in range(n_pairs):
        db.append_message(sid, role="user", content=f"historical pizza message {i}")
        db.append_message(sid, role="assistant", content=f"assistant reply about pasta {i}")
    db.close()
    return sid


class _FailAfterDeleteCursor:
    """Real-cursor proxy that simulates a crash right after the DELETE boundary.

    Every statement runs against the wrapped real cursor EXCEPT the base
    ``INSERT INTO messages_fts(...) SELECT ... FROM messages`` repopulation,
    which raises — reproducing an interruption after ``DELETE FROM
    messages_fts`` has already executed but before the historical rows are
    restored. Transaction-control statements (SAVEPOINT / RELEASE / ROLLBACK)
    pass straight through so the fix's rollback can run against the file.
    """

    def __init__(self, real: sqlite3.Cursor) -> None:
        self._real = real
        self.saw_base_delete = False

    def execute(self, sql: str, *args):
        normalized = " ".join(sql.split()).upper()
        if normalized.startswith("DELETE FROM MESSAGES_FTS") and "TRIGRAM" not in normalized:
            self.saw_base_delete = True
        if normalized.startswith("INSERT INTO MESSAGES_FTS(") and "SELECT" in normalized:
            raise sqlite3.OperationalError(
                "simulated interruption after DELETE FROM messages_fts"
            )
        return self._real.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._real, name)


# ── Root cause 1: base FTS rebuild must be atomic ────────────────────────────


def test_rebuild_fts_rolls_back_and_preserves_history_on_failure(tmp_path):
    """A failure after the DELETE boundary must NOT durably empty base FTS."""
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path, n_pairs=5)

    db = SessionDB(db_path=db_path)
    try:
        assert db._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 10
        assert db._conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'pizza'"
        ).fetchone()[0] == 5  # sanity: pre-existing history is searchable

        proxy = _FailAfterDeleteCursor(db._conn.cursor())
        with pytest.raises(sqlite3.OperationalError):
            SessionDB._rebuild_fts_indexes(proxy, include_trigram=False)

        # The destructive DELETE boundary really was crossed before the failure
        # — otherwise this test would prove nothing about atomicity.
        assert proxy.saw_base_delete is True

        # After the failed rebuild rolls back, the pre-existing base FTS rows
        # are still present (row-count parity with messages) and searchable.
        assert db._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 10
        assert db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 10
        assert db._conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'pizza'"
        ).fetchone()[0] == 5
    finally:
        db.close()

    # A fresh connection sees the same preserved rows: nothing was durably
    # emptied by an autocommitted DELETE.
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 10
        assert conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'pasta'"
        ).fetchone()[0] == 5
    finally:
        conn.close()


def test_rebuild_fts_success_reindexes_every_row(tmp_path):
    """The happy path still fully repopulates the inline base FTS index.

    Guards the SAVEPOINT commit path: the fix must not turn the rebuild into a
    no-op. Passes on the pre-fix code too (it is a non-regression guard).
    """
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path, n_pairs=5)

    db = SessionDB(db_path=db_path)
    try:
        db._conn.execute("DELETE FROM messages_fts")  # wipe out-of-band
        assert db._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 0

        SessionDB._rebuild_fts_indexes(db._conn.cursor(), include_trigram=False)

        assert db._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 10
        assert db._conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'pizza'"
        ).fetchone()[0] == 5
        assert db._conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'pasta'"
        ).fetchone()[0] == 5
    finally:
        db.close()


# ── Root cause 2: recovery must honor the trigram fuse ───────────────────────


def _corrupt_duplicate_fts(db_path: Path) -> None:
    """Inject a duplicate ``messages_fts`` row into sqlite_master.

    Reproduces 'malformed database schema (messages_fts) - table messages_fts
    already exists', which routes recovery through the dedup pass (that pass
    preserves the trigram index, so it exposes fuse-unawareness).
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA writable_schema=ON")
    conn.execute(
        "INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) "
        "SELECT type, name, tbl_name, rootpage, sql FROM sqlite_master "
        "WHERE name='messages_fts'"
    )
    conn.commit()
    conn.close()


def _count_trigram_objects(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'messages_fts_trigram%'"
        ).fetchone()[0]
    finally:
        conn.close()


def _count_trigram_triggers(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' "
            "AND name LIKE 'messages_fts_trigram%'"
        ).fetchone()[0]
    finally:
        conn.close()


def test_fused_profile_recovery_removes_all_trigram_objects(tmp_path):
    """With the fuse installed, recovery leaves zero trigram objects/triggers."""
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    assert _count_trigram_objects(db_path) > 0  # built while non-fused

    # Operator installs the profile-local fuse, then a recovery path fires.
    (tmp_path / _TRIGRAM_FTS_DISABLE_MARKER).write_text("", encoding="utf-8")
    _corrupt_duplicate_fts(db_path)

    report = repair_state_db_schema(db_path, backup=False)
    assert report["repaired"] is True

    # Fuse honored: no trigram virtual table, shadow tables, or triggers remain.
    assert _count_trigram_objects(db_path) == 0
    assert _count_trigram_triggers(db_path) == 0

    # The healthy base FTS index is preserved, still searchable; canonical rows
    # are untouched.
    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'pizza'"
        ).fetchone()[0] == 5
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 10
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
    finally:
        conn.close()

    # The natural post-recovery reopen (what auto-heal does) must not recreate
    # trigram on the fused profile, and base search must keep working.
    db = SessionDB(db_path=db_path)
    try:
        assert db._trigram_available is False
        assert db._conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'pasta'"
        ).fetchone()[0] == 5
    finally:
        db.close()
    assert _count_trigram_objects(db_path) == 0
    assert _count_trigram_triggers(db_path) == 0


def test_non_fused_recovery_preserves_trigram(tmp_path):
    """Without the fuse, recovery preserves the trigram index (unchanged).

    Non-regression guard for the fuse threading: it must key off the marker and
    never touch trigram on a normal (non-fused) profile.
    """
    db_path = tmp_path / "state.db"
    _build_healthy_db(db_path)
    assert _count_trigram_objects(db_path) > 0
    _corrupt_duplicate_fts(db_path)  # NO fuse installed

    report = repair_state_db_schema(db_path, backup=False)
    assert report["repaired"] is True

    assert _count_trigram_objects(db_path) > 0
    assert _count_trigram_triggers(db_path) == 3
