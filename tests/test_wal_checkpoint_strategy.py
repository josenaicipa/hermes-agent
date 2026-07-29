"""Tests for SessionDB WAL checkpoint strategy (issue #45383).

Verifies that periodic checkpoints use PASSIVE mode (safe for large DBs).
close() also uses PASSIVE — SessionDB is instantiated per request in
gateway and web-server paths, so close() is not necessarily a process
shutdown boundary. The exclusive TRUNCATE checkpoint remains reserved for
controlled maintenance (pre-VACUUM) while writers are quiesced.
"""

import sqlite3
import logging
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    try:
        session_db.close()
    except Exception:
        pass


class TestTryWalCheckpointPassive:
    """_try_wal_checkpoint() should use PASSIVE mode for periodic use."""

    def test_checkpoint_uses_passive_mode(self, db):
        """PASSIVE checkpoint does not require exclusive lock — safe for large DBs."""
        # Capture the real connection's execute before mocking
        real_conn = db._conn
        execute_calls = []

        def tracking_execute(sql, *args, **kwargs):
            execute_calls.append(sql)
            return real_conn.execute(sql, *args, **kwargs)

        # sqlite3.Connection.execute is read-only (C extension) — replace _conn
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = tracking_execute
        mock_conn.fetchone.return_value = None
        db._conn = mock_conn

        db._try_wal_checkpoint()

        passive_calls = [c for c in execute_calls if "wal_checkpoint(PASSIVE)" in c]
        truncate_calls = [c for c in execute_calls if "wal_checkpoint(TRUNCATE)" in c]
        assert len(passive_calls) == 1, (
            f"Expected 1 PASSIVE checkpoint call, got {len(passive_calls)}"
        )
        assert len(truncate_calls) == 0, (
            "Periodic checkpoint should NOT use TRUNCATE"
        )

    def test_checkpoint_logs_warning_on_failure(self, db, caplog):
        """Failed PASSIVE checkpoint logs a warning instead of silent pass."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("disk I/O error")
        db._conn = mock_conn

        with caplog.at_level(logging.WARNING):
            db._try_wal_checkpoint()

        assert any("WAL checkpoint (PASSIVE) failed" in r.message for r in caplog.records), (
            f"Expected warning log about PASSIVE checkpoint failure, got: {caplog.text}"
        )

    def test_checkpoint_returns_result_on_success(self, db):
        """Successful PASSIVE checkpoint does not raise."""
        db._try_wal_checkpoint()


class TestClosePassiveCheckpoint:
    """close() must use PASSIVE, not exclusive TRUNCATE.

    SessionDB is instantiated per request in gateway and web-server paths,
    so close() is not necessarily a process shutdown boundary (see
    hermes_state.SessionDB.close docstring).
    """

    def test_close_uses_passive_mode(self, db):
        """PASSIVE at close does not require an exclusive lock or block writers."""
        real_conn = db._conn
        execute_calls = []

        def tracking_execute(sql, *args, **kwargs):
            execute_calls.append(sql)
            return real_conn.execute(sql, *args, **kwargs)

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = tracking_execute
        db._conn = mock_conn

        db.close()

        passive_calls = [c for c in execute_calls if "wal_checkpoint(PASSIVE)" in c]
        truncate_calls = [c for c in execute_calls if "wal_checkpoint(TRUNCATE)" in c]
        assert len(passive_calls) == 1, (
            f"Expected 1 PASSIVE checkpoint at close, got {len(passive_calls)}"
        )
        assert len(truncate_calls) == 0, (
            "close() should NOT use an exclusive TRUNCATE checkpoint"
        )

    def test_close_logs_debug_on_failure(self, db, caplog):
        """Failed PASSIVE checkpoint at close logs debug (close is best-effort)."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("database is locked")
        db._conn = mock_conn

        with caplog.at_level(logging.DEBUG):
            db.close()

        assert any("WAL checkpoint (PASSIVE) at close failed" in r.message for r in caplog.records), (
            f"Expected debug log about PASSIVE failure at close, got: {caplog.text}"
        )


class TestCheckpointFrequency:
    """Checkpoint triggers every N writes."""

    def test_checkpoint_triggers_at_interval(self, db):
        """_try_wal_checkpoint is called every _CHECKPOINT_EVERY_N_WRITES writes."""
        call_count = [0]
        original = db._try_wal_checkpoint

        def counting_checkpoint():
            call_count[0] += 1
            original()

        db._try_wal_checkpoint = counting_checkpoint

        # Write exactly _CHECKPOINT_EVERY_N_WRITES sessions to trigger one checkpoint
        n = db._CHECKPOINT_EVERY_N_WRITES
        import time as _time
        for i in range(n):
            db._execute_write(lambda conn, _i=i: conn.execute(
                "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                (f"sess_{_i}", "test", _time.time()),
            ))

        assert call_count[0] == 1, (
            f"Expected 1 checkpoint after {n} writes, got {call_count[0]}"
        )
