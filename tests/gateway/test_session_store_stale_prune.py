"""Tests for SessionStore._prune_stale_sessions_locked — crash self-healing.

When a gateway crashes (exit code 1) the graceful shutdown path is skipped and
sessions.json is left pointing at sessions already ended in state.db. On the
next startup _ensure_loaded_locked calls _prune_stale_sessions_locked to detect
and remove those stale routing entries before get_or_create_session() can reuse
them and silently route incoming messages into a closed session (#52804).
"""

import json
import threading
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionSource, SessionStore
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(key: str, session_id: str) -> SessionEntry:
    now = datetime.now()
    return SessionEntry(
        session_key=key,
        session_id=session_id,
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=1),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )


def _make_entry_with_origin(key: str, session_id: str) -> SessionEntry:
    entry = _make_entry(key, session_id)
    entry.origin = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="5140768830",
        chat_type="dm",
        user_id="5140768830",
        user_name="João",
    )
    return entry


def _make_store_with_db(tmp_path, db_mock) -> SessionStore:
    """Build a SessionStore with a mock SessionDB, bypassing disk load."""
    config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = db_mock
    store._loaded = True
    return store


def _db_returning(rows: dict) -> MagicMock:
    """SessionDB mock where get_session maps session_id -> row dict."""
    db = MagicMock()
    db.get_session.side_effect = lambda sid: rows.get(sid)
    return db


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------

class TestPruneStaleSessionsLocked:
    def test_prunes_ended_session(self, tmp_path):
        db = _db_returning({"sid_dm": {"end_reason": "agent_close", "id": "sid_dm"}})
        store = _make_store_with_db(tmp_path, db)
        store._entries["dm_key"] = _make_entry("dm_key", "sid_dm")

        store._prune_stale_sessions_locked()

        assert "dm_key" not in store._entries

    def test_keeps_live_session(self, tmp_path):
        db = _db_returning({"sid_live": {"end_reason": None, "id": "sid_live"}})
        store = _make_store_with_db(tmp_path, db)
        store._entries["live_key"] = _make_entry("live_key", "sid_live")

        store._prune_stale_sessions_locked()

        assert "live_key" in store._entries

    def test_keeps_session_absent_from_db(self, tmp_path):
        """Entry for a session_id not in state.db (legacy) is left alone."""
        db = _db_returning({})
        store = _make_store_with_db(tmp_path, db)
        store._entries["legacy_key"] = _make_entry("legacy_key", "sid_legacy")

        store._prune_stale_sessions_locked()

        assert "legacy_key" in store._entries

    def test_prunes_multiple_stale_entries(self, tmp_path):
        db = _db_returning({
            "sid_a": {"end_reason": "agent_close", "id": "sid_a"},
            "sid_b": {"end_reason": "session_reset", "id": "sid_b"},
            "sid_c": {"end_reason": None, "id": "sid_c"},  # alive — keep
        })
        store = _make_store_with_db(tmp_path, db)
        store._entries["key_a"] = _make_entry("key_a", "sid_a")
        store._entries["key_b"] = _make_entry("key_b", "sid_b")
        store._entries["key_c"] = _make_entry("key_c", "sid_c")

        store._prune_stale_sessions_locked()

        assert "key_a" not in store._entries
        assert "key_b" not in store._entries
        assert "key_c" in store._entries

    def test_repoints_stale_compression_parent_to_latest_live_child(self, tmp_path):
        """Compression-ended parents should recover their live child mapping.

        A gateway crash can leave sessions.json pointing at the pre-compression
        parent (end_reason='compression') even though the agent already rotated
        into a live child session. If the child has gateway peer metadata, the
        startup prune pass must repoint the route instead of deleting it, or
        restart auto-resume and queued follow-ups have no session to continue.
        """
        key = "agent:main:telegram:dm:5140768830"
        db = _db_returning({
            "sid_parent": {"end_reason": "compression", "id": "sid_parent"},
        })
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_child",
            "started_at": 1782744974.0,
        }
        store = _make_store_with_db(tmp_path, db)
        store._entries[key] = _make_entry_with_origin(key, "sid_parent")

        store._prune_stale_sessions_locked()

        assert key in store._entries
        assert store._entries[key].session_id == "sid_child"
        db.find_latest_gateway_session_for_peer.assert_called_once()
        db.reopen_session.assert_called_once_with("sid_child")

    def test_repoints_compression_parent_via_lineage_when_child_lacks_peer_metadata(self, tmp_path):
        """A crash before gateway metadata propagation must not delete the route.

        Compression creates the child row before the gateway persists that
        child's peer metadata. If the process dies in that window, startup can
        still recover the live child through the parent-to-child lineage.
        """
        key = "agent:main:discord:group:123:456"
        db = _db_returning({
            "sid_parent": {"end_reason": "compression", "id": "sid_parent"},
            "sid_child": {"end_reason": None, "id": "sid_child"},
        })
        db.get_compression_tip.return_value = "sid_child"
        db.find_latest_gateway_session_for_peer.return_value = None
        store = _make_store_with_db(tmp_path, db)
        store._entries[key] = _make_entry_with_origin(key, "sid_parent")

        store._prune_stale_sessions_locked()

        assert key in store._entries
        assert store._entries[key].session_id == "sid_child"
        db.get_compression_tip.assert_called_once_with("sid_parent")
        db.find_latest_gateway_session_for_peer.assert_not_called()

    def test_repoints_compression_parent_via_real_sqlite_lineage(self, tmp_path):
        key = "agent:main:discord:group:123:456"
        db = SessionDB(tmp_path / "state.db")
        db.create_session("sid_parent", source="discord")
        db.end_session("sid_parent", "compression")
        db.create_session(
            "sid_child",
            source="discord",
            parent_session_id="sid_parent",
        )
        store = _make_store_with_db(tmp_path, db)
        store._entries[key] = _make_entry_with_origin(key, "sid_parent")

        store._prune_stale_sessions_locked()

        assert key in store._entries
        assert store._entries[key].session_id == "sid_child"

    def test_prunes_compression_parent_when_lineage_tip_is_ended(self, tmp_path):
        key = "agent:main:discord:group:123:456"
        db = SessionDB(tmp_path / "state.db")
        db.create_session("sid_parent", source="discord")
        db.end_session("sid_parent", "compression")
        db.create_session(
            "sid_child",
            source="discord",
            parent_session_id="sid_parent",
        )
        db.end_session("sid_child", "user_exit")
        store = _make_store_with_db(tmp_path, db)
        store._entries[key] = _make_entry_with_origin(key, "sid_parent")

        store._prune_stale_sessions_locked()

        assert key not in store._entries

    def test_prunes_compression_parent_when_no_lineage_child_exists(self, tmp_path):
        key = "agent:main:discord:group:123:456"
        db = SessionDB(tmp_path / "state.db")
        db.create_session("sid_parent", source="discord")
        db.end_session("sid_parent", "compression")
        store = _make_store_with_db(tmp_path, db)
        store._entries[key] = _make_entry_with_origin(key, "sid_parent")

        store._prune_stale_sessions_locked()

        assert key not in store._entries

    def test_prunes_stale_entry_when_recovery_only_finds_same_ended_session(self, tmp_path):
        key = "agent:main:telegram:dm:5140768830"
        db = _db_returning({"sid_parent": {"end_reason": "agent_close", "id": "sid_parent"}})
        db.find_latest_gateway_session_for_peer.return_value = {
            "id": "sid_parent",
            "started_at": 1782744974.0,
        }
        store = _make_store_with_db(tmp_path, db)
        store._entries[key] = _make_entry_with_origin(key, "sid_parent")

        store._prune_stale_sessions_locked()

        assert key not in store._entries

    def test_keeps_stale_entry_when_recovery_lookup_raises(self, tmp_path):
        """Indeterminate recovery must not delete the only routing handle.

        Startup pruning sees an ended parent and tries to repoint it to the
        latest live gateway child.  If that recovery query raises, deleting the
        sessions.json entry loses the routing key entirely; keeping it lets the
        runtime stale guard retry recovery on the next message.
        """
        key = "agent:main:telegram:dm:5140768830"
        db = _db_returning({"sid_parent": {"end_reason": "compression", "id": "sid_parent"}})
        db.find_latest_gateway_session_for_peer.side_effect = RuntimeError("db busy")
        store = _make_store_with_db(tmp_path, db)
        store._entries[key] = _make_entry_with_origin(key, "sid_parent")

        store._prune_stale_sessions_locked()

        assert key in store._entries
        assert store._entries[key].session_id == "sid_parent"

    def test_noop_when_db_is_none(self, tmp_path):
        config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = None
        store._loaded = True
        store._entries["key"] = _make_entry("key", "sid_x")

        store._prune_stale_sessions_locked()  # must not raise

        assert "key" in store._entries

    def test_noop_when_no_entries(self, tmp_path):
        db = MagicMock()
        store = _make_store_with_db(tmp_path, db)

        store._prune_stale_sessions_locked()

        db.get_session.assert_not_called()

    def test_db_error_is_non_fatal(self, tmp_path):
        db = MagicMock()
        db.get_session.side_effect = Exception("DB locked")
        store = _make_store_with_db(tmp_path, db)
        store._entries["key"] = _make_entry("key", "sid_x")

        store._prune_stale_sessions_locked()  # must not raise

        assert "key" in store._entries  # safe fallback — keep on error

    def test_sessions_json_rewritten_after_pruning(self, tmp_path):
        db = _db_returning({"sid_stale": {"end_reason": "agent_close", "id": "sid_stale"}})
        store = _make_store_with_db(tmp_path, db)
        store._entries["stale_key"] = _make_entry("stale_key", "sid_stale")

        with patch.object(store, "_save") as mock_save:
            store._prune_stale_sessions_locked()
            mock_save.assert_called_once()

    def test_sessions_json_not_rewritten_when_nothing_pruned(self, tmp_path):
        db = _db_returning({"sid_live": {"end_reason": None, "id": "sid_live"}})
        store = _make_store_with_db(tmp_path, db)
        store._entries["live_key"] = _make_entry("live_key", "sid_live")

        with patch.object(store, "_save") as mock_save:
            store._prune_stale_sessions_locked()
            mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: _ensure_loaded_locked calls _prune_stale_sessions_locked
# ---------------------------------------------------------------------------

class TestEnsureLoadedCallsPrune:
    def test_stale_entry_pruned_during_load(self, tmp_path):
        entry = _make_entry("dm_key", "sid_stale")
        (tmp_path / "sessions.json").write_text(
            json.dumps({"dm_key": entry.to_dict()}, indent=2), encoding="utf-8"
        )
        db = _db_returning({"sid_stale": {"end_reason": "agent_close", "id": "sid_stale"}})
        config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
        store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db

        store._ensure_loaded()

        assert "dm_key" not in store._entries

    def test_live_entry_survives_load(self, tmp_path):
        entry = _make_entry("active_key", "sid_live")
        (tmp_path / "sessions.json").write_text(
            json.dumps({"active_key": entry.to_dict()}, indent=2), encoding="utf-8"
        )
        db = _db_returning({"sid_live": {"end_reason": None, "id": "sid_live"}})
        config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
        store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db

        store._ensure_loaded()

        assert "active_key" in store._entries


# ---------------------------------------------------------------------------
# Concurrency barrier (G3): startup heal must run without SessionStore._lock
# held, and the real inbound path must wait for it to reach a terminal state
# before reading the routing index. Unlike the tests above (single-threaded,
# calling _prune_stale_sessions_locked directly), these use real threads, a
# real SessionDB, and threading.Event to pause/release the unlocked
# DB-resolution boundary — proving Thread B is blocked by the new heal
# barrier rather than by the pre-existing global lock.
# ---------------------------------------------------------------------------

class TestStartupHealBarrier:
    KEY = "agent:main:telegram:dm:919191"
    SOURCE = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="919191",
        chat_type="dm",
        user_id="919191",
    )

    @staticmethod
    def _make_unloaded_store(tmp_path, db) -> SessionStore:
        """A SessionStore that has NOT loaded yet — the next _ensure_loaded()
        / get_or_create_session() call performs the real, un-bypassed startup
        load and becomes the startup-heal owner."""
        config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
        store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db
        return store

    def _seed_compression_route(self, tmp_path, db):
        """Real SQLite compression parent -> live child, routed via a stale
        sessions.json entry pointing at the (ended) parent."""
        db.create_session("sid_parent", source="telegram")
        db.end_session("sid_parent", "compression")
        db.create_session("sid_child", source="telegram", parent_session_id="sid_parent")

        entry = _make_entry(self.KEY, "sid_parent")
        entry.origin = self.SOURCE
        (tmp_path / "sessions.json").write_text(
            json.dumps({self.KEY: entry.to_dict()}, indent=2), encoding="utf-8"
        )

    def _seed_ended_route_without_recovery(self, tmp_path, db):
        """Real SQLite: an agent_close-ended session with no lineage child and
        no recoverable gateway peer.

        Unlike the compression fixture above, nothing about this route is
        compression-specific, so get_or_create_session's separate per-request
        compression-tip healer (_heal_compression_tip_locked) is a structural
        no-op here (get_compression_tip finds no child and returns the same
        id). That isolates the startup-heal failure path under test from
        that other, pre-existing healing mechanism.
        """
        db.create_session("sid_ended", source="telegram")
        db.end_session("sid_ended", "agent_close")

        entry = _make_entry(self.KEY, "sid_ended")
        entry.origin = self.SOURCE
        (tmp_path / "sessions.json").write_text(
            json.dumps({self.KEY: entry.to_dict()}, indent=2), encoding="utf-8"
        )

    def test_startup_heal_db_phase_runs_without_routing_lock(self, tmp_path):
        """The first startup-heal DB lookup must run with SessionStore._lock
        free — proving the heal's DB-resolution phase is not nested inside
        the routing lock (d7c62193b holds it there via
        _ensure_loaded_locked -> _prune_stale_sessions_locked)."""
        db = SessionDB(tmp_path / "state.db")
        self._seed_compression_route(tmp_path, db)
        store = self._make_unloaded_store(tmp_path, db)

        real_get_session = db.get_session
        lock_was_free: list = []

        def instrumented_get_session(session_id):
            acquired = store._lock.acquire(blocking=False)
            lock_was_free.append(acquired)
            if acquired:
                store._lock.release()
            return real_get_session(session_id)

        db.get_session = instrumented_get_session

        store._ensure_loaded()

        assert lock_was_free, "startup-heal DB lookup was never invoked"
        assert all(lock_was_free), (
            "SessionStore._lock must be free during every startup-heal DB "
            "lookup, not just serialized behind it"
        )

    def test_inbound_waits_for_terminal_startup_heal(self, tmp_path):
        """Thread B (the real get_or_create_session inbound path) must block
        until Thread A's startup heal reaches a terminal state, then resolve
        the compression continuation child rather than the stale parent."""
        db = SessionDB(tmp_path / "state.db")
        self._seed_compression_route(tmp_path, db)
        store = self._make_unloaded_store(tmp_path, db)

        reached_snapshot = threading.Event()
        release_snapshot = threading.Event()
        original_resolve = store._resolve_stale_sessions

        def paused_resolve(items):
            actions = original_resolve(items)
            reached_snapshot.set()
            assert release_snapshot.wait(timeout=5), "test never released thread A"
            return actions

        store._resolve_stale_sessions = paused_resolve

        b_entered_wait = threading.Event()
        original_wait = store._wait_for_startup_heal_locked

        def instrumented_wait():
            b_entered_wait.set()
            return original_wait()

        store._wait_for_startup_heal_locked = instrumented_wait

        thread_a = threading.Thread(target=store._ensure_loaded)
        result: dict = {}

        def run_b():
            result["entry"] = store.get_or_create_session(self.SOURCE)

        thread_b = threading.Thread(target=run_b)

        thread_a.start()
        assert reached_snapshot.wait(timeout=5), (
            "thread A never reached the unlocked DB-resolution snapshot"
        )

        thread_b.start()
        assert b_entered_wait.wait(timeout=5), (
            "thread B never reached the startup-heal barrier"
        )
        thread_b.join(timeout=0.2)
        assert thread_b.is_alive(), (
            "inbound get_or_create_session must wait for terminal startup heal, "
            "not read/replace the unhealed parent entry"
        )
        assert "entry" not in result

        release_snapshot.set()
        thread_a.join(timeout=5)
        thread_b.join(timeout=5)
        assert not thread_a.is_alive(), "thread A (heal owner) never terminated"
        assert not thread_b.is_alive(), "thread B (inbound waiter) never terminated"
        assert result["entry"].session_id == "sid_child"

    def test_startup_heal_failure_releases_inbound_waiter(self, tmp_path):
        """A DB-resolution exception during startup heal must still release
        Thread B (conservatively — no pruning applied) rather than hanging."""
        db = SessionDB(tmp_path / "state.db")
        self._seed_ended_route_without_recovery(tmp_path, db)
        store = self._make_unloaded_store(tmp_path, db)

        resolve_started = threading.Event()
        release_failure = threading.Event()

        def failing_get_session(session_id):
            resolve_started.set()
            assert release_failure.wait(timeout=5), "test never released the DB failure"
            raise RuntimeError("simulated DB-resolution failure")

        db.get_session = failing_get_session

        b_entered_wait = threading.Event()
        original_wait = store._wait_for_startup_heal_locked

        def instrumented_wait():
            b_entered_wait.set()
            return original_wait()

        store._wait_for_startup_heal_locked = instrumented_wait

        thread_a = threading.Thread(target=store._ensure_loaded)
        result: dict = {}

        def run_b():
            result["entry"] = store.get_or_create_session(self.SOURCE)

        thread_b = threading.Thread(target=run_b)

        thread_a.start()
        assert resolve_started.wait(timeout=5), (
            "thread A never reached the DB-resolution phase"
        )

        thread_b.start()
        assert b_entered_wait.wait(timeout=5), (
            "thread B never reached the startup-heal barrier"
        )
        thread_b.join(timeout=0.2)
        assert thread_b.is_alive(), (
            "inbound waiter must block until heal reaches a terminal state, "
            "even on the failure path"
        )

        release_failure.set()
        thread_a.join(timeout=5)
        thread_b.join(timeout=5)
        assert not thread_a.is_alive(), "thread A never terminated after the DB failure"
        assert not thread_b.is_alive(), "thread B never released after the DB failure"

        # Conservative fallback: heal applied no changes on error, so B
        # resolves the original (unhealed) route rather than hanging or
        # silently losing the session.
        assert result["entry"].session_id == "sid_ended"

    def test_ensure_loaded_does_not_wait_for_in_progress_heal(self, tmp_path):
        """A direct, non-inbound _ensure_loaded() call must not be stranded
        behind an in-flight heal, and must never observe a half-applied
        (partially pruned/repointed) routing index."""
        db = SessionDB(tmp_path / "state.db")
        self._seed_compression_route(tmp_path, db)
        store = self._make_unloaded_store(tmp_path, db)

        reached_snapshot = threading.Event()
        release_snapshot = threading.Event()
        original_resolve = store._resolve_stale_sessions

        def paused_resolve(items):
            actions = original_resolve(items)
            reached_snapshot.set()
            assert release_snapshot.wait(timeout=5), "test never released the owner"
            return actions

        store._resolve_stale_sessions = paused_resolve

        owner = threading.Thread(target=store._ensure_loaded)
        owner.start()
        assert reached_snapshot.wait(timeout=5), (
            "owner never reached the unlocked DB-resolution snapshot"
        )

        direct_caller = threading.Thread(target=store._ensure_loaded)
        direct_caller.start()
        direct_caller.join(timeout=1)
        assert not direct_caller.is_alive(), (
            "_ensure_loaded() must not block on an in-progress startup heal"
        )

        # Pre-heal snapshot observed, never a half-applied one: the
        # compression-parent entry is still present, untouched.
        assert store._entries[self.KEY].session_id == "sid_parent"

        release_snapshot.set()
        owner.join(timeout=5)
        assert not owner.is_alive()
        assert store._entries[self.KEY].session_id == "sid_child"

    def test_has_any_sessions_db_error_fallback_does_not_wait_for_in_progress_heal(
        self, tmp_path
    ):
        """has_any_sessions()'s DB-error fallback must not be stranded behind
        an in-flight heal either."""
        db = SessionDB(tmp_path / "state.db")
        self._seed_compression_route(tmp_path, db)
        store = self._make_unloaded_store(tmp_path, db)

        reached_snapshot = threading.Event()
        release_snapshot = threading.Event()
        original_resolve = store._resolve_stale_sessions

        def paused_resolve(items):
            actions = original_resolve(items)
            reached_snapshot.set()
            assert release_snapshot.wait(timeout=5), "test never released the owner"
            return actions

        store._resolve_stale_sessions = paused_resolve

        owner = threading.Thread(target=store._ensure_loaded)
        owner.start()
        assert reached_snapshot.wait(timeout=5), (
            "owner never reached the unlocked DB-resolution snapshot"
        )

        with patch.object(db, "session_count", side_effect=Exception("db unavailable")):
            result: dict = {}

            def call_has_any():
                result["value"] = store.has_any_sessions()

            fallback_thread = threading.Thread(target=call_has_any)
            fallback_thread.start()
            fallback_thread.join(timeout=1)
            assert not fallback_thread.is_alive(), (
                "has_any_sessions() DB-error fallback must not block on an "
                "in-progress startup heal"
            )
            assert result["value"] is False  # one pre-heal entry: len == 1, not > 1

        release_snapshot.set()
        owner.join(timeout=5)
        assert not owner.is_alive()
