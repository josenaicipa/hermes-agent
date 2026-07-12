"""G3 startup-heal lock-freedom tests.

The G3 contract (see gateway/session.py ``_run_startup_heal_locked`` and the
startup-heal barrier) requires that NO SQLite read/write and no blocking wait
runs while ``SessionStore._lock`` is held, so a slow or blocked ``state.db`` can
never strand every gateway caller behind the global routing lock.

The prior barrier tests (``test_session_store_stale_prune.TestStartupHealBarrier``)
only instrument the heal's DB-resolution *reads*. These tests instrument the
remaining SQLite operations on the G3 path and prove ``_lock`` is free during
each of them:

  1. initial ``load_gateway_routing_entries`` (startup load)
  2. startup-heal persistence ``replace_gateway_routing_entries`` (``_save``)
  3. inbound ended-session DB lookup (``_is_session_ended_in_db`` -> ``get_session``)

Plus the snapshot/version guard that keeps an out-of-lock heal persist from
clobbering a concurrent in-lock route mutation.
"""

import json
import threading
from datetime import datetime, timedelta

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionSource, SessionStore
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KEY = "agent:main:telegram:dm:919191"
SOURCE = SessionSource(
    platform=Platform.TELEGRAM,
    chat_id="919191",
    chat_type="dm",
    user_id="919191",
)


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


def _make_unloaded_store(tmp_path, db) -> SessionStore:
    """A SessionStore that has NOT loaded yet — the next _ensure_loaded() /
    get_or_create_session() call performs the real startup load and becomes the
    startup-heal owner."""
    config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
    store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = db
    return store


def _probe_lock_free(store, recorder):
    """Record whether ``store._lock`` is free *right now* (from this thread).

    ``_lock`` is a plain, non-reentrant ``threading.Lock``: if the calling
    thread already holds it (i.e. the SQLite op ran nested inside the routing
    lock), ``acquire(blocking=False)`` returns ``False``. A ``True`` proves the
    op ran with the routing lock released.
    """
    acquired = store._lock.acquire(blocking=False)
    recorder.append(acquired)
    if acquired:
        store._lock.release()


# ---------------------------------------------------------------------------
# 1. Initial routing load
# ---------------------------------------------------------------------------

class TestInitialLoadLockFree:
    def test_initial_load_runs_without_routing_lock(self, tmp_path):
        """``load_gateway_routing_entries`` (the first startup DB read) must run
        with ``SessionStore._lock`` free."""
        db = SessionDB(tmp_path / "state.db")
        store = _make_unloaded_store(tmp_path, db)

        # Seed one live routing entry so the loader returns real data.
        entry = _make_entry(KEY, "sid_live")
        db.replace_gateway_routing_entries(
            {KEY: json.dumps(entry.to_dict())}, scope=store._routing_scope()
        )

        real_loader = db.load_gateway_routing_entries
        lock_free: list = []

        def instrumented_loader(*, scope=""):
            _probe_lock_free(store, lock_free)
            return real_loader(scope=scope)

        db.load_gateway_routing_entries = instrumented_loader

        store._ensure_loaded()

        assert lock_free, "initial load_gateway_routing_entries was never invoked"
        assert all(lock_free), (
            "SessionStore._lock must be free during every "
            "load_gateway_routing_entries call, not serialized behind it"
        )
        # And the entry actually loaded.
        assert KEY in store._entries
        assert store._entries[KEY].session_id == "sid_live"


# ---------------------------------------------------------------------------
# 2. Startup-heal persistence
# ---------------------------------------------------------------------------

class TestStartupHealPersistLockFree:
    def _seed_compression_route(self, tmp_path, db):
        """Real SQLite compression parent -> live child, routed via a stale
        sessions.json entry pointing at the (ended) parent. Startup heal
        repoints parent -> child, which is a change and therefore persists."""
        db.create_session("sid_parent", source="telegram")
        db.end_session("sid_parent", "compression")
        db.create_session("sid_child", source="telegram", parent_session_id="sid_parent")

        entry = _make_entry(KEY, "sid_parent")
        entry.origin = SOURCE
        (tmp_path / "sessions.json").write_text(
            json.dumps({KEY: entry.to_dict()}, indent=2), encoding="utf-8"
        )

    def test_startup_heal_persist_runs_without_routing_lock(self, tmp_path):
        """The startup-heal persist (``_save`` ->
        ``replace_gateway_routing_entries``) must run with ``_lock`` free."""
        db = SessionDB(tmp_path / "state.db")
        self._seed_compression_route(tmp_path, db)
        store = _make_unloaded_store(tmp_path, db)

        real_replace = db.replace_gateway_routing_entries
        lock_free: list = []

        def instrumented_replace(entries, *, scope=""):
            _probe_lock_free(store, lock_free)
            return real_replace(entries, scope=scope)

        db.replace_gateway_routing_entries = instrumented_replace

        store._ensure_loaded()

        assert lock_free, "startup-heal persistence never wrote to state.db"
        assert all(lock_free), (
            "SessionStore._lock must be free during the startup-heal "
            "replace_gateway_routing_entries persist"
        )
        # And the repoint was actually persisted.
        assert store._entries[KEY].session_id == "sid_child"


# ---------------------------------------------------------------------------
# 3. Inbound ended-session DB lookup (#54878, live-gateway variant)
# ---------------------------------------------------------------------------

class TestInboundEndedLookupLockFree:
    def test_inbound_ended_session_lookup_runs_without_routing_lock(self, tmp_path):
        """``get_or_create_session``'s ``_is_session_ended_in_db`` DB lookup must
        run with ``_lock`` free."""
        db = SessionDB(tmp_path / "state.db")

        # A live session, routed live in sessions.json. It survives startup heal.
        db.create_session("sid_live", source="telegram")
        entry = _make_entry(KEY, "sid_live")
        entry.origin = SOURCE
        (tmp_path / "sessions.json").write_text(
            json.dumps({KEY: entry.to_dict()}, indent=2), encoding="utf-8"
        )

        store = _make_unloaded_store(tmp_path, db)
        store._ensure_loaded()  # load + heal; the live entry survives
        assert store._entries[KEY].session_id == "sid_live"

        # The session now ends in the DB while the gateway stays alive (#54878).
        db.end_session("sid_live", "agent_close")

        real_get_session = db.get_session
        observed: list = []  # (session_id, lock_was_free)

        def instrumented_get_session(session_id):
            acquired = store._lock.acquire(blocking=False)
            observed.append((session_id, acquired))
            if acquired:
                store._lock.release()
            return real_get_session(session_id)

        db.get_session = instrumented_get_session

        store.get_or_create_session(SOURCE)

        ended_checks = [free for (sid, free) in observed if sid == "sid_live"]
        assert ended_checks, (
            "inbound ended-session DB lookup (_is_session_ended_in_db) was "
            "never invoked"
        )
        assert all(ended_checks), (
            "SessionStore._lock must be free during the inbound "
            "ended-session DB lookup, not held around get_session"
        )


# ---------------------------------------------------------------------------
# 4. Snapshot/version guard — out-of-lock persist must not clobber a
#    concurrent in-lock route mutation.
# ---------------------------------------------------------------------------

class TestPersistSnapshotVersionGuard:
    def test_stale_version_snapshot_is_dropped(self, tmp_path):
        """A snapshot with a version older than the last persisted one must be
        dropped rather than overwriting the newer routing state."""
        db = SessionDB(tmp_path / "state.db")
        store = _make_unloaded_store(tmp_path, db)
        store._ensure_loaded()
        scope = store._routing_scope()

        newer = {"k_new": _make_entry("k_new", "sid_new").to_dict()}
        older = {"k_old": _make_entry("k_old", "sid_old").to_dict()}

        # Persist the newer snapshot first, then a stale (lower-version) one.
        store._persist_routing_data(newer, 10_000)
        store._persist_routing_data(older, 1)

        loaded = db.load_gateway_routing_entries(scope=scope)
        assert "k_new" in loaded, "newer routing snapshot was lost"
        assert "k_old" not in loaded, (
            "a stale-version snapshot clobbered a newer persisted routing state"
        )

    def test_heal_persist_does_not_clobber_concurrent_route(self, tmp_path):
        """While the startup heal is persisting its snapshot OUTSIDE ``_lock``, a
        concurrent direct-caller route mutation + save must not be clobbered by
        the heal's (older) snapshot write."""
        db = SessionDB(tmp_path / "state.db")
        # Compression route so the heal makes a change and therefore persists.
        db.create_session("sid_parent", source="telegram")
        db.end_session("sid_parent", "compression")
        db.create_session("sid_child", source="telegram", parent_session_id="sid_parent")
        entry = _make_entry(KEY, "sid_parent")
        entry.origin = SOURCE
        (tmp_path / "sessions.json").write_text(
            json.dumps({KEY: entry.to_dict()}, indent=2), encoding="utf-8"
        )

        store = _make_unloaded_store(tmp_path, db)
        scope = store._routing_scope()

        real_persist = store._persist_routing_data
        heal_reached_persist = threading.Event()
        release_heal_persist = threading.Event()
        first = {"seen": False}

        def instrumented_persist(data, generation):
            # The heal owner is the first caller; pause it BEFORE the actual
            # write (with _lock already released) so a concurrent save can land.
            if not first["seen"]:
                first["seen"] = True
                heal_reached_persist.set()
                assert release_heal_persist.wait(timeout=5), (
                    "test never released the heal persist"
                )
            return real_persist(data, generation)

        store._persist_routing_data = instrumented_persist

        thread_a = threading.Thread(target=store._ensure_loaded)
        thread_a.start()
        assert heal_reached_persist.wait(timeout=5), (
            "startup heal never reached the out-of-lock persist"
        )

        # _lock must be free while the heal is paused at its out-of-lock persist.
        acquired = store._lock.acquire(timeout=5)
        assert acquired, "routing lock was held during the out-of-lock heal persist"
        try:
            store._entries["concurrent_key"] = _make_entry(
                "concurrent_key", "sid_concurrent"
            )
            store._save()  # newer snapshot (higher version), persisted now
        finally:
            store._lock.release()

        release_heal_persist.set()
        thread_a.join(timeout=5)
        assert not thread_a.is_alive(), "startup-heal owner thread never terminated"

        loaded = db.load_gateway_routing_entries(scope=scope)
        assert "concurrent_key" in loaded, (
            "the out-of-lock startup-heal persist clobbered a concurrently-added "
            "route (missing snapshot/version guard)"
        )
        # The heal's own repoint is still present too.
        assert store._entries[KEY].session_id == "sid_child"
