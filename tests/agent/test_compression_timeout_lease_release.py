"""Regression: a timed-out hygiene compressor must not hold the session lease.

Incident (profile ``vpsclone``, Discord, 2026-08-03) — causal chain:

1. A long session crossed the session-hygiene compression threshold and the
   gateway dispatched ``_compress_context`` to an executor with a
   ``CompressionCommitFence``.
2. The configured auxiliary compression model was SYNCHRONOUS (non-streaming),
   so the fence never saw a progress tick. The gateway extended once at 30s and
   gave up at 60s: "made no progress ... continuing without compression".
3. ``try_cancel_before_commit()`` returned True — cancellation won irrevocably,
   so the worker could never mutate the session again — and agent cleanup was
   correctly deferred until the executor future finished.
4. **But the worker still owned the durable per-session compression lease**,
   and its refresher kept extending it for the remaining lifetime of the
   auxiliary call (77.954s and 203.217s observed).
5. The very same turn resumed, and ``SessionDB.append_message`` fails CLOSED on
   a foreign live holder — ``Session ... is being compressed by another
   writer`` — so the turn exited ``session_persistence_failed`` and the user
   lost the reply. DB quick-check/disk/permissions were all healthy: a
   coordination race, not a filesystem fault.

The fix hands the live lease to the fence, so a cancellation that wins before
the commit boundary drops it immediately (holder-qualified), while a worker
that already crossed the boundary keeps its lease and commits normally.

These tests drive the REAL ``compress_context`` against a REAL ``SessionDB``
with a minimal fake agent (same approach as
``tests/run_agent/test_codex_app_server_compaction.py``) so the lease, the
fence, the refresher and the fail-closed append all interact for real.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from agent.conversation_compression import (
    COMPACTION_ABORTED_STATUS,
    COMPACTION_CANCELLED_STATUS,
    COMPACTION_DONE_STATUS,
    COMPACTION_SKIPPED_STATUS,
    COMPACTION_STATUS,
    CompressionCommitFence,
    _CompressionLockLeaseRefresher,
    compress_context,
)
from hermes_state import CompressionSessionBusyError, SessionDB

SUMMARY = "[CONTEXT COMPACTION] summary"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeCompressor:
    """Minimal context engine: blocks in ``compress()`` until released."""

    def __init__(self, gate: threading.Event | None = None):
        self.compression_count = 1
        self.last_compression_rough_tokens = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.awaiting_real_usage_after_compression = False
        self._last_summary_error = None
        self._last_compress_aborted = False
        self._last_aux_model_failure_model = None
        self._last_aux_model_failure_error = None
        self._last_compression_made_progress = True
        self._last_summary_fallback_used = False
        self.gate = gate
        self.entered = threading.Event()
        self.calls = 0
        # Non-committing shapes the real engines can return.
        self.return_input_unchanged = False
        self.return_empty = False

    def compress(
        self,
        messages,
        current_tokens=None,
        focus_topic=None,
        force=False,
        memory_context="",
    ):
        self.calls += 1
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(timeout=10), "compressor gate never released"
        if self._last_compress_aborted or self.return_input_unchanged:
            return messages
        if self.return_empty:
            return []
        return [
            {"role": "user", "content": SUMMARY},
            {"role": "user", "content": "tail"},
        ]


class _FakeAgent:
    """Just enough AIAgent surface for ``compress_context`` to run for real."""

    def __init__(self, db: SessionDB, session_id: str, compressor: _FakeCompressor):
        self.api_mode = "openai"
        self.model = "test/model"
        self.platform = "cli"
        self.log_prefix = ""
        self.tools = []
        self.session_id = session_id
        self._session_db = db
        self.context_compressor = compressor
        self.compression_in_place = True
        self._compression_feasibility_checked = True
        self._cached_system_prompt = "cached prompt"
        self._memory_manager = None
        self._memory_store = None
        self._memory_enabled = False
        self._user_profile_enabled = False
        self._todo_store = SimpleNamespace(format_for_injection=lambda: "")
        self._session_init_model_config = "{}"
        self._active_compression_lock_holder = None
        self._flushed_db_message_ids = set()
        self._last_compaction_in_place = None
        # Fast lease cadence so refresher behaviour is observable in-test.
        self._compression_lock_ttl_seconds = 30.0
        self._compression_lock_refresh_interval = 0.02
        self._compression_activity_heartbeat_interval = 0.05
        self.statuses: list[str] = []
        self.status_events: list[tuple[str, str]] = []
        self.warnings: list[str] = []
        self.memory_commits = 0

    # -- status / prompt plumbing -------------------------------------------
    def status_callback(self, kind, text):
        self.status_events.append((kind, text))

    def _emit_status(self, message):
        self.statuses.append(message)
        self.status_callback("lifecycle", message)

    def _emit_warning(self, message):
        self.warnings.append(message)
        self.status_callback("warn", message)

    def _build_system_prompt(self, system_message):
        return "built prompt"

    def _invalidate_system_prompt(self):
        self._cached_system_prompt = None

    def _touch_activity(self, desc):
        pass

    def commit_memory_session(self, messages):
        self.memory_commits += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_db(tmp_path: Path, session_id: str) -> SessionDB:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id, source="discord")
    db.append_message(session_id, "user", "hello")
    return db


def _messages(n: int = 12) -> list:
    return [{"role": "user", "content": f"m{i}"} for i in range(n)]


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _terminal_status(agent: _FakeAgent):
    edges = [text for kind, text in agent.status_events if kind == "compacted"]
    assert len(edges) <= 1, f"more than one terminal edge: {edges!r}"
    return edges[0] if edges else None


class _Worker:
    """Runs compress_context on a thread, like the gateway's executor."""

    def __init__(self, agent, messages, fence, **kwargs):
        self.agent = agent
        self.messages = messages
        self.fence = fence
        self.kwargs = kwargs
        self.result = None
        self.error = None
        self.thread = threading.Thread(
            target=self._run, name="hygiene-compression", daemon=True
        )

    def _run(self):
        try:
            self.result = compress_context(
                self.agent,
                self.messages,
                "system",
                approx_tokens=120_000,
                force=True,
                commit_fence=self.fence,
                **self.kwargs,
            )
        except BaseException as exc:  # pragma: no cover - surfaced by asserts
            self.error = exc

    def start(self):
        self.thread.start()
        return self

    def join(self, timeout: float = 10.0):
        self.thread.join(timeout=timeout)
        assert not self.thread.is_alive(), "compression worker never finished"
        assert self.error is None, f"worker raised: {self.error!r}"
        return self.result


# ---------------------------------------------------------------------------
# 1. The incident: cancel wins → lease gone before the turn persists
# ---------------------------------------------------------------------------


def test_cancelled_hygiene_worker_releases_lease_before_the_turn_resumes(tmp_path: Path):
    """The durable invariant the vpsclone incident violated.

    While the non-streaming worker is STILL RUNNING (exactly the 78s/203s
    window), a cancellation that won before the commit boundary must leave no
    live lease behind, and the same turn's ``append_message`` must succeed.
    """
    sid = "VPSCLONE_HYGIENE"
    db = _new_db(tmp_path, sid)
    gate = threading.Event()
    compressor = _FakeCompressor(gate)
    agent = _FakeAgent(db, sid, compressor)
    fence = CompressionCommitFence()
    messages = _messages()

    worker = _Worker(agent, messages, fence).start()
    assert compressor.entered.wait(timeout=5), "compressor never started"
    # The lease is live and owned by the worker — the state the gateway's
    # inactivity timeout fires in.
    assert _wait_until(lambda: db.get_compression_lock_holder(sid) is not None)
    worker_holder = db.get_compression_lock_holder(sid)

    # Gateway hygiene path: non-blocking cancel, then the off-loop release.
    assert fence.try_cancel_before_commit() is True
    assert fence.release_cancelled_lease() is True

    # ── The invariant ──────────────────────────────────────────────────────
    # The worker is still inside its auxiliary call, but its lease is gone.
    assert compressor.calls == 1
    assert worker.thread.is_alive()
    assert db.get_compression_lock_holder(sid) is None

    # ...so the turn that cancelled it can persist its reply. Before the fix
    # this raised CompressionSessionBusyError → session_persistence_failed.
    row_id = db.append_message(sid, "assistant", "the reply the user lost")
    assert isinstance(row_id, int)

    # Let the late worker return; it must abort before any mutation.
    gate.set()
    compressed, prompt = worker.join()
    assert compressed is messages
    assert prompt == "cached prompt"
    assert agent.session_id == sid
    assert agent._last_compaction_in_place is False
    assert agent.memory_commits == 0
    # The transcript still holds the original turn plus the reply — no
    # compaction, no rotation, nothing dropped.
    live = db.get_messages_as_conversation(sid)
    assert [m["content"] for m in live] == ["hello", "the reply the user lost"]
    assert db.get_compression_lock_holder(sid) is None
    assert worker_holder is not None


def test_cancelled_worker_emits_no_false_success_status(tmp_path: Path):
    """A cancelled compaction closes its phase truthfully, exactly once."""
    sid = "CANCELLED_STATUS"
    db = _new_db(tmp_path, sid)
    gate = threading.Event()
    compressor = _FakeCompressor(gate)
    agent = _FakeAgent(db, sid, compressor)
    fence = CompressionCommitFence()

    worker = _Worker(agent, _messages(), fence).start()
    assert compressor.entered.wait(timeout=5)
    assert fence.try_cancel_before_commit() is True
    fence.release_cancelled_lease()
    gate.set()
    worker.join()

    assert agent.status_events[0] == ("lifecycle", COMPACTION_STATUS)
    # The phase still closes (kind="compacted" retires the desktop/TUI
    # "Summarizing…" indicator) but never claims a completed compaction.
    assert _terminal_status(agent) == COMPACTION_CANCELLED_STATUS
    assert COMPACTION_DONE_STATUS not in [t for _k, t in agent.status_events]


# ---------------------------------------------------------------------------
# 2. Late cleanup is holder-qualified and idempotent
# ---------------------------------------------------------------------------


def test_late_worker_cleanup_cannot_evict_a_newer_holder(tmp_path: Path):
    """The late worker's own release must not touch the live turn's lease."""
    sid = "LATE_WORKER_CLEANUP"
    db = _new_db(tmp_path, sid)
    gate = threading.Event()
    compressor = _FakeCompressor(gate)
    agent = _FakeAgent(db, sid, compressor)
    fence = CompressionCommitFence()

    worker = _Worker(agent, _messages(), fence).start()
    assert compressor.entered.wait(timeout=5)
    assert _wait_until(lambda: db.get_compression_lock_holder(sid) is not None)

    assert fence.try_cancel_before_commit() is True
    assert fence.release_cancelled_lease() is True

    # The live turn now runs its own compression and takes the lease.
    assert db.try_acquire_compression_lock(sid, "live-turn-holder") is True

    # The cancelled worker returns late and runs its cleanup.
    gate.set()
    worker.join()

    # The newer holder is untouched, and its own append still works.
    assert db.get_compression_lock_holder(sid) == "live-turn-holder"
    assert isinstance(
        db.append_message(
            sid, "assistant", "new holder writes",
            compression_lock_holder="live-turn-holder",
        ),
        int,
    )
    # A foreign writer is still fenced out: fail-closed persistence intact.
    try:
        db.append_message(sid, "assistant", "foreign writer")
    except CompressionSessionBusyError:
        pass
    else:  # pragma: no cover - defended below
        raise AssertionError("append_message must fail closed on a foreign holder")

    db.release_compression_lock(sid, "live-turn-holder")


def test_release_cancelled_lease_is_idempotent(tmp_path: Path):
    """Repeated releases (and a late worker release) are safe no-ops."""
    sid = "IDEMPOTENT_RELEASE"
    db = _new_db(tmp_path, sid)
    gate = threading.Event()
    compressor = _FakeCompressor(gate)
    agent = _FakeAgent(db, sid, compressor)
    fence = CompressionCommitFence()

    worker = _Worker(agent, _messages(), fence).start()
    assert compressor.entered.wait(timeout=5)
    assert _wait_until(lambda: db.get_compression_lock_holder(sid) is not None)

    assert fence.try_cancel_before_commit() is True
    assert fence.release_cancelled_lease() is True
    # Second call has nothing pending — no second DELETE, no exception.
    assert fence.release_cancelled_lease() is False
    assert fence.release_cancelled_lease() is False

    gate.set()
    worker.join()
    assert db.get_compression_lock_holder(sid) is None


def test_blocking_cancel_releases_the_lease_inline(tmp_path: Path):
    """``cancel_before_commit`` (worker-thread callers) releases inline."""
    sid = "BLOCKING_CANCEL"
    db = _new_db(tmp_path, sid)
    gate = threading.Event()
    compressor = _FakeCompressor(gate)
    agent = _FakeAgent(db, sid, compressor)
    fence = CompressionCommitFence()

    worker = _Worker(agent, _messages(), fence).start()
    assert compressor.entered.wait(timeout=5)
    assert _wait_until(lambda: db.get_compression_lock_holder(sid) is not None)

    assert fence.cancel_before_commit() is True
    # No follow-up call needed: the lease is already gone.
    assert db.get_compression_lock_holder(sid) is None

    gate.set()
    worker.join()
    assert db.get_compression_lock_holder(sid) is None


# ---------------------------------------------------------------------------
# 3. Commit already started: consume the result, never touch the live lease
# ---------------------------------------------------------------------------


def test_commit_already_started_keeps_the_lease_and_commits(tmp_path: Path):
    """A worker past the boundary must finish — not be cancelled or released."""
    sid = "COMMIT_IN_FLIGHT"
    db = _new_db(tmp_path, sid)
    compressor = _FakeCompressor()
    agent = _FakeAgent(db, sid, compressor)
    fence = CompressionCommitFence()

    # Simulate the worker having entered the commit boundary. ``begin_commit``
    # HOLDS the fence lock until ``finish_commit``, so the blocking canceller
    # must run on a DIFFERENT thread — that is the real topology (gateway
    # canceller vs. executor worker), and calling it inline here would park
    # this thread on a lock only this thread can release.
    assert fence.begin_commit() is True
    cancel_entered = threading.Event()
    cancel_result: list = []

    def _blocking_cancel() -> None:
        cancel_entered.set()
        cancel_result.append(fence.cancel_before_commit())

    canceller = threading.Thread(
        target=_blocking_cancel, name="hygiene-cancel", daemon=True
    )
    try:
        # The gateway's non-blocking poll yields instead of blocking.
        assert fence.try_cancel_before_commit() is None
        canceller.start()
        assert cancel_entered.wait(timeout=5), "canceller thread never ran"
        # It must WAIT for the boundary, not cancel it: while this thread owns
        # the commit, the blocking cancel cannot return.
        canceller.join(timeout=0.25)
        assert canceller.is_alive(), (
            "cancel_before_commit must block until the commit boundary ends"
        )
        assert cancel_result == []
    finally:
        fence.finish_commit()

    # Once the boundary closes it returns — having lost the race outright.
    canceller.join(timeout=5)
    assert not canceller.is_alive(), "blocking cancel never returned"
    assert cancel_result == [False]

    # A loser never arms a release, so no lease can be dropped by it.
    assert fence.release_cancelled_lease() is False

    # A fresh fence on the same session commits the compaction for real.
    live_fence = CompressionCommitFence()
    compressed, prompt = compress_context(
        agent, _messages(), "system",
        approx_tokens=120_000, force=True, commit_fence=live_fence,
    )

    assert [m["content"] for m in compressed][:1] == [SUMMARY]
    # Built-in memory is disabled on this fake, so the cached prompt still
    # reflects current memory and compaction deliberately RETAINS it verbatim
    # to preserve the local-backend KV-cache prefix (pre-existing behaviour of
    # _cached_prompt_reflects_builtin_memory — untouched by the lease fix).
    assert prompt == "cached prompt"
    assert agent._cached_system_prompt == "cached prompt"
    assert agent._last_compaction_in_place is True
    assert agent.memory_commits == 1
    assert _terminal_status(agent) == COMPACTION_DONE_STATUS
    # Committed cleanly: lease released, compacted transcript is live.
    assert db.get_compression_lock_holder(sid) is None
    assert [m["content"] for m in db.get_messages_as_conversation(sid)] == [
        SUMMARY,
        "tail",
    ]


def test_cancel_after_commit_boundary_consumes_the_successful_result(tmp_path: Path):
    """Cancelling while the commit runs waits for it and keeps the compaction."""
    sid = "CANCEL_DURING_COMMIT"
    db = _new_db(tmp_path, sid)
    compressor = _FakeCompressor()
    agent = _FakeAgent(db, sid, compressor)
    fence = CompressionCommitFence()

    commit_entered = threading.Event()
    release_commit = threading.Event()
    original_archive = db.archive_and_compact

    def _slow_archive(session_id, compacted):
        commit_entered.set()
        assert release_commit.wait(timeout=10)
        return original_archive(session_id, compacted)

    db.archive_and_compact = _slow_archive

    worker = _Worker(agent, _messages(), fence).start()
    assert commit_entered.wait(timeout=5), "commit boundary never entered"

    # The worker owns the boundary: the poll must yield, not cancel.
    assert fence.try_cancel_before_commit() is None
    assert db.get_compression_lock_holder(sid) is not None

    release_commit.set()
    compressed, _prompt = worker.join()

    # The successful result is consumed, not discarded.
    assert [m["content"] for m in compressed][:1] == [SUMMARY]
    assert agent._last_compaction_in_place is True
    assert _terminal_status(agent) == COMPACTION_DONE_STATUS
    assert db.get_compression_lock_holder(sid) is None
    assert [m["content"] for m in db.get_messages_as_conversation(sid)] == [
        SUMMARY,
        "tail",
    ]


# ---------------------------------------------------------------------------
# 4. The refresher can neither resurrect nor prolong a released lease
# ---------------------------------------------------------------------------


def test_refresher_cannot_recreate_a_released_lease(tmp_path: Path):
    """A running refresher must not revive a lease that was released."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "REFRESHER_NO_RESURRECT"
    db.create_session(sid, source="cli")
    holder = "worker-holder"
    assert db.try_acquire_compression_lock(sid, holder, ttl_seconds=30.0) is True

    refresher = _CompressionLockLeaseRefresher(
        db, sid, holder, 30.0, refresh_interval_seconds=0.01
    ).start()
    try:
        time.sleep(0.05)  # let it tick a few times on a live lease
        assert db.get_compression_lock_holder(sid) == holder

        db.release_compression_lock(sid, holder)
        time.sleep(0.08)  # more ticks, now against a deleted row
        assert db.get_compression_lock_holder(sid) is None, (
            "refresh_compression_lock is an UPDATE — it must never re-insert"
        )

        # A new holder takes the lease while the stale refresher still runs.
        assert db.try_acquire_compression_lock(sid, "new-holder", ttl_seconds=5.0)
        expiry_before = db._conn.execute(
            "SELECT expires_at FROM compression_locks WHERE session_id = ?",
            (sid,),
        ).fetchone()[0]
        time.sleep(0.08)
        row = db._conn.execute(
            "SELECT holder, expires_at FROM compression_locks WHERE session_id = ?",
            (sid,),
        ).fetchone()
        assert row[0] == "new-holder"
        assert row[1] == expiry_before, "stale refresher prolonged a foreign lease"
    finally:
        refresher.stop()


def test_refresher_never_starts_for_an_already_released_lease(tmp_path: Path):
    """A lease cancelled before compress() starts must not get a refresher."""
    sid = "NO_REFRESHER_AFTER_CANCEL"
    db = _new_db(tmp_path, sid)
    compressor = _FakeCompressor()
    agent = _FakeAgent(db, sid, compressor)
    fence = CompressionCommitFence()
    # The caller gave up before the worker ever reached the lease.
    assert fence.cancel_before_commit() is True

    messages = _messages()
    compressed, prompt = compress_context(
        agent, messages, "system",
        approx_tokens=120_000, force=True, commit_fence=fence,
    )

    assert compressed is messages
    assert prompt == "cached prompt"
    assert compressor.calls == 0, "a cancelled caller must not run the summary"
    assert db.get_compression_lock_holder(sid) is None
    assert agent._active_compression_lock_holder is None
    assert agent._last_compaction_in_place is False
    assert _terminal_status(agent) == COMPACTION_CANCELLED_STATUS
    # Nothing mutated; a normal append still works.
    assert isinstance(db.append_message(sid, "assistant", "reply"), int)


# ---------------------------------------------------------------------------
# 5. Truthful lifecycle for the other non-committed outcomes
# ---------------------------------------------------------------------------


def test_lock_contention_emits_no_false_success_status(tmp_path: Path):
    """A lock-contended no-op must not report a completed compaction."""
    sid = "LOCK_CONTENDED"
    db = _new_db(tmp_path, sid)
    compressor = _FakeCompressor()
    agent = _FakeAgent(db, sid, compressor)
    assert db.try_acquire_compression_lock(sid, "other-writer") is True

    messages = _messages()
    compressed, prompt = compress_context(
        agent, messages, "system", approx_tokens=120_000, force=True,
    )

    assert compressed is messages
    assert prompt == "cached prompt"
    assert compressor.calls == 0
    assert agent._compression_skipped_due_to_lock == "other-writer"
    assert _terminal_status(agent) == COMPACTION_SKIPPED_STATUS
    # The winner's lease is untouched by the loser.
    assert db.get_compression_lock_holder(sid) == "other-writer"


def test_aborted_summary_emits_no_false_success_status(tmp_path: Path):
    """An aborted summary closes the phase as aborted, not complete."""
    sid = "ABORTED_SUMMARY"
    db = _new_db(tmp_path, sid)
    compressor = _FakeCompressor()
    compressor._last_compress_aborted = True
    compressor._last_summary_error = "auxiliary model unavailable"
    agent = _FakeAgent(db, sid, compressor)

    messages = _messages()
    compressed, prompt = compress_context(
        agent, messages, "system", approx_tokens=120_000, force=True,
    )

    assert compressed is messages
    assert prompt == "cached prompt"
    assert _terminal_status(agent) == COMPACTION_ABORTED_STATUS
    assert any("Compression aborted" in w for w in agent.warnings)
    assert db.get_compression_lock_holder(sid) is None


def test_no_progress_pass_emits_no_false_success_status(tmp_path: Path):
    """A pass that skips the boundary rewrite is a skip, not a success."""
    sid = "NO_PROGRESS"
    db = _new_db(tmp_path, sid)
    compressor = _FakeCompressor()
    compressor.return_input_unchanged = True
    agent = _FakeAgent(db, sid, compressor)

    messages = _messages()
    compressed, prompt = compress_context(
        agent, messages, "system", approx_tokens=120_000, force=True,
    )

    assert compressed is messages
    assert prompt == "cached prompt"
    assert compressor.calls == 1
    assert _terminal_status(agent) == COMPACTION_SKIPPED_STATUS
    assert db.get_compression_lock_holder(sid) is None
    # The transcript was never rewritten, so the turn keeps writing normally.
    assert isinstance(db.append_message(sid, "assistant", "reply"), int)


def test_empty_transcript_emits_no_false_success_status(tmp_path: Path):
    """Refusing to rotate on an empty transcript closes the phase as aborted."""
    sid = "EMPTY_TRANSCRIPT"
    db = _new_db(tmp_path, sid)
    compressor = _FakeCompressor()
    compressor.return_empty = True
    agent = _FakeAgent(db, sid, compressor)

    messages = _messages()
    compressed, prompt = compress_context(
        agent, messages, "system", approx_tokens=120_000, force=True,
    )

    assert compressed is messages
    assert prompt == "cached prompt"
    assert _terminal_status(agent) == COMPACTION_ABORTED_STATUS
    assert any("empty transcript" in w for w in agent.warnings)
    # The parent stays resumable and unlocked.
    assert db.get_compression_lock_holder(sid) is None
    assert [m["content"] for m in db.get_messages_as_conversation(sid)] == ["hello"]


# ---------------------------------------------------------------------------
# 6. Healthy control
# ---------------------------------------------------------------------------


def test_healthy_sqlite_control_compacts_and_persists(tmp_path: Path):
    """Control: nothing about the fix disturbs a healthy compaction."""
    sid = "HEALTHY_CONTROL"
    db = _new_db(tmp_path, sid)
    compressor = _FakeCompressor()
    agent = _FakeAgent(db, sid, compressor)

    compressed, prompt = compress_context(
        agent, _messages(), "system", approx_tokens=120_000, force=True,
    )

    assert [m["content"] for m in compressed][:1] == [SUMMARY]
    # See above: the KV-cache-preserving keep-prompt path is the healthy
    # default when built-in memory is disabled.
    assert prompt == "cached prompt"
    assert agent._last_compaction_in_place is True
    assert _terminal_status(agent) == COMPACTION_DONE_STATUS
    assert db.get_compression_lock_holder(sid) is None
    # The compacted set is live and the turn can keep writing.
    assert isinstance(db.append_message(sid, "assistant", "next reply"), int)
    assert [m["content"] for m in db.get_messages_as_conversation(sid)] == [
        SUMMARY,
        "tail",
        "next reply",
    ]
    # Pre-compaction rows are archived, not destroyed.
    archived = db.get_messages_as_conversation(sid, include_inactive=True)
    assert any(m["content"] == "hello" for m in archived)


# ---------------------------------------------------------------------------
# 7. Fence unit semantics
# ---------------------------------------------------------------------------


def test_unbind_lease_is_holder_scoped():
    """A late worker cannot detach a lease that a newer holder rebound."""
    fence = CompressionCommitFence()
    calls = []
    assert fence.bind_lease(lambda: calls.append("new"), holder="new-holder") is True

    fence.unbind_lease("stale-holder")  # must be ignored
    assert fence.cancel_before_commit() is True
    assert calls == ["new"]

    fence2 = CompressionCommitFence()
    calls2 = []
    assert fence2.bind_lease(lambda: calls2.append("x"), holder="h") is True
    fence2.unbind_lease("h")
    assert fence2.cancel_before_commit() is True
    assert calls2 == []


def test_bind_lease_is_refused_after_cancellation():
    """Binding after a winning cancel must fail so the worker releases now."""
    fence = CompressionCommitFence()
    assert fence.cancel_before_commit() is True
    assert fence.bind_lease(lambda: None, holder="late") is False
    assert fence.release_cancelled_lease() is False


def test_bind_lease_is_allowed_while_a_commit_runs():
    """The commit boundary holds ``_lock``; lease binding must not deadlock."""
    fence = CompressionCommitFence()
    assert fence.begin_commit() is True
    try:
        assert fence.bind_lease(lambda: None, holder="h") is True
        fence.unbind_lease("h")
    finally:
        fence.finish_commit()


def test_lease_release_failure_is_contained():
    """A failing release must not propagate into the canceller."""
    fence = CompressionCommitFence()

    def _boom():
        raise RuntimeError("db down")

    assert fence.bind_lease(_boom, holder="h") is True
    assert fence.cancel_before_commit() is True  # swallowed, cancel still wins
    assert fence.release_cancelled_lease() is False


# ---------------------------------------------------------------------------
# 8. The other half of the incident: the retry that burned the wait budget
# ---------------------------------------------------------------------------
#
# The lease fix above stops a cancelled worker from wedging the turn. This
# section covers WHY the worker got cancelled at all: the configured aux model
# was SYNCHRONOUS, so a same-provider transient retry reported no progress and
# consumed the caller's whole inactivity window — while a healthy configured
# fallback sat unused (``fallback_used=false``).


class _FakeTimeout(Exception):
    """Type name contains 'Timeout' — what ``_is_timeout_error`` keys on."""


def _with_task_config(config, fn):
    """Run ``fn`` with ``_get_auxiliary_task_config`` replaced, then restore."""
    import agent.auxiliary_client as aux

    original = aux._get_auxiliary_task_config
    if isinstance(config, BaseException):
        def _fake(_task):
            raise config
    else:
        def _fake(_task):
            return config
    aux._get_auxiliary_task_config = _fake
    try:
        return fn(aux)
    finally:
        aux._get_auxiliary_task_config = original


def test_compression_timeout_skips_the_same_provider_retry():
    """#54465: a full-budget timeout must not be retried on the same provider.

    True even with no configured fallback — the retry doubles an already
    user-visible stall, and the ordinary chain still runs after the raise.
    """
    def _check(aux):
        assert aux._compression_prefers_immediate_failover(
            "compression", _FakeTimeout("request timed out")
        ) is True

    _with_task_config({}, _check)


def test_compression_transient_blip_prefers_a_configured_fallback():
    """A cheap 5xx still fails over when a configured chain can serve it."""
    def _check(aux):
        assert aux._has_configured_fallback_chain("compression") is True
        assert aux._compression_prefers_immediate_failover(
            "compression", RuntimeError("503 UNAVAILABLE")
        ) is True

    _with_task_config(
        {"fallback_chain": [{"provider": "openrouter", "model": "m"}]}, _check
    )


def test_compression_transient_blip_retries_when_no_fallback_is_configured():
    """With nothing better to switch to, the same-provider retry is kept."""
    def _check(aux):
        assert aux._has_configured_fallback_chain("compression") is False
        assert aux._compression_prefers_immediate_failover(
            "compression", RuntimeError("503 UNAVAILABLE")
        ) is False

    # Entries without a usable provider are not a fallback chain.
    _with_task_config({"fallback_chain": [{"model": "m"}, "junk"]}, _check)
    _with_task_config({"fallback_chain": "not-a-list"}, _check)
    _with_task_config({}, _check)


def test_other_auxiliary_tasks_keep_the_ordinary_transient_retry():
    """The failover shortcut is scoped to compression alone."""
    def _check(aux):
        for task in ("title", "memory", None, ""):
            assert aux._compression_prefers_immediate_failover(
                task, RuntimeError("503 UNAVAILABLE")
            ) is False
            assert aux._compression_prefers_immediate_failover(
                task, _FakeTimeout("timed out")
            ) is False
        assert aux._has_configured_fallback_chain(None) is False

    _with_task_config(
        {"fallback_chain": [{"provider": "openrouter"}]}, _check
    )


def test_unreadable_fallback_config_preserves_the_retry():
    """Config that cannot be read means 'no fallback' — never a crash."""
    def _check(aux):
        assert aux._has_configured_fallback_chain("compression") is False
        assert aux._compression_prefers_immediate_failover(
            "compression", RuntimeError("503 UNAVAILABLE")
        ) is False
        # A timeout is still short-circuited without consulting config.
        assert aux._compression_prefers_immediate_failover(
            "compression", _FakeTimeout("timed out")
        ) is True

    _with_task_config(OSError("config.yaml unreadable"), _check)


def test_both_sync_and_async_call_paths_use_the_shared_failover_rule():
    """``call_llm`` and ``async_call_llm`` must not drift apart."""
    import inspect

    import agent.auxiliary_client as aux

    for fn in (aux.call_llm, aux.async_call_llm):
        source = inspect.getsource(fn)
        assert "_compression_prefers_immediate_failover" in source, (
            f"{fn.__name__} no longer routes compression through the shared "
            "immediate-failover rule"
        )
        # The old inline predicate must be gone from both paths.
        assert 'task == "compression" and _is_timeout_error' not in source
