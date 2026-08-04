"""Critical transcript writes must survive a long ``state.db`` lock holder.

Incident context (vpsclone, 2026-08-03).  Two separate windows the same day:

* 13:51 — ``append_message`` raised ``database is locked`` after ~16.2 s (15
  retries against a 1 s ``busy_timeout``).  ``tests/state/
  test_schema_init_lock_retry.py`` fixed the *constructor* side of that and
  said in its own docstring that it "does not change that path and does not
  claim to fix it".  This module is that path.
* 23:04-23:05 — token accounting, two ``append_message`` calls and a gateway
  routing save all failed with ``database is locked``; turns ended
  ``session_persistence_failed`` and a fail-closed reply was shown.  The
  database was healthy throughout: a transcript row committed at 23:04:54.480
  in the *middle* of the failure window, and a post-incident ``BEGIN
  IMMEDIATE; ROLLBACK`` probe took 0.05 ms.

**What is NOT proven** (and therefore not asserted anywhere below): who held
the lock.  ``hermes insights --days 30`` is correlated, not convicted, and no
particular statement was ever attributed.  So there is no test here for "the
holder is gone" and no budget derived from "the holder took ~55 s".

The proven bug class is that a write whose loss ends the turn had the same
short, uniform patience as routine bookkeeping, and that nothing ordered
writers or recorded who waited.  The contract asserted below is therefore:

* critical writes wait while contention stays *healthy* and stop only on
  cancellation, a permanent failure, or the anti-hang watchdog;
* non-critical writes keep exactly the bounded patience they always had,
  in-process queueing included;
* the in-process gate cannot be corrupted by an interrupt, and long or
  non-blocking maintenance never holds it;
* every wait is attributable and content-free.

Everything below uses real SQLite files and real ``BEGIN IMMEDIATE`` holders.
Synchronisation is by ``threading.Event`` and by the production contention
counters, never by sleeping long enough and hoping: the tests shrink the
budgets so the *ordering* they assert is deterministic and fast.
"""

import logging
import pathlib
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import hermes_state_writer as writer
from hermes_state import SessionDB

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clean_writer_state():
    writer.reset_contention_stats()
    writer.clear_write_cancellation()
    yield
    writer.reset_contention_stats()
    writer.clear_write_cancellation()


def _wait_until(predicate, timeout: float = 15.0, interval: float = 0.002) -> bool:
    """Poll *predicate* until true.  Generous timeout, tiny interval."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


class _ForeignWriteLockHolder:
    """Holds a real ``BEGIN IMMEDIATE`` on its own connection until released.

    Stands in for any writer this process cannot order: another Hermes
    process, an interactive ``sqlite3`` shell, a VACUUM.  The in-process gate
    deliberately has no power over these — only patience does.
    """

    def __init__(self, path, max_hold_s: float = 60.0):
        self.path = path
        self.max_hold_s = max_hold_s
        self.ready = threading.Event()
        self.release = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        conn = sqlite3.connect(str(self.path), timeout=1.0, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES ('holder', '1') "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
            )
            self.ready.set()
            self.release.wait(self.max_hold_s)
            conn.rollback()
        finally:
            conn.close()

    def __enter__(self):
        self._thread.start()
        assert self.ready.wait(15), "holder never acquired the write lock"
        return self

    def __exit__(self, *exc):
        self.release.set()
        self._thread.join(15)
        return False


def _shrink_budgets(
    monkeypatch,
    *,
    retries=3,
    busy_timeout_s=0.05,
    jitter_max=0.005,
    watchdog=None,
):
    """Compress every patience knob so ordering is provable in milliseconds.

    Ratios, not absolute values, are what the assertions rely on: the baseline
    budget stays well below the critical watchdog, exactly as in production
    (~17 s vs 60 s).
    """
    monkeypatch.setattr(SessionDB, "_WRITE_MAX_RETRIES", retries)
    monkeypatch.setattr(SessionDB, "_WRITE_BUSY_TIMEOUT_S", busy_timeout_s)
    monkeypatch.setattr(SessionDB, "_WRITE_RETRY_MIN_S", 0.001)
    monkeypatch.setattr(SessionDB, "_WRITE_RETRY_MAX_S", jitter_max)
    if watchdog is not None:
        monkeypatch.setattr(SessionDB, "_CRITICAL_WRITE_WATCHDOG_S", watchdog)


def _open(db_path, *, busy_timeout_ms=50):
    """A SessionDB with a short SQLite busy handler (fast, real contention).

    Always constructed BEFORE a holder is started: the constructor's schema
    init uses a deliberately widened busy handler, so opening under a live
    holder would block there instead of on the path under test.
    """
    db = SessionDB(db_path=db_path)
    db._conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    return db


def _rows_from_disk(db_path, session_id):
    """Read committed message content with a connection nobody else owns."""
    conn = sqlite3.connect(str(db_path))
    try:
        return [
            r[0]
            for r in conn.execute(
                "SELECT content FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            )
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 1. Critical patience: progress-based, cancellable, still fail-closed
# ---------------------------------------------------------------------------


def test_critical_append_persists_once_a_long_holder_releases(tmp_path, monkeypatch):
    """The incident, reproduced and fixed.

    A foreign connection holds the write lock past the point where a normal
    write has already given up.  The transcript append must NOT report
    failure; it must wait, commit, and be durable.  Before the fix it raised
    ``database is locked`` and the turn ended ``session_persistence_failed``.
    """
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.create_session("S1", "cli")
    seed.close()

    _shrink_budgets(monkeypatch)
    normal_db = _open(db_path)
    critical_db = _open(db_path)
    result = {}

    def _append():
        try:
            result["row_id"] = critical_db.append_message(
                session_id="S1", role="assistant", content="must survive"
            )
        except BaseException as exc:  # pragma: no cover - the bug being fixed
            result["error"] = exc

    try:
        with _ForeignWriteLockHolder(db_path) as holder:
            # Proof (no timing assumption) that the holder really does outlast
            # a normal write's whole budget.
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                normal_db.set_meta("probe", "1")

            appender = threading.Thread(target=_append, daemon=True)
            appender.start()

            # Barrier: wait until the append has itself been refused MORE times
            # than the entire normal budget allows.  This is the property under
            # test — patience beyond the old give-up point — and it is observed
            # through the production counter, not inferred from a sleep.
            assert _wait_until(
                lambda: writer.busy_retries_for("append_message")
                > SessionDB._WRITE_MAX_RETRIES
            ), "the critical append gave up inside the normal budget"

            holder.release.set()
            appender.join(30)

        assert not appender.is_alive()
        assert "error" not in result, f"append still failed: {result.get('error')!r}"
        assert isinstance(result["row_id"], int)
        # Durability, not a return value: read it back from a fresh connection.
        assert _rows_from_disk(db_path, "S1") == ["must survive"]
    finally:
        normal_db.close()
        critical_db.close()


def test_cancelling_a_critical_wait_shortens_it(tmp_path, monkeypatch, caplog):
    """Cancellation is what ends an extended wait — and it is observable.

    Two halves, both asserted, because only the pair proves the contract:

    1. While the lock is merely held, the critical append keeps waiting: it is
       still running long after a non-critical write would have failed.  (A
       fixed budget would have ended it here.)
    2. The moment cancellation is requested — production sets this from the
       SIGTERM/SIGHUP handlers and from ``close()`` — the wait ends promptly
       and the write fails CLOSED, with no phantom row.

    The heartbeat is asserted too: an extended wait that logged nothing is
    indistinguishable from a hang, which is precisely why the incident was
    unresolvable.
    """
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.create_session("S2", "cli")
    seed.close()

    _shrink_budgets(monkeypatch)
    db = _open(db_path)
    baseline = SessionDB._baseline_write_budget_s()
    outcome = {}

    def _append():
        try:
            db.append_message(
                session_id="S2", role="assistant", content="never lands"
            )
            outcome["ok"] = True
        except BaseException as exc:
            outcome["error"] = exc

    try:
        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            with _ForeignWriteLockHolder(db_path):
                appender = threading.Thread(target=_append, daemon=True)
                started = time.monotonic()
                appender.start()

                # (1) Still waiting well past the point a bounded write dies.
                assert _wait_until(
                    lambda: time.monotonic() - started > baseline * 4
                    and writer.busy_retries_for("append_message") > 2
                )
                assert appender.is_alive(), (
                    "the critical append gave up on its own; nothing but "
                    "cancellation, a permanent failure or the watchdog may "
                    "end an extended wait"
                )

                # (2) Cancellation ends it promptly.
                db.request_write_cancellation("test")
                cancelled_at = time.monotonic()
                appender.join(30)
                stopped_after = time.monotonic() - cancelled_at

        assert not appender.is_alive()
        assert "ok" not in outcome, "a locked database must not report success"
        assert isinstance(outcome.get("error"), sqlite3.OperationalError)
        assert "locked" in str(outcome["error"]).lower()
        # Prompt: at most one gate slice plus one busy attempt.
        assert stopped_after < 10.0, f"cancellation took {stopped_after:.2f}s"
    finally:
        db.close()

    # Fail closed means fail closed: no phantom row.
    assert _rows_from_disk(db_path, "S2") == []
    snapshot = writer.contention_snapshot()
    assert snapshot["failures"] >= 1
    assert snapshot.get("stopped.cancelled", 0) >= 1

    heartbeats = [
        r.getMessage() for r in caplog.records if "STILL WAITING" in r.getMessage()
    ]
    assert heartbeats, "an extended critical wait must not be silent"
    assert "op=append_message" in heartbeats[0]
    assert "class=critical" in heartbeats[0]
    assert "never lands" not in heartbeats[0]
    assert "S2" not in heartbeats[0]


def test_critical_wait_is_bounded_by_the_anti_hang_watchdog(tmp_path, monkeypatch):
    """Patience is not a hang: the watchdog terminates a wedged wait.

    The watchdog is a safety net, not a patience knob — nothing about
    durability depends on its value — but it must exist, and it must fail
    CLOSED with the real SQLite error so the turn's fail-closed path runs.
    """
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.create_session("S3", "cli")
    seed.close()

    _shrink_budgets(monkeypatch, watchdog=0.5)
    db = _open(db_path)
    try:
        with _ForeignWriteLockHolder(db_path):
            started = time.monotonic()
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                db.append_message(
                    session_id="S3", role="assistant", content="never lands"
                )
            elapsed = time.monotonic() - started
        assert elapsed < 20.0, f"watchdog did not bound the wait: {elapsed:.2f}s"
    finally:
        db.close()

    assert _rows_from_disk(db_path, "S3") == []
    assert writer.contention_snapshot().get("stopped.watchdog", 0) >= 1


def test_watchdog_bounds_thread_occupancy_as_well_as_the_wait():
    """The watchdog has an upper bound too, and it is not folklore either.

    A critical write blocks the THREAD it runs on for as long as it waits, and
    the gateway reaches ``state.db`` through ``asyncio.to_thread`` — the event
    loop's default executor, a small pool shared with every other offloaded
    call.  So this constant is simultaneously:

    * the worst-case occupancy of one pooled thread (times however many
      appends are queued behind the same lock), and
    * the worst-case delay of an interpreter exit that never received a
      signal, because CPython joins non-daemon workers on the way out.

    Hence a two-sided invariant instead of "as large as possible":

    * far enough above the baseline that surviving a healthy long holder is
      real (that is the durability goal), and
    * inside the window a default supervisor allows a service to stop
      (systemd's ``TimeoutStopSec`` defaults to 90 s), so that even where the
      interpreter-exit hook is unavailable — a runtime without
      ``threading._register_atexit`` — a shutdown finishes on its own rather
      than being escalated to SIGKILL mid-write.

    900 s satisfied neither: it hedged against a *hypothetical* long legitimate
    holder by guaranteeing a stall long enough to saturate the executor and
    outlive any supervisor's patience.
    """
    baseline = SessionDB._baseline_write_budget_s()
    watchdog = SessionDB._CRITICAL_WRITE_WATCHDOG_S
    assert watchdog > 3 * baseline, (
        f"watchdog {watchdog}s is not meaningfully beyond the ordinary budget "
        f"({baseline}s): extended critical patience would be theatre"
    )
    assert watchdog <= 90.0, (
        f"watchdog {watchdog}s pins a pooled thread — and, without the "
        "interpreter-exit hook, a signal-less shutdown — for longer than a "
        "default supervisor's stop timeout"
    )


def test_a_stuck_critical_write_does_not_pin_a_pooled_thread(tmp_path, monkeypatch):
    """Occupancy is asserted, not assumed: the worker must come back.

    Patience that never returns its thread is a resource leak wearing a
    durability costume.  Proven the only way that means anything: a second
    task submitted to a ONE-worker pool has to run while the foreign holder
    still owns the write lock, which can only happen if the stuck critical
    write released the worker first.
    """
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.create_session("S3B", "cli")
    seed.close()

    _shrink_budgets(monkeypatch, watchdog=1.0)
    db = _open(db_path)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        with _ForeignWriteLockHolder(db_path):
            stuck = pool.submit(
                db.append_message,
                session_id="S3B",
                role="assistant",
                content="never lands",
            )
            def _unrelated_offloaded_work():
                return "free"

            assert pool.submit(_unrelated_offloaded_work).result(
                timeout=30
            ) == "free", (
                "the pool's only worker never came back: a critical write may "
                "wait, but it may not own a shared thread indefinitely"
            )
            # Still fail-closed, on the same fail-closed error as before.
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                stuck.result(timeout=30)
    finally:
        pool.shutdown(wait=True)
        db.close()

    assert _rows_from_disk(db_path, "S3B") == []


def _threading_shutdown_callbacks():
    """The functions behind ``threading``'s pre-join shutdown callbacks.

    CPython wraps every registration (a ``functools.partial`` in some versions,
    a closure in others), so the registry cannot be compared by identity.
    Unwrap one level — a partial's ``func``, else the first callable closure
    cell — so the ORDER of the real callbacks becomes observable.  That order
    is the whole question: it decides whether cancellation runs before or after
    ``concurrent.futures`` joins its workers.
    """
    resolved = []
    for hook in list(threading._threading_atexits):
        target = getattr(hook, "func", None)
        if target is None:
            for cell in hook.__closure__ or ():
                try:
                    value = cell.cell_contents
                except ValueError:  # pragma: no cover - empty cell
                    continue
                if callable(value):
                    target = value
                    break
        resolved.append(target if target is not None else hook)
    return resolved


def test_interpreter_exit_cancellation_beats_the_pool_thread_join():
    """Ordering, not intent: the exit hook must run BEFORE the worker join.

    A plain ``atexit.register`` looks equivalent here and is useless: CPython
    runs ``threading``'s shutdown callbacks and joins every non-daemon thread
    *before* ``atexit`` fires, and ``concurrent.futures.thread`` joins its
    workers from one of those ``threading`` callbacks.  Cancelling from
    ``atexit`` would therefore run after the join it exists to shorten.

    ``threading`` invokes its callbacks in REVERSE registration order, so ours
    must be registered *after* ``concurrent.futures.thread``'s to run before
    it.  That is why the module imports ``concurrent.futures.thread`` itself
    instead of hoping something else imported it first.
    """
    if not hasattr(threading, "_register_atexit"):  # pragma: no cover
        pytest.skip("interpreter has no threading-level atexit registry")

    import concurrent.futures.thread as cf_thread

    assert writer.exit_cancellation_mechanism() == "threading"
    callbacks = _threading_shutdown_callbacks()
    ours = writer._cancel_writes_at_interpreter_exit
    theirs = cf_thread._python_exit
    if theirs not in callbacks:  # pragma: no cover - stdlib changed its wiring
        pytest.skip("concurrent.futures no longer joins workers from threading")
    assert ours in callbacks, (
        "nothing cancels critical-write patience at interpreter exit; an "
        "atexit.register hook does not count — it runs after the join"
    )
    assert callbacks.index(ours) > callbacks.index(theirs), (
        "the cancellation hook is registered before concurrent.futures', so "
        "reverse-order shutdown runs it AFTER the pool-worker join it is "
        "supposed to unblock"
    )


def test_clean_interpreter_exit_does_not_wait_out_the_watchdog(tmp_path):
    """A shutdown with no signal at all must still be bounded.

    The signal handlers (``cli.py``, ``gateway/run.py``) already collapse
    critical patience, but plenty of exits never see a signal: a CLI command
    returning, ``sys.exit()``, a worker script finishing while a transcript
    append is still queued behind a lock.  In that shape the last thing the
    interpreter does is join the ``ThreadPoolExecutor`` workers that
    ``asyncio.to_thread`` uses — so an uncancelled critical write turns "the
    process is done" into "the process is done in fifteen minutes".

    Real subprocess, real holder, real clean exit.  The child sets a watchdog
    (600 s) far beyond anything this test will wait for, so it can only leave
    promptly by having its patience cancelled at shutdown, in time to matter.
    """
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.create_session("SX", "cli")
    seed.close()

    marker = tmp_path / "in_flight.txt"
    child = tmp_path / "clean_exit_writer.py"
    child.write_text(
        textwrap.dedent(
            f"""
            import pathlib
            import sqlite3
            import sys
            import threading
            import time

            # Inherit this interpreter's import paths so the child sees the
            # same repo (and venv) the test process does.
            sys.path[:0] = {[p for p in sys.path if p]!r}

            # Imported BEFORE concurrent.futures on purpose: that is
            # production's order (the event loop's default executor is created
            # lazily, long after the state modules load), and it is the order
            # in which registering the exit hook first would silently put it
            # behind the pool-worker join.
            import hermes_state_writer as writer
            from hermes_state import SessionDB

            from concurrent.futures import ThreadPoolExecutor

            DB = {str(db_path)!r}

            # Far longer than the parent is willing to wait: waiting the
            # watchdog out is exactly the failure being excluded.
            SessionDB._CRITICAL_WRITE_WATCHDOG_S = 600.0
            # Cancellation collapses patience to the baseline, so keep the
            # baseline small — the assertion is about ordering, not tuning.
            SessionDB._WRITE_MAX_RETRIES = 3
            SessionDB._WRITE_BUSY_TIMEOUT_S = 0.05
            SessionDB._WRITE_RETRY_MIN_S = 0.001
            SessionDB._WRITE_RETRY_MAX_S = 0.005
            SessionDB._WRITE_GATE_WAIT_S = 0.05

            db = SessionDB(db_path=pathlib.Path(DB))
            db._conn.execute("PRAGMA busy_timeout = 50")

            holding = threading.Event()

            def _hold_forever():
                conn = sqlite3.connect(DB, timeout=1.0, isolation_level=None)
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO state_meta (key, value) "
                    "VALUES ('holder', '1') "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
                )
                holding.set()
                time.sleep(600)

            threading.Thread(target=_hold_forever, daemon=True).start()
            if not holding.wait(30):
                raise SystemExit("holder never took the write lock")

            # A non-daemon pool worker: what asyncio.to_thread runs on, and
            # what the interpreter joins on the way out.
            pool = ThreadPoolExecutor(max_workers=1)

            def _append():
                try:
                    db.append_message(
                        session_id="SX", role="assistant", content="never lands"
                    )
                except BaseException:
                    pass

            pool.submit(_append)
            deadline = time.monotonic() + 30
            while writer.busy_retries_for("append_message") < 1:
                if time.monotonic() > deadline:
                    raise SystemExit("append never reached SQLite contention")
                time.sleep(0.005)

            pathlib.Path({str(marker)!r}).write_text(repr(time.time()))
            # Fall off the end: no signal, no pool.shutdown(), no atexit hook
            # of our own.  Only interpreter shutdown is left.
            """
        )
    )

    proc = subprocess.Popen(
        [sys.executable, str(child)],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:  # pragma: no cover - the bug being fixed
        proc.kill()
        out, err = proc.communicate()
        pytest.fail(
            "a signal-less interpreter exit hung with a critical write in "
            "flight: patience must be cancelled before the pool-worker join, "
            f"not waited out.\nstdout: {out}\nstderr: {err}"
        )
    exited_at = time.time()

    assert marker.exists(), (
        f"child never got a critical write in flight.\nstdout: {out}\n"
        f"stderr: {err}"
    )
    assert proc.returncode == 0, f"child failed ({proc.returncode}): {err}"
    shutdown_s = exited_at - float(marker.read_text())
    assert shutdown_s < 30.0, (
        f"clean exit took {shutdown_s:.1f}s with a 600s watchdog: the "
        "cancellation hook did not run before the pool-worker join"
    )
    # Fail-closed survives shutdown: no phantom transcript row.
    assert _rows_from_disk(db_path, "SX") == []


def test_normal_write_keeps_its_own_bounded_budget(tmp_path, monkeypatch):
    """Non-critical writes are unchanged: still bounded, still fail closed.

    The fix must not turn every write into a long wait — only the ones whose
    loss ends a turn.
    """
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()

    _shrink_budgets(monkeypatch)
    db = _open(db_path)
    try:
        with _ForeignWriteLockHolder(db_path):
            started = time.monotonic()
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                db.set_meta("k", "v")
            elapsed = time.monotonic() - started
        # Loose upper bound — asserts "bounded", not a specific duration.
        assert elapsed < 10.0, f"normal write was not bounded: {elapsed:.2f}s"
    finally:
        db.close()


def test_normal_write_worst_case_bound_under_sustained_gate_contention(
    tmp_path, monkeypatch
):
    """In-process queueing must not add a SECOND budget to a normal write.

    The regression this pins down: if gate queueing had its own cap on top of
    the SQLite retry budget, a WRITE_NORMAL worst case would grow from the
    ~16 s it has always been to the sum of both (~77 s with production
    numbers), and the gateway offloads these onto ``asyncio.to_thread``, so
    several of them saturate the default executor.

    Constructed so the two are trivially distinguishable: a single gate slice
    is set to 5 s, far LONGER than the whole (shrunk) write budget.  If
    queueing were not clamped by the write's own deadline, the first
    admission wait alone would blow the assertion.
    """
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()

    _shrink_budgets(monkeypatch)
    monkeypatch.setattr(SessionDB, "_WRITE_GATE_WAIT_S", 5.0)
    baseline = SessionDB._baseline_write_budget_s()
    assert baseline < 1.0, "test budgets must be far below one gate slice"

    blocker_db = SessionDB(db_path=db_path)
    waiter_db = SessionDB(db_path=db_path)
    try:
        with _BlockingWriter(blocker_db) as blocker:
            started = time.monotonic()
            with pytest.raises(
                sqlite3.OperationalError, match="another writer in this process"
            ):
                waiter_db.set_meta("k", "v")
            elapsed = time.monotonic() - started
            blocker.let_go.set()
        assert blocker.error is None
        assert elapsed < 3.0, (
            f"a normal write waited {elapsed:.2f}s for in-process admission; "
            "gate queueing must consume the write's own budget, not a new one"
        )
    finally:
        blocker_db.close()
        waiter_db.close()


def test_baseline_budget_is_derived_from_the_retry_contract():
    """The only wall clock in the non-critical path is the old contract.

    Stated as an invariant so nobody re-introduces an incident-shaped
    constant: the budget is ``retries x (busy_timeout + max jitter)`` — the
    worst case the retry loop always described — and there is no separate
    "critical budget" number to drift away from it.
    """
    assert SessionDB._baseline_write_budget_s() == pytest.approx(
        SessionDB._WRITE_MAX_RETRIES
        * (SessionDB._WRITE_BUSY_TIMEOUT_S + SessionDB._WRITE_RETRY_MAX_S)
    )
    for gone in (
        "_CRITICAL_WRITE_BUDGET_S",
        "_CRITICAL_WRITE_PATIENCE_FACTOR",
        "_WRITE_GATE_TOTAL_WAIT_S",
    ):
        assert not hasattr(SessionDB, gone), (
            f"{gone} is back: critical patience must be progress-based, not a "
            "wall-clock number chosen to cover one observed holder"
        )
    # The watchdog is not that number either: it bounds a wedge (and the thread
    # the wedge occupies), so it lives strictly above the baseline instead of
    # being derived from it.  Its two-sided invariant is asserted in
    # test_watchdog_bounds_thread_occupancy_as_well_as_the_wait.
    assert (
        SessionDB._CRITICAL_WRITE_WATCHDOG_S
        > SessionDB._baseline_write_budget_s()
    )


# ---------------------------------------------------------------------------
# 2. In-process fairness: the half of the incident we CAN order
# ---------------------------------------------------------------------------


class _BlockingWriter:
    """Holds a real write transaction open from inside ``_execute_write``.

    Uses a genuine ``SessionDB`` write callback, so the writer holds the
    per-path gate, the per-connection lock and SQLite's write lock at once —
    the same state a slow real write occupies.
    """

    def __init__(self, db, write_class=writer.WRITE_NORMAL):
        self.db = db
        self.write_class = write_class
        self.entered = threading.Event()
        self.let_go = threading.Event()
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        def _do(conn):
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES ('blocker', '1') "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
            )
            self.entered.set()
            self.let_go.wait(30)

        try:
            self.db._execute_write(
                _do, op="test_blocker", write_class=self.write_class
            )
        except BaseException as exc:  # pragma: no cover
            self.error = exc

    def __enter__(self):
        self._thread.start()
        assert self.entered.wait(15), "blocking writer never started its txn"
        return self

    def __exit__(self, *exc):
        self.let_go.set()
        self._thread.join(15)
        return False


def test_critical_append_is_admitted_before_queued_routine_writes(
    tmp_path, monkeypatch
):
    """A transcript append jumps the in-process queue.

    This is the 23:04 shape stripped to its mechanism: several ``SessionDB``
    objects in ONE process, each with its own connection and its own
    ``threading.Lock``, all writing to one ``state.db``.  A transcript append
    must not sit behind routine work just because the routine work asked
    first.
    """
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.create_session("S4", "cli")
    seed.close()

    # Keep queued writers queued so the assertion is about gate ordering and
    # not about who happened to re-probe first.
    monkeypatch.setattr(SessionDB, "_WRITE_GATE_WAIT_S", 30.0)

    blocker_db = SessionDB(db_path=db_path)
    routine_db = SessionDB(db_path=db_path)
    critical_db = SessionDB(db_path=db_path)
    # Same file, three independent objects: distinct locks and connections.
    assert routine_db._lock is not critical_db._lock
    assert routine_db._conn is not critical_db._conn
    # ...but one shared admission gate, which is the fix.
    assert routine_db._writer_gate is critical_db._writer_gate

    order = []
    order_lock = threading.Lock()

    def _record(name):
        with order_lock:
            order.append(name)

    def _routine():
        def _do(conn):
            # Recorded INSIDE the transaction: the assertion is about which
            # writer SQLite actually let in, not about thread start order.
            _record("routine")
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES ('routine', '1') "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
            )

        routine_db._execute_write(_do, op="test_routine")

    def _critical():
        def _do(conn):
            _record("critical")
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, "
                "active) VALUES (?, 'assistant', 'priority', ?, 1)",
                ("S4", time.time()),
            )

        critical_db._execute_write(
            _do, op="append_message", write_class=writer.WRITE_CRITICAL
        )

    try:
        with _BlockingWriter(blocker_db) as blocker:
            gate = critical_db._writer_gate
            routine = threading.Thread(target=_routine, daemon=True)
            routine.start()
            # The routine write asks FIRST and is queued first.
            assert _wait_until(lambda: gate.pending()[1] == 1), gate.pending()

            crit = threading.Thread(target=_critical, daemon=True)
            crit.start()
            assert _wait_until(lambda: gate.pending()[0] == 1), gate.pending()

            blocker.let_go.set()
            crit.join(30)
            routine.join(30)

        assert blocker.error is None
    finally:
        blocker_db.close()
        routine_db.close()
        critical_db.close()

    # Queued second, served first.
    assert order == ["critical", "routine"], order
    assert _rows_from_disk(db_path, "S4") == ["priority"]


def test_queueing_behind_our_own_writer_does_not_consume_sqlite_retries(
    tmp_path, monkeypatch
):
    """Waiting for admission is not a failed attempt.

    Before the gate, writers on one ``SessionDB`` already queued on
    ``self._lock`` with no timeout, and they succeeded when their turn came.
    Charging the SQLite retry budget for in-process queueing would have turned
    those waits into brand-new failures under heavy load — a regression
    dressed up as a fix.  The retry budget counts ``BEGIN IMMEDIATE``
    attempts; queueing consumes only wall clock.
    """
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()

    # One SQLite attempt allowed, a generous wall clock, and a gate slice far
    # shorter than the hold: the waiter must re-queue many times, spend no
    # attempt doing so, and still succeed.
    _shrink_budgets(monkeypatch, retries=1, busy_timeout_s=5.0)
    monkeypatch.setattr(SessionDB, "_WRITE_GATE_WAIT_S", 0.01)

    blocker_db = SessionDB(db_path=db_path)
    waiter_db = SessionDB(db_path=db_path)
    outcome = {}

    def _queued_write():
        def _do(conn):
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES ('queued', 'ok') "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
            )

        try:
            waiter_db._execute_write(_do, op="test_queued")
            outcome["ok"] = True
        except BaseException as exc:
            outcome["error"] = exc

    try:
        with _BlockingWriter(blocker_db) as blocker:
            gate = waiter_db._writer_gate
            t = threading.Thread(target=_queued_write, daemon=True)
            t.start()
            # Prove it really did re-queue repeatedly rather than fail fast.
            assert _wait_until(lambda: gate.stats()["timeouts"] >= 3)
            assert "error" not in outcome
            blocker.let_go.set()
            t.join(30)
        assert blocker.error is None
    finally:
        blocker_db.close()
        waiter_db.close()

    assert outcome.get("ok") is True, outcome.get("error")
    assert writer.busy_retries_for("test_queued") == 0, (
        "in-process queueing must not be counted as SQLite contention"
    )


def test_persistence_paths_declare_the_right_durability_class(tmp_path):
    """The classification contract, asserted per CALL and including nesting.

    Recorded as an ordered list of ``(op, class)`` pairs, not a dict keyed by
    op: ``update_token_counts`` and ``record_auxiliary_usage`` each perform a
    NESTED ``create_session`` FK-guard write, and a dict would let the nested
    entry overwrite (or be overwritten by) the explicit ``create_session``
    earlier in the test — which is exactly how an escalation of that guard to
    priority 0 stayed invisible.
    """
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    calls = []
    real = db._execute_write

    def _capture(fn, *, op=None, write_class=writer.WRITE_NORMAL):
        calls.append((op or writer.describe_op(fn), write_class))
        return real(fn, op=op, write_class=write_class)

    def _since(mark):
        return calls[mark:]

    try:
        db._execute_write = _capture

        mark = len(calls)
        db.create_session("S7", "cli")
        assert _since(mark) == [("create_session", writer.WRITE_CRITICAL)]

        mark = len(calls)
        db.append_message(session_id="S7", role="user", content="x")
        assert _since(mark) == [("append_message", writer.WRITE_CRITICAL)]

        # Accounting: BOTH the FK guard and the update itself must yield.
        mark = len(calls)
        db.update_token_counts("S7", input_tokens=1, output_tokens=1)
        assert _since(mark) == [
            ("create_session", writer.WRITE_BEST_EFFORT),
            ("update_token_counts", writer.WRITE_BEST_EFFORT),
        ]

        mark = len(calls)
        db.record_auxiliary_usage("S7", "vision", input_tokens=1)
        assert _since(mark) == [
            ("create_session", writer.WRITE_BEST_EFFORT),
            ("record_auxiliary_usage", writer.WRITE_BEST_EFFORT),
        ]

        mark = len(calls)
        db.ensure_session("S8", "cli")
        assert _since(mark) == [("create_session", writer.WRITE_NORMAL)]
    finally:
        db._execute_write = real
        db.close()


# ---------------------------------------------------------------------------
# 3. Maintenance must never become the starver
# ---------------------------------------------------------------------------


def test_passive_checkpoint_does_not_take_writer_admission(tmp_path, monkeypatch):
    """A PASSIVE checkpoint stays concurrent, as SQLite designed it.

    PASSIVE takes the CHECKPOINTER lock, not the WRITER lock, so writers on
    other connections proceed alongside it.  The writer gate, by contrast, is
    non-preemptible — so taking admission here would convert a concurrent
    maintenance operation into an exclusive one whose blast radius is every
    later writer including a transcript append, for what this method's own
    docstring calls "minutes of I/O".  That would be a NEW cause of
    ``session_persistence_failed``, introduced by the fix.

    Asserted structurally, not by timing luck: one gate slice is 5 s, so an
    admission attempt would be unmissable, and the gate counters must show the
    checkpoint never even queued.
    """
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    monkeypatch.setattr(SessionDB, "_WRITE_GATE_WAIT_S", 5.0)

    blocker_db = SessionDB(db_path=db_path)
    maint_db = _open(db_path)
    gate = maint_db._writer_gate
    try:
        with _BlockingWriter(blocker_db) as blocker:
            before = gate.stats()
            started = time.monotonic()
            maint_db._try_wal_checkpoint()  # must not raise, must not queue
            elapsed = time.monotonic() - started
            after = gate.stats()
            blocker.let_go.set()
        assert blocker.error is None
        assert elapsed < 2.0, f"checkpoint waited {elapsed:.2f}s for admission"
        assert after["timeouts"] == before["timeouts"]
        assert after["admissions"] == before["admissions"]
    finally:
        blocker_db.close()
        maint_db.close()


def test_late_critical_append_is_not_blocked_by_a_best_effort_merge_pass(
    tmp_path, monkeypatch
):
    """A slow maintenance pass must yield to a LATER-arriving critical write.

    The gate is non-preemptible, so "the merge takes admission" only helps
    writers that were already queued at the instant it acquired.  A bounded FTS
    merge issues up to ``_FTS_MERGE_COMMANDS_PER_PASS`` commands per index;
    holding ONE admission across all of them blocks every append that arrives
    mid-pass for the rest of the pass — and the write lock is already released
    between commands, so those appends could interleave before any gate
    existed.  The contract asserted here is per-command admission plus an
    explicit higher-priority check between commands.

    Fully event-driven, no timing luck.  Two critical writers are queued while
    command #1 holds admission; the first of them (deliberately slow) claims
    the gate when #1 releases, which leaves the second one QUEUED at the
    moment the pass makes its between-commands decision.  It must yield there.
    """
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.create_session("S5", "cli")
    seed.append_message(session_id="S5", role="user", content="indexed content")
    seed.close()

    # Long enough that a refused admission would be a test failure, not a
    # silent skip: everything here is supposed to be admitted eventually.
    monkeypatch.setattr(SessionDB, "_WRITE_GATE_WAIT_S", 10.0)

    maint_db = SessionDB(db_path=db_path)
    slow_db = SessionDB(db_path=db_path)
    late_db = SessionDB(db_path=db_path)
    gate = maint_db._writer_gate
    commands_run = []
    first_command_holding = threading.Event()
    release_first_command = threading.Event()

    def _stub_command(tbl, max_pages, priority, **kwargs):
        """One merge command: real admission, no real FTS work."""
        with maint_db._writer_admission(
            priority=priority, timeout=maint_db._WRITE_GATE_WAIT_S
        ) as admission:
            if not admission.admitted:
                return None
            commands_run.append(tbl)
            if len(commands_run) == 1:
                first_command_holding.set()
                release_first_command.wait(15)
            return True  # pretend real merge work happened

    monkeypatch.setattr(maint_db, "_merge_fts_one_command", _stub_command)

    pass_result = {}
    append_result = {}

    def _merge():
        pass_result["executed"] = maint_db._merge_fts_incrementally(
            max_pages=500, max_commands=4
        )

    # The first critical writer holds its transaction open on request, so the
    # second one is still queued when the merge pass looks.
    slow_critical = _BlockingWriter(slow_db, write_class=writer.WRITE_CRITICAL)

    def _late_append():
        try:
            late_db.append_message(
                session_id="S5", role="assistant", content="late but critical"
            )
            append_result["ok"] = True
        except BaseException as exc:  # pragma: no cover
            append_result["error"] = exc

    late = threading.Thread(target=_late_append, daemon=True)
    merger = threading.Thread(target=_merge, daemon=True)
    try:
        merger.start()
        assert first_command_holding.wait(15), "merge pass never started"

        # Both criticals arrive AFTER the pass began: exactly the case a
        # non-preemptible per-pass admission would punish.
        slow_critical._thread.start()
        assert _wait_until(lambda: gate.pending()[0] == 1), gate.pending()
        late.start()
        assert _wait_until(lambda: gate.pending()[0] == 2), gate.pending()

        release_first_command.set()
        # The pass must return without running its remaining commands.
        merger.join(30)
        assert not merger.is_alive()
        assert pass_result.get("executed") == 1, (
            f"the merge pass ran {pass_result.get('executed')} commands while a "
            "critical writer was queued; it must yield between commands"
        )

        slow_critical.let_go.set()
        slow_critical._thread.join(15)
        late.join(30)
    finally:
        # Unblock every helper even if an assertion above failed.
        release_first_command.set()
        slow_critical.let_go.set()
        maint_db.close()
        slow_db.close()
        late_db.close()

    assert slow_critical.error is None
    assert append_result.get("ok") is True, append_result.get("error")
    assert _rows_from_disk(db_path, "S5") == ["indexed content", "late but critical"]


def test_best_effort_maintenance_yields_instead_of_racing_a_writer(
    tmp_path, monkeypatch
):
    """Routine maintenance must never compete with a live writer.

    The bounded FTS merge is best effort by contract — a skipped pass costs
    nothing because the cadence comes back — so when another writer of ours
    holds admission it returns a truthful 0 rather than fighting for the write
    lock, and it still does real work once uncontended.
    """
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.create_session("S6", "cli")
    seed.append_message(session_id="S6", role="user", content="indexed content")
    seed.close()

    # Maintenance should give up admission quickly rather than wait.
    monkeypatch.setattr(SessionDB, "_WRITE_GATE_WAIT_S", 0.05)

    blocker_db = SessionDB(db_path=db_path)
    maint_db = SessionDB(db_path=db_path)
    try:
        with _BlockingWriter(blocker_db) as blocker:
            # Skipped (0 commands), not raised, not blocked for the duration.
            assert maint_db._merge_fts_incrementally(max_pages=500) == 0
            blocker.let_go.set()
        assert blocker.error is None

        # Uncontended, the same call does real work again.
        assert maint_db._merge_fts_incrementally(max_pages=500) >= 1
    finally:
        blocker_db.close()
        maint_db.close()


# ---------------------------------------------------------------------------
# 3b. ...including maintenance on the writer's OWN connection
# ---------------------------------------------------------------------------
#
# The gate is process-wide (one per database FILE); ``self._lock`` is not (one
# per ``SessionDB`` OBJECT).  Several production paths take ``self._lock`` while
# holding NO admission at all — ``_try_wal_checkpoint``, ``close``, ``vacuum``,
# ``optimize_fts``, ``rebuild_fts``, the FTS trash-teardown probe, every
# ``get_*``/``set_*`` helper — and the long ones are deliberately outside the
# gate (see ``vacuum``: a non-preemptible admission held for a minutes-long
# rewrite would put every later writer behind it instead of leaving them to
# SQLite).
#
# That has a consequence for the writers that DO take admission: waiting for
# ``self._lock`` without a bound while holding the gate hands the whole file's
# admission to one connection's maintenance run.  Every OTHER ``SessionDB`` on
# that file is refused, a transcript append among them, and the waiter cannot
# re-check cancellation, the watchdog or its own deadline while it is parked
# there — the exact ``session_persistence_failed`` shape this module exists to
# remove, reintroduced from inside.  Lock ORDER is not the issue and must not
# change (gate -> self._lock, never the reverse, or the bounded merge command
# below deadlocks against a writer); the missing bound is.


class _MaintenanceLockHolder:
    """Hold one ``SessionDB``'s connection lock through a REAL maintenance call.

    ``rebuild_fts()`` is one of the paths that take ``self._lock`` with no
    writer admission whatsoever.  Only its per-table probe is stubbed, and only
    so the hold lasts exactly as long as the test needs: the lock, the
    connection and the production call stack are the real ones.
    """

    def __init__(self, db, monkeypatch):
        self.db = db
        self.holding = threading.Event()
        self.release = threading.Event()
        self.error = None
        monkeypatch.setattr(db, "_fts_table_exists", self._probe)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _probe(self, name):
        self.holding.set()
        self.release.wait(60)
        return False  # holding self._lock is the point; skip the real rebuild

    def _run(self):
        try:
            self.db.rebuild_fts()
        except BaseException as exc:  # pragma: no cover - helper must not mask
            self.error = exc

    def __enter__(self):
        self._thread.start()
        assert self.holding.wait(15), "maintenance never entered rebuild_fts()"
        assert self.db._lock.locked(), (
            "the maintenance stand-in is not holding the connection lock"
        )
        return self

    def __exit__(self, *exc):
        self.release.set()
        self._thread.join(15)
        return False


def _conn_lock_timeouts(op: str) -> int:
    """Times *op* handed admission back instead of parking on self._lock.

    Read from the LIVE production counter, so every barrier below synchronises
    on real contention as it happens rather than on a sleep — the same reason
    ``note_busy_retry`` exists.
    """
    return writer.conn_lock_timeouts_for(op)


def test_maintenance_on_our_connection_cannot_hold_the_gate_hostage(
    tmp_path, monkeypatch
):
    """One connection's maintenance must not deny the file's other writers.

    Shape: maintenance holds ``maint_db``'s connection lock (no admission, as
    production does), an ordinary write on THAT SAME object takes admission and
    finds the lock busy, and a transcript append arrives on a DIFFERENT
    ``SessionDB`` — a separate lock and connection, one shared gate.

    Asserted, in this order:

    1. the writer behind maintenance hands admission back repeatedly (the wait
       is sliced, so the gate is never held for the maintenance run's duration);
    2. a ``critical``-priority arrival still wins the gate over that bouncing
       ``normal`` writer, so the fix does not cost priority;
    3. the append COMMITS while maintenance still owns the other connection;
    4. nothing is lost or duplicated: the write that had to wait lands too, once
       maintenance lets go.
    """
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.create_session("SL1", "cli")
    seed.close()

    maint_db = SessionDB(db_path=db_path)
    critical_db = SessionDB(db_path=db_path)
    gate = maint_db._writer_gate
    assert critical_db._lock is not maint_db._lock
    assert critical_db._writer_gate is maint_db._writer_gate

    # Per-instance slices shadow the class attribute: the writer stuck behind
    # maintenance re-checks every 50 ms, while the transcript append is willing
    # to queue for far longer than the whole test — so step 3 is a statement
    # about admission, never about who happened to re-probe first.
    maint_db._WRITE_GATE_WAIT_S = 0.05
    critical_db._WRITE_GATE_WAIT_S = 30.0

    outcome = {}
    critical_done = threading.Event()

    def _behind_maintenance():
        def _do(conn):
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, "
                "active) VALUES (?, 'assistant', 'behind maintenance', ?, 1)",
                ("SL1", time.time()),
            )

        try:
            maint_db._execute_write(_do, op="test_behind_maintenance")
            outcome["behind"] = "ok"
        except BaseException as exc:
            outcome["behind"] = exc

    def _critical():
        try:
            critical_db.append_message(
                session_id="SL1", role="assistant", content="late but critical"
            )
            outcome["critical"] = "ok"
        except BaseException as exc:
            outcome["critical"] = exc
        finally:
            critical_done.set()

    stuck = threading.Thread(target=_behind_maintenance, daemon=True)
    crit = threading.Thread(target=_critical, daemon=True)
    try:
        with _MaintenanceLockHolder(maint_db, monkeypatch) as holder:
            stuck.start()
            # (1) Admission taken and given back, repeatedly.
            assert _wait_until(
                lambda: _conn_lock_timeouts("test_behind_maintenance") >= 2
            ), (
                "the writer behind maintenance kept admission while waiting "
                f"for self._lock; gate={gate.stats()}"
            )
            assert stuck.is_alive(), (
                "the writer behind maintenance gave up instead of re-queueing"
            )

            # (2) Priority survives the re-queueing.
            admission = gate.acquire_admission(
                priority=writer.priority_of(writer.WRITE_CRITICAL), timeout=10
            )
            assert admission.admitted, (
                f"the gate was held hostage by one connection: {gate.stats()}"
            )
            gate.release()

            # (3) End to end, on another SessionDB, while the lock is held.
            crit.start()
            assert critical_done.wait(15), (
                "a transcript append on another SessionDB was starved by a "
                "writer parked on an unrelated connection's lock"
            )
            assert outcome["critical"] == "ok", outcome["critical"]
            assert maint_db._lock.locked(), (
                "maintenance let go early — the test proved nothing"
            )
        assert holder.error is None
        # (4) The write that had to wait still lands.
        stuck.join(30)
        assert not stuck.is_alive()
        assert outcome.get("behind") == "ok", outcome.get("behind")
    finally:
        maint_db.close()
        critical_db.close()

    assert _rows_from_disk(db_path, "SL1") == [
        "late but critical",
        "behind maintenance",
    ]


def test_shutdown_cancellation_reaches_a_writer_waiting_on_our_connection_lock(
    tmp_path, monkeypatch, caplog
):
    """Waiting for ``self._lock`` must stay cancellable and observable.

    A critical write parked on a plain ``threading.Lock`` cannot poll anything:
    no cancellation, no watchdog, no heartbeat.  So the shutdown contract closed
    for SQLite waits has to hold here too — and this is the process-wide switch
    the SIGTERM/SIGHUP handlers and the interpreter-exit hook set, not the
    per-instance one.

    Fail-closed is asserted with it: the row is not reported as persisted, and
    it does not appear on disk either.
    """
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.create_session("SL2", "cli")
    seed.close()

    _shrink_budgets(monkeypatch)
    monkeypatch.setattr(SessionDB, "_WRITE_GATE_WAIT_S", 0.05)
    db = SessionDB(db_path=db_path)
    outcome = {}
    done = threading.Event()
    stopped_after = None

    def _append():
        try:
            db.append_message(
                session_id="SL2", role="assistant", content="never lands"
            )
            outcome["ok"] = True
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            done.set()

    appender = threading.Thread(target=_append, daemon=True)
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            with _MaintenanceLockHolder(db, monkeypatch):
                appender.start()
                assert _wait_until(
                    lambda: _conn_lock_timeouts("append_message") >= 3
                ), (
                    "the append never re-checked anything while maintenance "
                    "held the connection"
                )
                # Patience is unchanged: our own maintenance is not a failure,
                # so nothing but cancellation or the watchdog may end this.
                assert appender.is_alive()
                # The heartbeat proves the loop top is reached while waiting —
                # the same place cancellation and the watchdog are checked.
                assert _wait_until(
                    lambda: any(
                        "STILL WAITING" in r.getMessage() for r in caplog.records
                    )
                ), "an extended wait on our own connection lock was silent"

                writer.request_write_cancellation("signal-15")
                cancelled_at = time.monotonic()
                assert done.wait(15), (
                    "shutdown cancellation never reached a writer waiting for "
                    "self._lock"
                )
                stopped_after = time.monotonic() - cancelled_at
    finally:
        appender.join(15)
        db.close()

    assert "ok" not in outcome, "a write that never ran must not report success"
    error = outcome.get("error")
    assert isinstance(error, sqlite3.OperationalError), error
    assert "locked" in str(error).lower()
    assert "cancelled (process)" in str(error), error
    assert stopped_after is not None and stopped_after < 10.0, stopped_after
    # Fail closed: no phantom transcript row.
    assert _rows_from_disk(db_path, "SL2") == []
    assert writer.contention_snapshot().get("stopped.cancelled", 0) >= 1


def test_watchdog_reaches_a_writer_waiting_on_our_connection_lock(
    tmp_path, monkeypatch
):
    """The anti-hang net must cover this wait too, and attribute it honestly.

    Nothing releases the connection here, so only the watchdog can end the
    write.  Its message must not blame "another writer in this process" when the
    blocker was this connection's own maintenance: misattribution is what made
    both 2026-08-03 incidents unresolvable.
    """
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.create_session("SL3", "cli")
    seed.close()

    _shrink_budgets(monkeypatch, watchdog=1.0)
    monkeypatch.setattr(SessionDB, "_WRITE_GATE_WAIT_S", 0.05)
    db = SessionDB(db_path=db_path)
    outcome = {}
    done = threading.Event()

    def _append():
        try:
            db.append_message(
                session_id="SL3", role="assistant", content="never lands"
            )
            outcome["ok"] = True
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            done.set()

    appender = threading.Thread(target=_append, daemon=True)
    try:
        with _MaintenanceLockHolder(db, monkeypatch):
            started = time.monotonic()
            appender.start()
            assert done.wait(15), (
                "the watchdog never fired: a writer waiting for self._lock was "
                "unreachable, so the wait was not bounded at all"
            )
            elapsed = time.monotonic() - started
    finally:
        appender.join(15)
        db.close()

    assert "ok" not in outcome
    error = outcome.get("error")
    assert isinstance(error, sqlite3.OperationalError), error
    message = str(error)
    assert "watchdog" in message, message
    assert "locked" in message.lower(), message
    assert "maintenance" in message, (
        f"the failure blames the wrong holder: {message}"
    )
    assert elapsed < 20.0, f"the watchdog did not bound the wait: {elapsed:.2f}s"
    assert _rows_from_disk(db_path, "SL3") == []
    assert writer.contention_snapshot().get("stopped.watchdog", 0) >= 1


def test_waiting_for_our_connection_lock_consumes_no_sqlite_attempt(
    tmp_path, monkeypatch
):
    """Bouncing off ``self._lock`` is not a failed write attempt.

    The retry budget counts ``BEGIN IMMEDIATE`` attempts.  A writer that was
    refused the connection never reached SQLite, so charging it would turn a
    wait that always succeeded into a brand-new failure under maintenance —
    the same regression ``test_queueing_behind_our_own_writer_does_not_consume_
    sqlite_retries`` pins down for admission.

    Proven with a budget of exactly ONE attempt: the write bounces repeatedly
    and must still succeed the moment maintenance lets go.
    """
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()

    # One SQLite attempt, a generous wall clock, and a 10 ms slice: many
    # bounces, none of them allowed to cost the single attempt.
    _shrink_budgets(monkeypatch, retries=1, busy_timeout_s=5.0)
    monkeypatch.setattr(SessionDB, "_WRITE_GATE_WAIT_S", 0.01)
    db = SessionDB(db_path=db_path)
    outcome = {}
    done = threading.Event()

    def _write():
        def _do(conn):
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES "
                "('one_attempt', 'ok') "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
            )

        try:
            db._execute_write(_do, op="test_one_attempt")
            outcome["ok"] = True
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            done.set()

    t = threading.Thread(target=_write, daemon=True)
    try:
        with _MaintenanceLockHolder(db, monkeypatch) as holder:
            t.start()
            assert _wait_until(
                lambda: _conn_lock_timeouts("test_one_attempt") >= 3
            ), (
                "the writer never handed admission back while maintenance held "
                "the connection"
            )
            assert "error" not in outcome, outcome.get("error")
            holder.release.set()
        assert done.wait(15)
    finally:
        t.join(15)
        db.close()

    assert outcome.get("ok") is True, outcome.get("error")
    assert writer.busy_retries_for("test_one_attempt") == 0, (
        "waiting for our own connection lock must not be counted as SQLite "
        "contention"
    )
    # Attributable, and per operation: a snapshot must say which write was made
    # to wait by our own maintenance, not merely that something was.
    snapshot = writer.contention_snapshot()
    assert snapshot["conn_lock_timeouts"] >= 3
    assert snapshot["conn_lock_timeouts_by_op"]["test_one_attempt"] >= 3
    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute(
            "SELECT value FROM state_meta WHERE key = 'one_attempt'"
        ).fetchone() == ("ok",)
    finally:
        conn.close()


def test_a_spent_slice_polls_the_connection_lock_instead_of_raising(tmp_path):
    """The slice must survive a budget that is already gone.

    ``threading.Lock.acquire`` rejects a negative timeout, and the slice is
    clamped by the write's remaining deadline — which is exactly zero on the
    last iteration a non-critical write is allowed.  Clamping to a
    non-blocking poll keeps that last honest attempt possible; raising
    ``ValueError`` there would turn contention into a crash.  And neither
    branch may leave the lock held.

    ``queued`` is asserted with it, because the contention *classification*
    depends on it: taking a free lock costs microseconds, so counting that as a
    wait would report every ordinary write as contended — the same trap
    ``Admission.queued`` exists to avoid.
    """
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        with db._write_connection_lock(-1.0) as conn_lock:
            assert conn_lock.locked is True
            assert conn_lock.queued is False, "a free lock is not a wait"
            assert db._lock.locked()
        assert not db._lock.locked()

        db._lock.acquire()
        try:
            with db._write_connection_lock(-1.0) as conn_lock:
                assert conn_lock.locked is False
                assert conn_lock.queued is True
        finally:
            db._lock.release()
        assert not db._lock.locked()

        # An interrupt inside the write must not strand the connection: a lock
        # left held here would wedge every later write on this object AND its
        # close(), for the life of the process.
        with pytest.raises(KeyboardInterrupt):
            with db._write_connection_lock(1.0) as conn_lock:
                assert conn_lock.locked is True
                raise KeyboardInterrupt
        assert not db._lock.locked()
    finally:
        db.close()


def test_a_merge_command_gives_the_gate_back_instead_of_waiting_out_maintenance(
    tmp_path, monkeypatch
):
    """The bounded merge holds admission too, so it has the same duty.

    ``_merge_fts_one_command`` takes admission and then this connection's lock.
    Parking there would hand the whole file's gate to a best-effort maintenance
    command for the duration of an unrelated ``vacuum()`` on the same object,
    with a transcript append queued behind it.  Skipping is free — the merge
    cadence comes back on the next boundary — so it must skip.
    """
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.create_session("SL5", "cli")
    seed.append_message(session_id="SL5", role="user", content="indexed content")
    seed.close()

    monkeypatch.setattr(SessionDB, "_WRITE_GATE_WAIT_S", 0.05)
    maint_db = SessionDB(db_path=db_path)
    critical_db = SessionDB(db_path=db_path)
    gate = maint_db._writer_gate
    merged = {}

    def _merge():
        merged["value"] = maint_db._merge_fts_one_command(
            "messages_fts", 500, writer.priority_of(writer.WRITE_BEST_EFFORT)
        )

    merger = threading.Thread(target=_merge, daemon=True)
    try:
        with _MaintenanceLockHolder(maint_db, monkeypatch):
            merger.start()
            merger.join(10)
            assert not merger.is_alive(), (
                "the merge command waited out maintenance while holding the "
                "process-wide gate"
            )
            assert merged["value"] is None, (
                "nothing ran, so the command must report that nothing ran"
            )
            assert not gate.held(), f"admission was not released: {gate.stats()}"
            # ...and a transcript append on another SessionDB is unaffected.
            critical_db.append_message(
                session_id="SL5", role="assistant", content="unaffected"
            )
    finally:
        merger.join(15)
        maint_db.close()
        critical_db.close()

    assert _rows_from_disk(db_path, "SL5") == ["indexed content", "unaffected"]


# ---------------------------------------------------------------------------
# 4. Telemetry: attributable, bounded, content-free
# ---------------------------------------------------------------------------


def test_routine_writes_emit_no_contention_logs(tmp_path, caplog):
    """Ordinary traffic must stay silent.

    Sub-second lock blips are normal on a shared database.  If they logged,
    the signal that matters would be buried and operators would filter the
    whole logger out.
    """
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        with caplog.at_level(logging.DEBUG, logger="hermes_state"):
            db.create_session("S8", "cli")
            for i in range(10):
                db.append_message(session_id="S8", role="user", content=f"m{i}")
    finally:
        db.close()

    assert "was starved but succeeded" not in caplog.text
    assert "GAVE UP" not in caplog.text
    assert "STILL WAITING" not in caplog.text
    snapshot = writer.contention_snapshot()
    assert snapshot["quiet"] >= 11
    assert snapshot.get("failures", 0) == 0


def test_contention_report_names_the_operation_without_leaking_content(
    tmp_path, monkeypatch, caplog
):
    """A starved-but-successful write is reported, once, with no payload.

    Attribution is the whole point: both incidents were unresolvable because
    no log said which operation waited, for how long, or on what.  Equally
    important is what must NEVER appear — transcript text, session ids, user
    or chat identifiers, credentials.
    """
    db_path = tmp_path / "state.db"
    seed = SessionDB(db_path=db_path)
    seed.create_session("S9", "cli")
    seed.close()

    _shrink_budgets(monkeypatch)
    db = _open(db_path)
    # Report every measurable wait so the test needs no multi-second stall.
    # Patched AFTER the open so schema-init writes don't also get reported.
    monkeypatch.setattr(writer, "CONTENTION_LOG_THRESHOLD_S", 0.0)
    secret = "swordfish-transcript-body"
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            with _ForeignWriteLockHolder(db_path) as holder:
                done = threading.Event()

                def _append():
                    db.append_message(
                        session_id="S9", role="assistant", content=secret
                    )
                    done.set()

                t = threading.Thread(target=_append, daemon=True)
                t.start()
                assert _wait_until(
                    lambda: writer.busy_retries_for("append_message") > 0
                )
                holder.release.set()
                assert done.wait(30)
                t.join(15)
    finally:
        db.close()

    reports = [
        r.getMessage()
        for r in caplog.records
        if "state.db write was starved" in r.getMessage()
        and "op=append_message" in r.getMessage()
    ]
    assert len(reports) == 1, reports
    report = reports[0]
    assert "class=critical" in report
    # The holder is a separate connection, so the wait is SQLite's, not ours.
    # Misattributing it to in-process queueing would send the next
    # investigation down exactly the wrong path.
    assert "contention=sqlite" in report
    assert "waited=" in report
    assert "busy_retries=" in report
    # Content-free: nothing from the row, and no identifiers.
    assert secret not in report
    assert "S9" not in report
    assert str(db_path) not in report


def test_repeated_contention_is_throttled_but_still_counted(monkeypatch):
    """Bounded noise: identical events collapse, the count survives.

    A storm must not spam, and must not silently vanish either — the
    suppressed count is folded into the next line that does get emitted.
    """
    monkeypatch.setattr(writer, "CONTENTION_LOG_THRESHOLD_S", 0.0)
    monkeypatch.setattr(writer, "CONTENTION_LOG_COOLDOWN_S", 3600.0)

    logged = []
    monkeypatch.setattr(
        writer.logger, "warning", lambda msg, *a: logged.append(msg % a)
    )

    for _ in range(25):
        writer.record_outcome(
            op="append_message",
            write_class=writer.WRITE_CRITICAL,
            succeeded=True,
            waited_s=3.0,
            gate_waited_s=0.0,
            attempts=4,
            sqlite_busy_retries=3,
        )

    assert len(logged) == 1, logged
    snapshot = writer.contention_snapshot()
    assert snapshot["suppressed"]["slow/append_message/sqlite"] == 24

    # The next event past the cooldown carries the suppressed count forward.
    monkeypatch.setattr(writer, "CONTENTION_LOG_COOLDOWN_S", 0.0)
    writer.record_outcome(
        op="append_message",
        write_class=writer.WRITE_CRITICAL,
        succeeded=True,
        waited_s=3.0,
        gate_waited_s=0.0,
        attempts=4,
        sqlite_busy_retries=3,
    )
    assert len(logged) == 2, logged
    assert "+24 similar suppressed" in logged[1]


def test_long_wait_heartbeat_is_throttled_and_content_free(monkeypatch):
    """The heartbeat is bounded too: one line per operation per window."""
    logged = []
    monkeypatch.setattr(
        writer.logger, "warning", lambda msg, *a: logged.append(msg % a)
    )
    monkeypatch.setattr(writer, "LONG_WAIT_HEARTBEAT_S", 3600.0)

    for _ in range(5):
        writer.note_long_wait(
            op="append_message",
            write_class=writer.WRITE_CRITICAL,
            waited_s=42.0,
            attempts=40,
            sqlite_busy_retries=39,
            gate_waited_s=0.0,
        )
    assert len(logged) == 1, logged
    assert "op=append_message" in logged[0]
    assert "waited=42.0s" in logged[0]
    assert writer.contention_snapshot()["long_waits"] == 5


def test_contention_class_distinguishes_our_process_from_a_foreign_holder():
    """The class must describe what was observed, never guess an owner."""
    assert writer.classify_contention(0.0, 0) == writer.CONTENTION_NONE
    assert writer.classify_contention(0.5, 0) == writer.CONTENTION_GATE
    assert writer.classify_contention(0.0, 3) == writer.CONTENTION_SQLITE
    assert writer.classify_contention(0.5, 3) == writer.CONTENTION_BOTH


def test_cancellation_state_is_reported_without_leaking_anything():
    """Diagnostics can see cancellation; the reason is our own constant."""
    assert writer.contention_snapshot()["cancellation_active"] == 0
    writer.request_write_cancellation("signal-15")
    try:
        snapshot = writer.contention_snapshot()
        assert snapshot["cancellation_active"] == 1
        assert snapshot["cancellation_reason"] == "signal-15"
        assert writer.write_cancellation_requested() is True
    finally:
        writer.clear_write_cancellation()
    assert writer.write_cancellation_requested() is False


# ---------------------------------------------------------------------------
# 5. Gate invariants (the primitive itself)
# ---------------------------------------------------------------------------


class _InterruptOnWait:
    """Condition proxy that raises a BaseException at the wait point.

    Delegates the mutex to the real condition, so the interrupt is delivered
    exactly where a signal handler's ``KeyboardInterrupt`` would land: inside
    the gate's critical section, with the mutex held.
    """

    def __init__(self, cond, injections=1, exc=KeyboardInterrupt):
        self._cond = cond
        self.remaining = injections
        self._exc = exc

    def __enter__(self):
        return self._cond.__enter__()

    def __exit__(self, *exc):
        return self._cond.__exit__(*exc)

    def notify_all(self):
        self._cond.notify_all()

    def wait(self, timeout=None):
        if self.remaining > 0:
            self.remaining -= 1
            raise self._exc("interrupted while waiting for writer admission")
        return self._cond.wait(timeout)


def test_gate_admission_survives_an_interrupt_at_the_wait_point():
    """An interrupted waiter must leave NOTHING behind.

    This is the P0.  The previous design dropped the mutex between "queue
    myself" and "claim the grant"; a ``KeyboardInterrupt`` / ``SystemExit`` in
    that window (this repo's signal handlers raise exactly those, and the CLI
    invites a second Ctrl+C) left a waiter in the queue that no thread would
    ever look at again.  The next release handed ownership to that corpse, and
    from then on EVERY write on that database path failed with "locked by
    another writer in this process" for the life of the process — a worse,
    non-recoverable version of the outage the gate exists to prevent.
    """
    gate = writer.WriterGate("interrupt-cleanup")
    assert gate.acquire(priority=1, timeout=0), "main thread should own the gate"
    gate._cond = _InterruptOnWait(gate._cond)

    interrupted = {}

    def _waiter():
        try:
            gate.acquire_admission(priority=1, timeout=30)
        except BaseException as exc:
            interrupted["exc"] = exc

    t = threading.Thread(target=_waiter, daemon=True)
    t.start()
    t.join(15)
    assert not t.is_alive()
    assert isinstance(interrupted.get("exc"), KeyboardInterrupt)

    # No phantom waiter, and the accounting used for diagnostics is truthful.
    assert gate.waiting_count() == 0
    assert gate.pending() == (0, 0, 0)
    assert gate.stats()["waiting"] == 0

    # And the gate is still usable: hand-off to a later waiter still works.
    served = threading.Event()

    def _later():
        if gate.acquire(priority=1, timeout=15):
            served.set()
            gate.release()

    later = threading.Thread(target=_later, daemon=True)
    later.start()
    gate.release()
    assert served.wait(15), "the gate was stranded by the interrupted waiter"
    later.join(15)
    assert not gate.held()


def test_gate_survives_a_system_exit_at_the_wait_point():
    """Same contract for ``SystemExit`` (interpreter shutdown / sys.exit)."""
    gate = writer.WriterGate("interrupt-cleanup-sysexit")
    assert gate.acquire(priority=0, timeout=0)
    gate._cond = _InterruptOnWait(gate._cond, exc=SystemExit)

    errors = {}

    def _waiter():
        try:
            gate.acquire_admission(priority=0, timeout=30)
        except BaseException as exc:
            errors["exc"] = exc

    t = threading.Thread(target=_waiter, daemon=True)
    t.start()
    t.join(15)
    assert isinstance(errors.get("exc"), SystemExit)
    assert gate.waiting_count() == 0
    gate.release()
    assert gate.acquire(priority=0, timeout=0)
    gate.release()


def test_writer_admission_never_strands_ownership_on_an_interrupt(tmp_path):
    """The caller's cleanup covers the narrowest window of all.

    An interrupt can land between the gate claiming ownership and
    ``acquire_admission`` returning it, i.e. after the primitive is already
    correct but before the caller knows it owns anything.
    ``_writer_admission`` must release what this thread ended up owning before
    re-raising, or the gate is stranded exactly as in the P0.
    """
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    gate = db._writer_gate
    real_acquire = gate.acquire_admission
    fired = {"n": 0}

    def _claim_then_interrupt(*, priority, timeout):
        real_acquire(priority=priority, timeout=timeout)
        fired["n"] += 1
        raise KeyboardInterrupt("interrupt between claiming and returning")

    try:
        gate.acquire_admission = _claim_then_interrupt
        with pytest.raises(KeyboardInterrupt):
            db.set_meta("k", "v")
        assert fired["n"] == 1
        assert not gate.held(), "ownership was stranded by the interrupt"
    finally:
        gate.acquire_admission = real_acquire

    # The database is still writable afterwards — the real regression.
    db.set_meta("k", "v")
    assert db.get_meta("k") == "v"
    db.close()


def test_gate_leaves_no_duplicate_or_orphaned_waiters():
    """Waiters that time out and retry must not accumulate tickets.

    A leaked ticket is quieter than a stranded owner but just as fatal: the
    gate would keep choosing an absent waiter as "next" and every real writer
    behind it would starve.  Each thread here times out several times before
    being served, and each must be served exactly once with the queue empty at
    the end.
    """
    gate = writer.WriterGate("duplicate-waiters")
    assert gate.acquire(priority=1, timeout=0)

    served = []
    served_lock = threading.Lock()
    stop = threading.Event()

    def _retrying_waiter(tag):
        while not stop.is_set():
            if gate.acquire(priority=1, timeout=0.01):
                with served_lock:
                    served.append(tag)
                gate.release()
                return

    threads = [
        threading.Thread(target=_retrying_waiter, args=(tag,), daemon=True)
        for tag in ("a", "b", "c", "d")
    ]
    for t in threads:
        t.start()

    # Everyone has queued-and-timed-out repeatedly by now.
    assert _wait_until(lambda: gate.stats()["timeouts"] >= 8)
    # ...and no ticket has leaked: at most one per live waiter thread.
    assert gate.waiting_count() <= len(threads)

    gate.release()
    for t in threads:
        t.join(15)
    stop.set()

    assert sorted(served) == ["a", "b", "c", "d"], served
    assert gate.waiting_count() == 0
    assert gate.pending() == (0, 0, 0)
    assert not gate.held()


def test_one_gate_per_database_file_regardless_of_spelling(tmp_path):
    """Aliased paths must share a gate or the coordination does nothing."""
    direct = tmp_path / "state.db"
    dotted = tmp_path / "." / "state.db"
    assert writer.gate_for(direct) is writer.gate_for(dotted)
    assert writer.gate_for(direct) is not writer.gate_for(tmp_path / "other.db")


def test_admission_reports_queueing_only_when_it_really_queued():
    """``queued`` must mean "waited behind one of ours", nothing looser.

    Acquiring a free gate takes measurable microseconds.  If that counted, the
    contention class of every write in production would read ``gate+sqlite``
    and the telemetry would point future investigations at the wrong culprit.
    """
    gate = writer.WriterGate("queued-flag")
    first = gate.acquire_admission(priority=1, timeout=0)
    assert first == writer.Admission(admitted=True, queued=False)

    refused = {}

    def _blocked():
        refused["result"] = gate.acquire_admission(priority=1, timeout=0.05)

    t = threading.Thread(target=_blocked)
    t.start()
    t.join(10)
    assert refused["result"] == writer.Admission(admitted=False, queued=True)
    gate.release()


def test_gate_is_reentrant_for_the_owning_thread():
    """A nested write on the same path must not deadlock against itself."""
    gate = writer.WriterGate("reentrant")
    assert gate.acquire(priority=1, timeout=0)
    assert gate.acquire(priority=1, timeout=0)  # would hang if not reentrant
    # Re-entry must not queue a ticket, or the owner would block itself out.
    assert gate.pending() == (0, 0, 0)
    gate.release()
    assert gate.held(), "outer level must still be held"
    gate.release()
    assert not gate.held()


def test_gate_refuses_to_release_another_writers_admission():
    """Never release a gate we do not own — that is how two writers collide."""
    gate = writer.WriterGate("foreign")
    assert gate.acquire(priority=1, timeout=0)
    error = {}

    def _foreign():
        try:
            gate.release()
        except RuntimeError as exc:
            error["exc"] = exc
        # The non-raising variant used by production cleanup must simply
        # report "not mine" instead of stealing the admission.
        error["soft"] = gate.release_if_owner()

    t = threading.Thread(target=_foreign)
    t.start()
    t.join(10)
    assert isinstance(error.get("exc"), RuntimeError)
    assert error.get("soft") is False
    assert gate.held(), "the real owner must still hold it"
    gate.release()


def test_gate_serves_same_class_waiters_in_arrival_order():
    """FIFO inside a class — priority must not reintroduce starvation."""
    gate = writer.WriterGate("fifo")
    assert gate.acquire(priority=1, timeout=0)
    served = []
    served_lock = threading.Lock()
    started = []

    def _waiter(tag):
        started.append(tag)
        assert gate.acquire(priority=1, timeout=30)
        with served_lock:
            served.append(tag)
        gate.release()

    threads = []
    for tag in ("a", "b", "c"):
        t = threading.Thread(target=_waiter, args=(tag,), daemon=True)
        threads.append(t)
        t.start()
        # Enqueue one at a time so arrival order is unambiguous.
        assert _wait_until(lambda n=len(threads): gate.waiting_count() == n)

    gate.release()
    for t in threads:
        t.join(30)
    assert served == ["a", "b", "c"], served


def test_higher_priority_waiting_sees_only_writers_that_outrank_us():
    """The preemption signal maintenance uses between bounded steps."""
    gate = writer.WriterGate("preempt")
    best_effort = writer.priority_of(writer.WRITE_BEST_EFFORT)
    assert gate.acquire(priority=best_effort, timeout=0)
    assert gate.higher_priority_waiting(best_effort) is False

    queued = threading.Event()

    def _critical():
        queued.set()
        if gate.acquire(priority=writer.priority_of(writer.WRITE_CRITICAL),
                        timeout=15):
            gate.release()

    t = threading.Thread(target=_critical, daemon=True)
    t.start()
    assert queued.wait(15)
    assert _wait_until(lambda: gate.waiting_count() == 1)
    assert gate.higher_priority_waiting(best_effort) is True
    # A critical writer is not preempted by its own class.
    assert gate.higher_priority_waiting(
        writer.priority_of(writer.WRITE_CRITICAL)
    ) is False
    gate.release()
    t.join(15)


def test_fork_reset_drops_inherited_gate_state(tmp_path):
    """Hardening: a forked child must not inherit a held gate.

    No ``os.fork()`` call site writes to state.db today, so this is a gap being
    closed rather than a bug being fixed: a child inherits the parent's mutex
    (possibly held by a thread that does not exist in the child) and the
    parent's waiter tickets, and its first write would wedge.  Registered via
    ``os.register_at_fork`` at import; exercised here directly.
    """
    gate = writer.gate_for(tmp_path / "forked.db")
    assert gate.acquire(priority=0, timeout=0)
    assert gate.held()

    writer.reset_gates_after_fork()

    assert not gate.held()
    assert gate.waiting_count() == 0
    # Usable immediately in the "child", and the stale ownership record cannot
    # make the surviving thread's cleanup raise.
    assert gate.release_if_owner() is False
    assert gate.acquire(priority=0, timeout=0)
    gate.release()


def test_unknown_write_class_is_rejected_loudly():
    """A typo must not silently demote a critical write to routine."""
    with pytest.raises(ValueError, match="unknown write class"):
        writer.priority_of("criticl")


def test_write_classes_are_ordered_critical_first():
    assert (
        writer.priority_of(writer.WRITE_CRITICAL)
        < writer.priority_of(writer.WRITE_NORMAL)
        < writer.priority_of(writer.WRITE_BEST_EFFORT)
    )


def test_operation_names_come_from_the_write_callback():
    """Telemetry attributes a write without touching any call site."""

    class _Fake:
        def append_message(self):
            def _do(conn):
                return None

            return _do

    assert writer.describe_op(_Fake().append_message()) == "append_message"
    assert writer.describe_op(lambda conn: None, "explicit") == "explicit"
