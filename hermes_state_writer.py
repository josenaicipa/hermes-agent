"""Writer coordination, fairness and contention telemetry for ``state.db``.

Why this module exists
======================

SQLite allows exactly **one** writer per database file, and its busy handling
is not a queue: a blocked writer re-probes the lock and whoever probes at the
right instant wins.  Nothing guarantees that a writer which has been waiting
longer gets served first, so under sustained load a particular writer can lose
every probe until its bounded retry budget runs out.

A single Hermes process holds **many** independent ``SessionDB`` objects
against the *same* ``state.db`` — the gateway runner, the gateway session
store, the gateway mirror, every running agent, the cron scheduler, slash
commands, the async token-accounting thread, plus routine maintenance (WAL
checkpoint, bounded FTS merge).  Each of those has its own connection and its
own ``threading.Lock``, so before this module they contended at the SQLite
layer with **no ordering at all** and with no way to tell a transcript append
apart from best-effort accounting.

That produced starvation, not deadlock.  Two vpsclone incidents on 2026-08-03
show the shape:

* 13:51 — ``append_message`` gave up with ``database is locked`` after ~16.2 s
  (15 retries against a 1 s ``busy_timeout``).  The holder was never logged.
* 23:04-23:05 — a transcript row committed at 23:04:54.480 in the *middle* of
  a window where other appends failed with ``database is locked`` at 23:05:02
  and 23:05:18, and a post-incident ``BEGIN IMMEDIATE; ROLLBACK`` probe
  succeeded in 0.05 ms.  The database was writable throughout; specific
  writers simply kept losing.  The writer that outlasted every budget appears
  to have held the lock for tens of seconds.

**Attribution caveat (deliberate).**  The ~55 s holder was never identified.
``hermes insights --days 30`` is *correlated* (it was the last command started
before the window) and nothing more; ``UPDATE messages SET active=1 WHERE
active IS NULL`` was never established as the cause.  Nothing in this module
or its callers may be described as fixing a proven root cause.  What is proven
is the *class* of failure: patience was uniform and short, ordering was
absent, and no telemetry existed to attribute either.

Two mechanisms live here, covering different halves of that class:

``WriterGate``
    Removes the *in-process* half.  One gate per canonical database path,
    shared by every ``SessionDB`` pointing at that file, so at most one writer
    *of this process* attempts ``BEGIN IMMEDIATE`` at a time and a critical
    transcript append is admitted ahead of queued routine and best-effort
    work.  It has no effect on other processes — SQLite still arbitrates those
    — and it never touches another writer's transaction: it is a plain Python
    admission token, acquired strictly *outside* the per-connection lock and
    released only by the thread that holds it.

    The whole acquire is a single critical section on one
    ``threading.Condition``: a waiter registers a ticket, then claims the gate
    *itself* when it becomes the best ticket.  Ownership is never handed to a
    thread that is not running, and a waiter's bookkeeping is undone in a
    ``finally`` that runs with the mutex held — so an interrupt
    (``KeyboardInterrupt``/``SystemExit`` from the signal handlers this repo
    installs, or an interpreter shutdown) cannot leave a phantom waiter or a
    stranded owner behind.  An earlier hand-off design could: it dropped the
    mutex between "queue myself" and "claim the grant", and an interrupt in
    that window permanently wedged every writer on the path.

``record_outcome`` / ``note_long_wait`` / ``contention_snapshot``
    Bounded, content-free telemetry.  Both incidents were unresolvable
    because nothing recorded *who* was waiting, *how long*, or *on what*.
    Every wait is now attributed to a contention class (``gate`` for our own
    process, ``sqlite`` for a holder we cannot see, ``gate+sqlite`` for both)
    and reported with an operation name and a duration — never with
    transcript content, user identifiers or credentials.  Reporting is
    threshold-gated and rate-limited so routine sub-second lock blips stay
    silent while suppressed events are still counted and folded into the next
    emitted line.

The *cross-process* half is handled by the caller (``SessionDB``): a critical
write keeps re-probing while contention stays healthy (SQLITE_BUSY/LOCKED, so
the database is answering) and stops on explicit cancellation
(:func:`request_write_cancellation`, ``SessionDB.close()``, or interpreter
shutdown via :func:`_cancel_writes_at_interpreter_exit`), on a classified
permanent failure, or on the caller's anti-hang watchdog.  When a write
genuinely cannot become durable it must still raise, so the turn fails closed
rather than pretending the transcript was persisted.

This module must not import ``hermes_state`` (cycle): ``hermes_state`` and the
``SessionDB`` mixin modules both import *from* here.
"""

import logging
import os
import threading
import time
from typing import Any, Dict, NamedTuple, Optional, Set, Tuple

# Contention is logged under the "hermes_state" logger so existing log
# routing, filtering and test capture for the session store keep working.
logger = logging.getLogger("hermes_state")


# ── Write classes ──────────────────────────────────────────────────────────
#
# The class is a durability contract, not a hint:
#
#   critical    — the caller cannot proceed correctly without this row being
#                 durable.  Losing it fails the turn closed (transcript
#                 appends: ``session_persistence_failed``).  Admitted first,
#                 and kept waiting while contention stays healthy.
#   normal      — ordinary state the caller reports on but can retry later
#                 (session metadata, routing index, lifecycle rows).  Bounded
#                 by exactly the patience it had before this module existed.
#   best_effort — the caller already treats failure as loggable-and-continue
#                 (async token accounting, WAL checkpoint, FTS merge).  These
#                 must always yield to the two classes above.

WRITE_CRITICAL = "critical"
WRITE_NORMAL = "normal"
WRITE_BEST_EFFORT = "best_effort"

#: Priority order used by the gate.  Lower is admitted first.
_PRIORITIES: Dict[str, int] = {
    WRITE_CRITICAL: 0,
    WRITE_NORMAL: 1,
    WRITE_BEST_EFFORT: 2,
}

_NUM_PRIORITIES = 3


def priority_of(write_class: str) -> int:
    """Return the gate priority for *write_class*.

    Raises ``ValueError`` for an unknown class rather than silently degrading
    a critical write to a routine one — a typo here would be invisible in
    production but would quietly remove the priority that keeps transcript
    appends ahead of accounting.
    """
    try:
        return _PRIORITIES[write_class]
    except KeyError:
        raise ValueError(
            f"unknown write class {write_class!r}; expected one of "
            f"{sorted(_PRIORITIES)}"
        ) from None


# ── Contention classes (telemetry vocabulary) ─────────────────────────────

#: Waited only behind another writer *of this process* — queueing for admission,
#: or (bounded, one slice at a time) for the per-connection lock a caller's own
#: maintenance holds.  Both are in-process waits; distinguishing them is what
#: ``gate_timeouts`` and ``conn_lock_timeouts`` are for.
CONTENTION_GATE = "gate"
#: Waited on SQLite itself — the holder is another process, or an in-process
#: path that writes outside the gate (schema init, VACUUM).  Deliberately
#: named after the observation, not a guess at the owner.
CONTENTION_SQLITE = "sqlite"
#: Both of the above happened during the same write.
CONTENTION_BOTH = "gate+sqlite"
#: No measurable wait.
CONTENTION_NONE = "none"

#: Why a write stopped waiting.  Reported, never used for control flow.
STOP_BUDGET = "budget"
STOP_CANCELLED = "cancelled"
STOP_WATCHDOG = "watchdog"


def classify_contention(gate_waited_s: float, sqlite_busy_retries: int) -> str:
    """Name the source of a write's wait from what was actually observed."""
    gated = gate_waited_s > 0.0
    busy = sqlite_busy_retries > 0
    if gated and busy:
        return CONTENTION_BOTH
    if busy:
        return CONTENTION_SQLITE
    if gated:
        return CONTENTION_GATE
    return CONTENTION_NONE


# ── Cancellation (process-wide) ────────────────────────────────────────────
#
# A critical write has no fixed wall-clock budget: as long as SQLite keeps
# answering SQLITE_BUSY the database is healthy and someone else is mid
# transaction, so giving up would report a failure the database did not have.
# That patience needs an OFF switch that real code sets, and these are the
# callers that set it today (no invented token nobody uses):
#
#   * cli.py ``_signal_handler``   — SIGTERM / SIGHUP in interactive mode
#   * cli.py ``_signal_handler_q`` — SIGTERM in single-query mode
#   * gateway/run.py ``shutdown_signal_handler`` — SIGINT / SIGTERM
#   * SessionDB.close() — per instance, via SessionDB.request_write_cancellation
#   * interpreter shutdown — :func:`_cancel_writes_at_interpreter_exit` below,
#     for the many exits that never involve a signal at all
#
# Deliberately NOT wired to a first Ctrl+C in the TUI: that is a steer/stop for
# the model, not a shutdown, and cancelling a transcript append there would
# turn an ordinary interruption into a lost row.
#
# Cancellation does not abandon a write — it collapses the extended patience
# back to the ordinary bounded budget, so a cancelled critical write is never
# *less* patient than the same write was before any of this existed.

#: Plain module globals, NOT a threading.Event and NOT counter-protected by
#: _stats_lock, because request_write_cancellation() runs inside real signal
#: handlers.  A signal handler executes on the main thread, so touching
#: any lock this module's other functions can already hold would self-deadlock
#: the process it is trying to shut down.  A bare bool assignment/read is
#: atomic in CPython and needs no lock; the retry loop polls it at most one
#: gate slice (~1 s) later, which is immediate enough for shutdown.
_cancel_requested = False
_cancel_reason = ""


def request_write_cancellation(reason: str = "shutdown") -> None:
    """Stop extending critical-write patience process-wide.  Never raises.

    Signal-handler safe: two global assignments, no locks, no logging, no
    allocation beyond the caller's own string.

    Cancellation does NOT abandon a write.  It collapses a critical write's
    extended patience back to the ordinary bounded budget (see
    ``SessionDB._execute_write``), so a cancelled critical write is still at
    least as patient as the same write was before writer classes existed —
    shutdown must not become a new way to lose a transcript row.
    """
    global _cancel_requested, _cancel_reason
    _cancel_reason = reason
    _cancel_requested = True


def write_cancellation_requested() -> bool:
    """True when process-wide write cancellation has been requested."""
    return _cancel_requested


def write_cancellation_reason() -> str:
    """Last reason passed to :func:`request_write_cancellation` (or "")."""
    return _cancel_reason


def clear_write_cancellation() -> None:
    """Undo :func:`request_write_cancellation` (tests, supervised restarts)."""
    global _cancel_requested, _cancel_reason
    _cancel_requested = False
    _cancel_reason = ""


# ── Interpreter exit: the shutdowns nobody signals ─────────────────────────
#
# The handlers listed above cover shutdowns that arrive as a signal.  Plenty do
# not: a CLI command returning, ``sys.exit()``, a script whose main thread
# simply ends.  In that shape the last thing CPython does is join every
# non-daemon thread — including the ``ThreadPoolExecutor`` workers
# ``asyncio.to_thread`` runs on — so one critical write still waiting on a
# foreign lock holder turns "the process finished" into "the process finishes
# when the watchdog fires".  Nothing kills it in between, because nobody sent
# a signal.
#
# ``atexit.register`` cannot fix that, however natural it looks:
# ``threading._shutdown()`` (its own callbacks, then the non-daemon joins) runs
# BEFORE any ``atexit`` callback, and ``concurrent.futures.thread`` joins its
# workers from one of those ``threading`` callbacks.  An ``atexit`` hook would
# cancel *after* the join it exists to shorten — a hook that reads as coverage
# and provides none.
#
# ``threading`` invokes its callbacks in REVERSE registration order, so
# registering after ``concurrent.futures.thread`` puts us in front of its
# ``_python_exit``.  We import that module here rather than hoping something
# else already did: the default executor is created lazily, so its
# registration could otherwise land after ours and run first.
#
# The registry is private, so its absence is handled rather than assumed — and
# deliberately NOT papered over with an ``atexit`` fallback that would not run
# in time.  Where it is missing, the wait stays bounded by
# ``SessionDB._CRITICAL_WRITE_WATCHDOG_S``, which is exactly why that constant
# is capped at a duration an operator would call bounded.

#: Cancellation reason recorded for a shutdown that arrived without a signal.
EXIT_CANCEL_REASON = "interpreter-exit"


def _cancel_writes_at_interpreter_exit() -> None:
    """Collapse critical-write patience as interpreter shutdown begins.

    Same contract as every other cancellation: the write is not abandoned, its
    patience drops back to the ordinary baseline budget, and a write that
    still cannot become durable raises, so the caller fails closed instead of
    reporting an unwritten row as persisted.
    """
    request_write_cancellation(EXIT_CANCEL_REASON)


def _install_exit_cancellation() -> str:
    """Register the exit hook ahead of the pool-worker join.  Never raises.

    Returns the mechanism actually in force (``"threading"`` or ``"none"``).
    """
    register = getattr(threading, "_register_atexit", None)
    if register is None:  # pragma: no cover - non-CPython runtimes only
        return "none"
    try:
        # Import order is the whole point: callbacks run in reverse
        # registration order, so concurrent.futures.thread must already be
        # registered when we register.
        import concurrent.futures.thread  # noqa: F401

        register(_cancel_writes_at_interpreter_exit)
    except Exception:  # pragma: no cover - import failure / already shutting down
        return "none"
    return "threading"


_exit_cancellation_mechanism = _install_exit_cancellation()


def exit_cancellation_mechanism() -> str:
    """Which mechanism cancels write patience at interpreter exit.

    ``"threading"`` — registered ahead of ``concurrent.futures``' worker join,
    the only placement that shortens a signal-less shutdown.  ``"none"`` —
    nothing does, and the caller's anti-hang watchdog is the only bound.
    Diagnostics and tests read this; no control flow depends on it.
    """
    return _exit_cancellation_mechanism


# ── The gate ───────────────────────────────────────────────────────────────


class Admission(NamedTuple):
    """Outcome of one admission request.

    ``queued`` distinguishes "the gate was free" from "we waited behind
    another writer of this process".  Without it, contention classification
    would be useless: acquiring a free gate still takes a measurable number of
    microseconds, so any elapsed-time-based test would report in-process
    contention on every single write.
    """

    admitted: bool
    queued: bool


class WriterGate:
    """Fair, priority-aware admission gate for one canonical database path.

    Ordering guarantees:

    * At most one admitted writer per database path per process.
    * Barging-free: an arriving writer takes the free-gate fast path only when
      nobody is queued.  Otherwise it registers a ticket and can claim the
      gate only once its ticket is the best one outstanding, so FIFO holds
      within a priority class and starvation *within a class* is impossible.
    * Across classes, ``critical`` is admitted before ``normal``, which is
      admitted before ``best_effort``.  A steady stream of critical writes can
      therefore delay best-effort maintenance — the intended trade: accounting
      and FTS merges are retried on the next cadence, a dropped transcript row
      ends the turn.
    * Re-entrant per thread, so a write helper that nests another write on the
      same path cannot deadlock against itself.
    * Interruption-safe: registering, claiming and un-registering a ticket all
      happen under one mutex, and the un-register runs in a ``finally``.  A
      ``BaseException`` anywhere in ``acquire_admission`` therefore leaves no
      phantom waiter, and ownership is only ever taken by the thread that is
      about to return with it (see also :meth:`release_if_owner`, which the
      caller uses so an interrupt between claiming and returning cannot strand
      the gate).

    The gate is intentionally *not* a substitute for SQLite's own locking, and
    it does **not** cover every writer in the process: schema init/migration,
    ``VACUUM``, the opt-in FTS storage optimize, offline repair and the
    checkpoint in ``close()`` all touch SQLite outside it (see
    ``SessionDB._execute_write``'s docstring for why).  Cross-process
    contention keeps its existing semantics, which is why callers must still
    handle ``database is locked``.

    Non-preemption is a deliberate limit, not an oversight: an admitted writer
    keeps the gate until it releases.  Long or already-non-blocking
    maintenance must therefore NOT hold admission across its whole run — it
    acquires per bounded step and re-checks :meth:`higher_priority_waiting`
    between steps.
    """

    # Deliberately no __slots__: there is one gate per database path (never a
    # hot allocation), and tests must be able to substitute the condition or
    # the acquire method to inject an interrupt at an exact point.

    def __init__(self, key: str) -> None:
        self.key = key
        self._cond = threading.Condition(threading.Lock())
        self._owner: Optional[int] = None
        self._depth = 0
        self._seq = 0
        # (priority, seq) tickets of every queued waiter.  A set plus min() is
        # O(waiters) per wake, and waiters here are bounded by the number of
        # SessionDB writers in one process (tens); the simplicity is worth far
        # more than the constant factor on this path.
        self._tickets: Set[Tuple[int, int]] = set()
        # Diagnostics only — never used for control flow.
        self._admissions = 0
        self._handoffs = 0
        self._timeouts = 0

    # -- introspection (diagnostics and tests) --

    def pending(self) -> Tuple[int, ...]:
        """Queued waiter count per priority class, highest priority first."""
        with self._cond:
            counts = [0] * _NUM_PRIORITIES
            for prio, _seq in self._tickets:
                counts[prio] += 1
            return tuple(counts)

    def waiting_count(self) -> int:
        """Total number of writers currently queued on this gate."""
        with self._cond:
            return len(self._tickets)

    def higher_priority_waiting(self, priority: int) -> bool:
        """True when a queued waiter outranks *priority*.

        Long-running maintenance calls this between bounded steps so it can
        yield instead of holding a non-preemptible admission while a
        transcript append waits.
        """
        with self._cond:
            return any(prio < priority for prio, _seq in self._tickets)

    def held(self) -> bool:
        with self._cond:
            return self._owner is not None

    def owns(self) -> bool:
        """True when the calling thread currently holds this gate.

        Deliberately lock-free.  ``_owner`` is only ever set to a thread's own
        ident by that thread (there is no hand-off in this design) and only
        cleared by the owner, so the question "is it me?" cannot observe a
        transitional value — while taking the mutex here would put a lock
        acquisition on the hot path of every write just to answer it.
        """
        return self._owner == threading.get_ident()

    def stats(self) -> Dict[str, int]:
        with self._cond:
            return {
                "admissions": self._admissions,
                "handoffs": self._handoffs,
                "timeouts": self._timeouts,
                "waiting": len(self._tickets),
            }

    # -- acquire / release --

    def acquire_admission(self, *, priority: int, timeout: float) -> Admission:
        """Admit the calling thread, reporting whether it had to queue.

        ``admitted=False`` means "no attempt was made" — the caller has not
        touched SQLite and must treat it as contention, not as a failed
        write.  Never raises for contention; a negative or zero *timeout*
        still performs the uncontended fast path so a caller whose budget is
        already spent can make one last honest attempt.
        """
        ident = threading.get_ident()
        with self._cond:
            if self._owner == ident:
                # Re-entrant: the caller already owns the gate.
                self._depth += 1
                return Admission(True, False)
            if self._owner is None and not self._tickets:
                # Uncontended and nobody queued, so this cannot barge.
                self._owner = ident
                self._depth = 1
                self._admissions += 1
                return Admission(True, False)

            self._seq += 1
            ticket = (priority, self._seq)
            deadline = time.monotonic() + max(0.0, timeout)
            try:
                # Inside the try: an interrupt delivered between add() and the
                # loop must still reach the finally that removes the ticket.
                self._tickets.add(ticket)
                while True:
                    if self._owner is None and min(self._tickets) == ticket:
                        self._owner = ident
                        self._depth = 1
                        self._admissions += 1
                        self._handoffs += 1
                        return Admission(True, True)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._timeouts += 1
                        return Admission(False, True)
                    # Condition.wait re-acquires the mutex before returning OR
                    # before propagating an exception, so the finally below
                    # always runs with the invariant protected.
                    self._cond.wait(remaining)
            finally:
                self._tickets.discard(ticket)
                if self._owner is None and self._tickets:
                    # We left the queue while the gate was free: whoever is
                    # next has to be told, since no release() will follow.
                    self._cond.notify_all()

    def acquire(self, *, priority: int, timeout: float) -> bool:
        """``acquire_admission`` reduced to its admitted flag."""
        return self.acquire_admission(priority=priority, timeout=timeout).admitted

    def release(self) -> None:
        """Release one level of ownership held by the calling thread.

        Raises ``RuntimeError`` when the caller is not the owner.  This is a
        hard invariant, not a nicety: releasing a gate we do not own would let
        two of our writers attempt ``BEGIN IMMEDIATE`` concurrently again, and
        the whole point of the gate is that we never interfere with another
        writer's transaction.
        """
        self._release(strict=True)

    def release_if_owner(self) -> bool:
        """Release one level of ownership if this thread has it.

        Returns False instead of raising when it does not.  Production callers
        use this in their cleanup path: an interrupt delivered between
        claiming the gate and returning the admission would otherwise strand
        ownership with nobody able to release it, and a post-fork state reset
        (see :func:`reset_gates_after_fork`) legitimately drops ownership too.
        """
        return self._release(strict=False)

    def _release(self, *, strict: bool) -> bool:
        ident = threading.get_ident()
        with self._cond:
            if self._owner != ident:
                if strict:
                    raise RuntimeError(
                        "WriterGate.release() from a thread that does not own "
                        f"the gate for {self.key!r}"
                    )
                return False
            self._depth -= 1
            if self._depth > 0:
                return True
            self._owner = None
            self._depth = 0
            if self._tickets:
                # Every waiter re-evaluates its own claim: ownership is never
                # pushed onto a thread that may already be unwinding.
                self._cond.notify_all()
            return True

    # -- fork hardening --

    def _reinit_after_fork(self) -> None:
        """Drop inherited state in a forked child.

        A child inherits the parent's mutex byte-for-byte, including one held
        by a thread that does not exist in the child, plus tickets for threads
        that do not exist either.  Both would wedge the child's first write.
        Ownership is dropped rather than guessed, which is why production
        cleanup goes through :meth:`release_if_owner`.
        """
        self._cond = threading.Condition(threading.Lock())
        self._owner = None
        self._depth = 0
        self._tickets = set()


# ── Gate registry (one per canonical database path) ───────────────────────

_gates_lock = threading.Lock()
_gates: Dict[str, WriterGate] = {}


def canonical_db_key(db_path: Any) -> str:
    """Canonicalize a database path so aliases share one gate.

    ``realpath`` collapses symlinks and relative spellings; ``normcase``
    folds case on filesystems where the OS does (macOS, Windows).  Two
    ``SessionDB`` objects opened as ``~/.hermes/state.db`` and
    ``/home/u/.hermes/state.db`` are the same file and must share a gate, or
    the coordination silently does nothing.
    """
    text = str(db_path)
    try:
        return os.path.normcase(os.path.realpath(text))
    except OSError:
        # A path we cannot resolve (permissions on a parent directory) still
        # needs *a* stable key; falling back keeps coordination working for
        # every caller that spells it the same way.
        return os.path.normcase(text)


def gate_for(db_path: Any) -> WriterGate:
    """Return the process-wide gate for *db_path*, creating it on demand."""
    key = canonical_db_key(db_path)
    with _gates_lock:
        gate = _gates.get(key)
        if gate is None:
            gate = WriterGate(key)
            _gates[key] = gate
        return gate


def reset_gates_after_fork() -> None:
    """Re-create every lock this module owns.  Registered as a fork hook.

    Nothing in the tree forks while holding a gate today, so this is
    hardening, not a bug fix: it makes an inherited-lock deadlock impossible
    for any future ``os.fork()`` caller (multiprocessing's default start
    method on Linux) instead of relying on nobody ever adding one.
    """
    global _gates_lock, _stats_lock
    _gates_lock = threading.Lock()
    _stats_lock = threading.Lock()
    for gate in list(_gates.values()):
        try:
            gate._reinit_after_fork()
        except Exception:  # pragma: no cover - a child must still start
            pass


if hasattr(os, "register_at_fork"):  # pragma: no branch - POSIX only
    os.register_at_fork(after_in_child=reset_gates_after_fork)


# ── Bounded contention telemetry ──────────────────────────────────────────
#
# Bounding rules, in order:
#   1. Successful writes whose total wait is below the threshold are counted
#      and never logged.  A couple of 20-150 ms retries is normal life on a
#      shared database and must not produce a log line.
#   2. Above the threshold, at most one line per (kind, operation, contention
#      class) per cooldown window.  Suppressed events are counted and the
#      count is folded into the next line that does get emitted, so the
#      signal survives even when the detail is dropped.
#   3. Failures are reported on their own throttle so a storm cannot spam,
#      but the suppressed count is always carried.  The exception itself
#      still propagates to the caller: this telemetry never decides whether
#      a write failed, only how loudly the wait is described.
#   4. A write that is STILL waiting emits a throttled heartbeat.  Without it
#      an extended critical wait is indistinguishable from a hang, which is
#      exactly the blindness that made both incidents unresolvable.

#: Total wait (gate + SQLite) above which a *successful* write is reported.
CONTENTION_LOG_THRESHOLD_S = 2.0
#: Minimum gap between reports for the same (kind, operation, class).
CONTENTION_LOG_COOLDOWN_S = 60.0
#: Minimum gap between "still waiting" heartbeats for the same operation.
LONG_WAIT_HEARTBEAT_S = 15.0

_stats_lock = threading.Lock()
_counters: Dict[str, int] = {}
_last_log: Dict[Tuple[str, str, str], float] = {}
_suppressed: Dict[Tuple[str, str, str], int] = {}
_slowest: Dict[str, float] = {}
_busy_by_op: Dict[str, int] = {}
_conn_lock_by_op: Dict[str, int] = {}


def _bump(name: str, amount: int = 1) -> None:
    _counters[name] = _counters.get(name, 0) + amount


def note_busy_retry(*, op: str, write_class: str) -> None:
    """Count one ``database is locked`` retry **as it happens**.

    Deliberately separate from :func:`record_outcome`, which can only report
    once a write has finished.  Both 2026-08-03 incidents were diagnosed after
    the fact from message-row timestamps because nothing was observable *while*
    a writer was being starved.  Counting here makes an in-flight starvation
    visible to diagnostics (and lets a regression test synchronise on real
    contention instead of a sleep).  Never logs — the throttled reporting all
    happens in :func:`record_outcome` and :func:`note_long_wait`.
    """
    try:
        with _stats_lock:
            _bump("sqlite_busy_retries")
            _busy_by_op[op] = _busy_by_op.get(op, 0) + 1
            _bump(f"busy_class.{write_class}")
    except Exception:  # pragma: no cover - telemetry must never break a write
        pass


def busy_retries_for(op: str) -> int:
    """Busy retries observed so far for *op*.  Diagnostics and tests."""
    with _stats_lock:
        return _busy_by_op.get(op, 0)


def note_conn_lock_timeout(*, op: str, write_class: str) -> None:
    """Count one write that handed admission back **as it happens**.

    A writer holds process-wide admission while it waits for its own
    ``SessionDB``'s connection lock, which maintenance (``vacuum``,
    ``rebuild_fts``, a checkpoint, ``close``) takes without any admission at
    all.  That wait is therefore sliced: on timeout the writer releases
    admission — so no other ``SessionDB`` on the file is denied — and re-queues.

    Live, for the same reason as :func:`note_busy_retry`: "our own maintenance
    is currently starving writers" must be observable WHILE it happens, not only
    once the write finishes (or fails).  Never logs; the throttled reporting
    stays in :func:`record_outcome` and :func:`note_long_wait`.
    """
    try:
        with _stats_lock:
            _bump("conn_lock_timeouts")
            _conn_lock_by_op[op] = _conn_lock_by_op.get(op, 0) + 1
            _bump(f"conn_lock_class.{write_class}")
    except Exception:  # pragma: no cover - telemetry must never break a write
        pass


def conn_lock_timeouts_for(op: str) -> int:
    """Connection-lock timeouts observed so far for *op*.  Diagnostics/tests."""
    with _stats_lock:
        return _conn_lock_by_op.get(op, 0)


def _throttle_key(kind: str, op: str, contention: str) -> Tuple[str, str, str]:
    return (kind, op, contention)


def _claim_log_slot(
    kind: str, op: str, contention: str, now: float, cooldown: float
) -> Optional[int]:
    """Return suppressed-count to report, or None when this event is muted.

    Called with ``_stats_lock`` held.
    """
    key = _throttle_key(kind, op, contention)
    last = _last_log.get(key)
    if last is not None and (now - last) < cooldown:
        _suppressed[key] = _suppressed.get(key, 0) + 1
        return None
    _last_log[key] = now
    return _suppressed.pop(key, 0)


def note_long_wait(
    *,
    op: str,
    write_class: str,
    waited_s: float,
    attempts: int,
    sqlite_busy_retries: int,
    gate_waited_s: float,
) -> None:
    """Heartbeat for a write that is still waiting.  Never raises.

    Content-free, same as :func:`record_outcome`: an operation name from this
    source tree and numbers.
    """
    try:
        contention = classify_contention(gate_waited_s, sqlite_busy_retries)
        now = time.monotonic()
        with _stats_lock:
            _bump("long_waits")
            suppressed = _claim_log_slot(
                "waiting", op, contention, now, LONG_WAIT_HEARTBEAT_S
            )
        if suppressed is None:
            return
        logger.warning(
            "state.db write STILL WAITING: op=%s class=%s contention=%s "
            "waited=%.1fs attempts=%d busy_retries=%d (gate=%.1fs)%s",
            op, write_class, contention, waited_s, attempts,
            sqlite_busy_retries, gate_waited_s,
            f" (+{suppressed} similar suppressed)" if suppressed else "",
        )
    except Exception:  # pragma: no cover - telemetry must never break a write
        pass


def record_outcome(
    *,
    op: str,
    write_class: str,
    succeeded: bool,
    waited_s: float,
    gate_waited_s: float,
    attempts: int,
    sqlite_busy_retries: int,
    gate_timeouts: int = 0,
    conn_lock_timeouts: int = 0,
    budget_s: Optional[float] = None,
    stop_reason: Optional[str] = None,
) -> None:
    """Record one completed write attempt sequence.  Never raises.

    Deliberately content-free: *op* is a method name chosen in this
    repository, and every other field is a number or one of this module's own
    constants.  No session id, user id, chat id, message body, tool argument
    or credential is accepted here, so the telemetry cannot leak transcript
    content or PII even if log level is turned all the way up.

    ``gate_timeouts`` and ``conn_lock_timeouts`` split the in-process wait by
    sub-cause: refused admission (another writer of ours) versus a
    per-connection lock held by work that takes no admission (maintenance:
    checkpoint, VACUUM, FTS rebuild).  Both are counted so "our own maintenance
    is starving writers" is a readable state instead of an unexplained wait.
    """
    try:
        contention = classify_contention(gate_waited_s, sqlite_busy_retries)
        now = time.monotonic()
        with _stats_lock:
            _bump("writes")
            if not succeeded:
                _bump("failures")
                if stop_reason:
                    _bump(f"stopped.{stop_reason}")
            # ``sqlite_busy_retries`` and ``conn_lock_timeouts`` are already
            # counted live by note_busy_retry() / note_conn_lock_timeout(); they
            # arrive here only to be reported.
            if gate_timeouts:
                _bump("gate_timeouts", gate_timeouts)
            if contention != CONTENTION_NONE:
                _bump(f"contended.{contention}")
            if waited_s > _slowest.get(op, 0.0):
                _slowest[op] = waited_s

            if not succeeded:
                kind = "failed"
            elif waited_s >= CONTENTION_LOG_THRESHOLD_S:
                kind = "slow"
            else:
                _bump("quiet")
                return
            suppressed = _claim_log_slot(
                kind, op, contention, now, CONTENTION_LOG_COOLDOWN_S
            )

        if suppressed is None:
            return

        extra = f" (+{suppressed} similar suppressed)" if suppressed else ""
        budget = f", budget={budget_s:.1f}s" if budget_s is not None else ""
        stopped = f", stopped_on={stop_reason}" if stop_reason else ""
        # Only when it happened: a writer that never queued behind our own
        # maintenance must not carry a field implying it did.
        busy_conn = (
            f", busy_connection={conn_lock_timeouts}" if conn_lock_timeouts else ""
        )
        if kind == "failed":
            logger.error(
                "state.db write GAVE UP: op=%s class=%s contention=%s "
                "waited=%.2fs (gate=%.2fs) attempts=%d busy_retries=%d%s%s%s%s",
                op, write_class, contention, waited_s, gate_waited_s,
                attempts, sqlite_busy_retries, busy_conn, budget, stopped, extra,
            )
        else:
            logger.warning(
                "state.db write was starved but succeeded: op=%s class=%s "
                "contention=%s waited=%.2fs (gate=%.2fs) attempts=%d "
                "busy_retries=%d%s%s%s",
                op, write_class, contention, waited_s, gate_waited_s,
                attempts, sqlite_busy_retries, busy_conn, budget, extra,
            )
    except Exception:  # pragma: no cover - telemetry must never break a write
        pass


def contention_snapshot() -> Dict[str, Any]:
    """Counters for diagnostics and tests.  Content-free by construction.

    Every key is either a counter name or an operation name from this source
    tree.  Gate statistics are deliberately AGGREGATED rather than keyed by
    database path: a canonical path embeds the OS username (and, for profiles,
    a profile name), and this dict is exactly the kind of thing a diagnostics
    command dumps into a log or a bug report.  The useful axis for attribution
    is the operation, and that is preserved in full.
    """
    with _stats_lock:
        snapshot: Dict[str, Any] = dict(_counters)
        snapshot["slowest_wait_s"] = dict(_slowest)
        snapshot["busy_retries_by_op"] = dict(_busy_by_op)
        snapshot["conn_lock_timeouts_by_op"] = dict(_conn_lock_by_op)
        snapshot["suppressed"] = {
            "/".join(key): count for key, count in _suppressed.items()
        }
    with _gates_lock:
        gates = [gate.stats() for gate in _gates.values()]
    aggregate = {"count": len(gates)}
    for field in ("admissions", "handoffs", "timeouts", "waiting"):
        aggregate[field] = sum(g[field] for g in gates)
    snapshot["gates"] = aggregate
    snapshot["cancellation_active"] = int(write_cancellation_requested())
    # A reason is one of this module's / the CLI's own short constants
    # ("shutdown", "signal-15", "close"), never caller data.
    snapshot["cancellation_reason"] = write_cancellation_reason()
    return snapshot


def reset_contention_stats() -> None:
    """Clear telemetry counters.  Intended for tests and diagnostics resets.

    Deliberately does not touch the gates: clearing live admission state
    could strand a held gate or let two writers in at once.
    """
    with _stats_lock:
        _counters.clear()
        _last_log.clear()
        _suppressed.clear()
        _slowest.clear()
        _busy_by_op.clear()
        _conn_lock_by_op.clear()


def describe_op(fn: Any, explicit: Optional[str] = None) -> str:
    """Best-effort short operation name for telemetry.

    Prefers an explicit name.  Otherwise derives one from the write
    callback's ``__qualname__`` — ``SessionDB.append_message.<locals>._do``
    becomes ``append_message`` — so every existing ``_execute_write`` call
    site gets attributable telemetry without being touched.  The result is
    always a bounded identifier from this source tree, never caller data.
    """
    if explicit:
        return explicit
    qualname = getattr(fn, "__qualname__", "") or getattr(fn, "__name__", "")
    if not qualname:
        return "unknown"
    parts = [p for p in qualname.split(".") if p and p != "<locals>"]
    # Drop the inner callback name (_do / _apply / ...) and the class.
    named = [p for p in parts if not p.startswith("_") and p != "SessionDB"]
    if named:
        return named[-1]
    return parts[-1] if parts else "unknown"


__all__ = [
    "Admission",
    "CONTENTION_BOTH",
    "CONTENTION_GATE",
    "CONTENTION_LOG_COOLDOWN_S",
    "CONTENTION_LOG_THRESHOLD_S",
    "CONTENTION_NONE",
    "CONTENTION_SQLITE",
    "EXIT_CANCEL_REASON",
    "LONG_WAIT_HEARTBEAT_S",
    "STOP_BUDGET",
    "STOP_CANCELLED",
    "STOP_WATCHDOG",
    "WRITE_BEST_EFFORT",
    "WRITE_CRITICAL",
    "WRITE_NORMAL",
    "WriterGate",
    "busy_retries_for",
    "canonical_db_key",
    "classify_contention",
    "clear_write_cancellation",
    "conn_lock_timeouts_for",
    "contention_snapshot",
    "describe_op",
    "exit_cancellation_mechanism",
    "gate_for",
    "note_busy_retry",
    "note_conn_lock_timeout",
    "note_long_wait",
    "priority_of",
    "record_outcome",
    "request_write_cancellation",
    "reset_contention_stats",
    "reset_gates_after_fork",
    "write_cancellation_reason",
    "write_cancellation_requested",
]
