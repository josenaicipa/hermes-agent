"""Autonomous profile labels for non-``no_agent`` cron agentic runs.

Hermes V2.9: ``autonomous_profile`` is a **routing/audit label only**. It
records *what kind of work* a job is (so downstream routing, reporting and
the agentic-efficiency ledger can classify it) and nothing else. It carries
no wall-clock, turn, token or USD envelope, and the supervisor in this
module never stops a run for any of those reasons.

What is still enforced here:

* the label set is fail-closed validated — unknown, empty, numeric or
  ``0``/``unlimited``-style sentinel values are rejected *before* an agent
  is constructed, so a typo can never silently fall back to a default;
* precedence stays ``job['autonomous_profile']`` →
  ``config['cron']['autonomous_limits']['default_profile']`` → ``standard``.

What deliberately no longer exists: wall-clock timeout, max-turns, token
budget, USD budget, the inactivity watchdog and the one-shot rescope
guard that reacted to them. Cron agentic runs are bounded by the things
that are actually authoritative — manual cancellation, provider-native
rate limits, real model context limits and compression.

``no_agent`` script-only jobs never enter this module — they do not spawn
``AIAgent``.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from agent.iteration_budget import UNLIMITED_ITERATIONS, IterationCap

logger = logging.getLogger(__name__)

# Canonical reasons kept for administrative/manual stop alerts. No resource
# threshold in this module raises them any more.
LIMIT_HARD_TIMEOUT = "hard-timeout"
LIMIT_MAX_TURNS = "max-turns"
LIMIT_BUDGET = "budget"
CANONICAL_LIMIT_REASONS = frozenset(
    {LIMIT_HARD_TIMEOUT, LIMIT_MAX_TURNS, LIMIT_BUDGET}
)

ACTION_RESCOPE_PERMITTED = "rescope-permitted"
ACTION_STOP_AND_ALERT = "stop-and-alert"

_DEFAULT_PROFILE_NAME = "standard"
_SENTINEL_NAMES = frozenset({"0", "unlimited", "none", "null", "false"})

#: The complete, fail-closed set of accepted ``autonomous_profile`` labels.
#: Every label persisted by an earlier release is still valid here, so
#: existing jobs need no migration.
PROFILE_LABELS: frozenset[str] = frozenset(
    {
        "light",
        "standard",
        "implementation",
        "large",
        "high-risk-review",
        "retry",
        "experimental",
    }
)


@dataclass(frozen=True)
class AutonomousProfile:
    """A validated routing/audit label for one cron agentic run.

    Intentionally has exactly one field. Any resource-envelope attribute
    here would re-introduce the caps this release removes.
    """

    name: str


class AutonomousProfileError(ValueError):
    """Profile label could not be resolved; fail closed before agent spawn."""


class AutonomousLimitError(RuntimeError):
    """A cron agentic run was stopped by an administrative limit.

    No longer raised for wall-clock, turn, token or USD thresholds — those
    caps were removed in V2.9. Retained as the canonical carrier for a
    manual/administrative stop so the scheduler's delivery and alert paths
    keep a single, parseable error shape (``limit_reason=``,
    ``limit_count=``, ``action=`` are always in the string form).
    """

    def __init__(
        self,
        limit_reason: str,
        *,
        limit_count: int,
        action: str,
        detail: str = "",
    ) -> None:
        if limit_reason not in CANONICAL_LIMIT_REASONS:
            raise ValueError(f"non-canonical limit_reason: {limit_reason!r}")
        self.limit_reason = limit_reason
        self.limit_count = int(limit_count)
        self.action = action
        parts = [
            "Cron autonomous limit reached",
            f"limit_reason={limit_reason}",
            f"limit_count={self.limit_count}",
            f"action={action}",
        ]
        if detail:
            parts.append(detail)
        super().__init__(" ".join(parts))


def validate_profile(profile: AutonomousProfile) -> AutonomousProfile:
    """Fail closed unless *profile* carries a known, non-sentinel label."""
    if not isinstance(profile, AutonomousProfile):
        raise AutonomousProfileError(
            f"invalid autonomous profile type: {type(profile).__name__}"
        )
    name = str(profile.name or "").strip().lower()
    if not name or name in _SENTINEL_NAMES:
        raise AutonomousProfileError(
            f"autonomous profile label rejected (missing/sentinel): {profile.name!r}"
        )
    if name not in PROFILE_LABELS:
        raise AutonomousProfileError(f"unknown autonomous profile: {profile.name!r}")
    return profile


def resolve_autonomous_profile(
    job: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
) -> AutonomousProfile:
    """Resolve the routing/audit label for a cron job.

    Order: ``job['autonomous_profile']`` →
    ``config['cron']['autonomous_limits']['default_profile']`` →
    ``standard``. Unknown / empty / sentinel values fail closed.
    """
    job = job if isinstance(job, Mapping) else {}
    config = config if isinstance(config, Mapping) else {}

    raw_name: Any = job.get("autonomous_profile", None)
    job_explicit = "autonomous_profile" in job

    if raw_name is None or (isinstance(raw_name, str) and not raw_name.strip()):
        if job_explicit and raw_name is not None:
            # Explicit empty/blank on the job is fail-closed, not "use default".
            raise AutonomousProfileError(
                f"unknown or empty autonomous profile: {raw_name!r}"
            )
        cron_cfg = config.get("cron") if isinstance(config.get("cron"), Mapping) else {}
        limits_cfg = (
            cron_cfg.get("autonomous_limits")
            if isinstance(cron_cfg, Mapping)
            and isinstance(cron_cfg.get("autonomous_limits"), Mapping)
            else {}
        )
        raw_name = (
            limits_cfg.get("default_profile")
            if isinstance(limits_cfg, Mapping)
            else None
        ) or _DEFAULT_PROFILE_NAME

    if isinstance(raw_name, (int, float)) and not isinstance(raw_name, bool):
        # Numeric sentinels (0) and bare integers are never valid labels.
        raise AutonomousProfileError(
            f"unknown or invalid autonomous profile: {raw_name!r}"
        )

    name = str(raw_name).strip().lower()
    if not name or name in _SENTINEL_NAMES:
        raise AutonomousProfileError(
            f"unknown or sentinel autonomous profile: {raw_name!r}"
        )
    if name not in PROFILE_LABELS:
        raise AutonomousProfileError(f"unknown autonomous profile: {raw_name!r}")
    return validate_profile(AutonomousProfile(name=name))


def resolve_cron_iteration_cap() -> IterationCap:
    """Return the iteration cap for a cron agentic run: explicitly unlimited.

    Cron deliberately does **not** inherit ``agent.max_turns`` or the 500
    default, and unlimited is never spelled ``0`` or a large integer — see
    :data:`agent.iteration_budget.UNLIMITED_ITERATIONS`.
    """
    return UNLIMITED_ITERATIONS


def run_supervised_agentic_run(
    agent: Any,
    profile: AutonomousProfile,
    *,
    prompt: str,
    task_id: Optional[str] = None,
    poll_interval: float = 1.0,
    heartbeat_fn: Optional[Callable[[], None]] = None,
    context: Optional[contextvars.Context] = None,
    run_fn: Optional[Callable[..., Any]] = None,
) -> dict:
    """Run ``agent.run_conversation`` on a supervised worker thread.

    "Supervised" here means *plumbing*, not policing: the run is executed in
    a worker so the scheduler keeps a Future to hang teardown off, the
    caller's ContextVars follow it, ``task_id`` reaches the tool calls, and
    the one-shot ``run_claim`` heartbeat keeps firing while it works.

    The supervisor never interrupts the agent. It returns whatever
    ``run_conversation`` returned (including failed/incomplete results — the
    scheduler decides what those mean) and propagates worker exceptions
    unchanged. Manual cancellation still works exactly as before: whoever
    cancels calls ``agent.interrupt()`` and the conversation loop unwinds
    itself.
    """
    profile = validate_profile(profile)
    if poll_interval <= 0:
        poll_interval = 0.5

    run_callable = run_fn or agent.run_conversation
    ctx = context if context is not None else contextvars.copy_context()

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        # Mirror scheduler: Context.run(callable, *args, **kwargs).
        if task_id is not None:
            future = pool.submit(ctx.run, run_callable, prompt, task_id=task_id)
        else:
            future = pool.submit(ctx.run, run_callable, prompt)

        # Scheduler teardown waits on this future when present.
        with contextlib.suppress(Exception):
            setattr(agent, "_cron_worker_future", future)

        while True:
            done, _ = concurrent.futures.wait({future}, timeout=poll_interval)
            # The Future is the source of truth for completion; ``wait``'s
            # return value is only a convenience. Consulting both means a
            # wait() that under-reports can never strand this loop. This is
            # NOT a cap: the loop still ends only when the worker is done.
            # (getattr guard mirrors _teardown_cron_agent — test doubles may
            # supply a Future-like object without ``done``.)
            _done_check = getattr(future, "done", None)
            if done or (callable(_done_check) and _done_check()):
                result = future.result()
                break
            if heartbeat_fn is not None:
                with contextlib.suppress(Exception):
                    heartbeat_fn()

        if not isinstance(result, dict):
            raise RuntimeError(
                f"agent.run_conversation returned {type(result).__name__} "
                f"instead of dict: {result!r}"
            )
        return result
    finally:
        # Never cancels the in-flight worker: cancel_futures only drops
        # queued work, and teardown waits on the Future above.
        pool.shutdown(wait=False, cancel_futures=True)


__all__ = [
    "ACTION_RESCOPE_PERMITTED",
    "ACTION_STOP_AND_ALERT",
    "CANONICAL_LIMIT_REASONS",
    "LIMIT_BUDGET",
    "LIMIT_HARD_TIMEOUT",
    "LIMIT_MAX_TURNS",
    "PROFILE_LABELS",
    "AutonomousLimitError",
    "AutonomousProfile",
    "AutonomousProfileError",
    "resolve_autonomous_profile",
    "resolve_cron_iteration_cap",
    "run_supervised_agentic_run",
    "validate_profile",
]
