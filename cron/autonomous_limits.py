"""Strict autonomous limits for non-no_agent cron agentic runs.

Every agentic cron invocation in ``scheduler.run_job`` must resolve a fixed
positive profile and run the agent under this supervisor. The supervisor
enforces wall-clock timeout, max turns/API calls, processed tokens, and
metered USD spend. On threshold it interrupts the agent with a canonical
reason and records a deterministic one-shot rescope decision.

``no_agent`` script-only jobs never enter this module — they do not spawn
``AIAgent``.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Canonical interrupt / limit reasons emitted by the supervisor.
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
_STATE_SCHEMA_VERSION = 1
_PROCESS_LOCK = threading.RLock()


@dataclass(frozen=True)
class AutonomousProfile:
    """Fixed positive resource envelope for one cron agentic run."""

    name: str
    timeout_seconds: int
    max_turns: int
    token_budget: int
    usd_budget: float


# Fixed positive profiles — exact values required by the contract.
PROFILES: dict[str, AutonomousProfile] = {
    "light": AutonomousProfile(
        name="light",
        timeout_seconds=300,
        max_turns=6,
        token_budget=200_000,
        usd_budget=0.50,
    ),
    "standard": AutonomousProfile(
        name="standard",
        timeout_seconds=600,
        max_turns=16,
        token_budget=1_200_000,
        usd_budget=2,
    ),
    "implementation": AutonomousProfile(
        name="implementation",
        timeout_seconds=1200,
        max_turns=30,
        token_budget=2_000_000,
        usd_budget=5,
    ),
    "large": AutonomousProfile(
        name="large",
        timeout_seconds=1800,
        max_turns=45,
        token_budget=4_000_000,
        usd_budget=10,
    ),
    "high-risk-review": AutonomousProfile(
        name="high-risk-review",
        timeout_seconds=900,
        max_turns=24,
        token_budget=1_500_000,
        usd_budget=8,
    ),
    "retry": AutonomousProfile(
        name="retry",
        timeout_seconds=600,
        max_turns=16,
        token_budget=1_000_000,
        usd_budget=2,
    ),
}


class AutonomousProfileError(ValueError):
    """Profile could not be resolved; fail closed before agent spawn."""


class AutonomousLimitError(RuntimeError):
    """An autonomous limit was reached during a supervised cron run.

    ``limit_reason`` is one of the canonical reasons. ``action`` is either
    ``rescope-permitted`` (first hit for the task key) or ``stop-and-alert``
    (second hit). The string form always includes ``limit_reason=``,
    ``limit_count=``, and ``action=`` so scheduler delivery/error paths
    surface them without parsing structured fields.
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
    """Fail closed on missing/nonpositive fields or sentinel 0/unlimited."""
    if not isinstance(profile, AutonomousProfile):
        raise AutonomousProfileError(
            f"invalid autonomous profile type: {type(profile).__name__}"
        )
    name = str(profile.name or "").strip().lower()
    if not name or name in _SENTINEL_NAMES:
        raise AutonomousProfileError(
            f"autonomous profile name rejected (missing/sentinel): {profile.name!r}"
        )
    for field_name, value in (
        ("timeout_seconds", profile.timeout_seconds),
        ("max_turns", profile.max_turns),
        ("token_budget", profile.token_budget),
        ("usd_budget", profile.usd_budget),
    ):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise AutonomousProfileError(
                f"autonomous profile {name!r} has non-numeric {field_name}"
            ) from exc
        if numeric <= 0:
            raise AutonomousProfileError(
                f"autonomous profile {name!r} has nonpositive {field_name}="
                f"{value!r}; 0/unlimited is never accepted in cron mode"
            )
    return profile


def resolve_autonomous_profile(
    job: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
) -> AutonomousProfile:
    """Resolve a fixed positive profile for a cron job.

    Order: ``job['autonomous_profile']`` →
    ``config['cron']['autonomous_limits']['default_profile']`` →
    ``standard``. Unknown / empty / sentinel / nonpositive values fail closed.
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
        # Numeric sentinels (0) and bare integers are never valid profile names.
        raise AutonomousProfileError(
            f"unknown or invalid autonomous profile: {raw_name!r}"
        )

    name = str(raw_name).strip().lower()
    if not name or name in _SENTINEL_NAMES:
        raise AutonomousProfileError(
            f"unknown or sentinel autonomous profile: {raw_name!r}"
        )
    profile = PROFILES.get(name)
    if profile is None:
        raise AutonomousProfileError(f"unknown autonomous profile: {raw_name!r}")
    return validate_profile(profile)


def default_state_path(hermes_home: Optional[Path] = None) -> Path:
    """Profile-scoped durable state for the one-shot rescope guard."""
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    return Path(home) / "cron" / "autonomous_limits_state.json"


def _safe_task_key(task_key: str) -> str:
    raw = str(task_key or "").strip() or "unknown"
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw)[:128] or "unknown"


def _empty_state() -> dict[str, Any]:
    return {"schema_version": _STATE_SCHEMA_VERSION, "tasks": {}}


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_state()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_state()
    if not isinstance(raw, dict):
        return _empty_state()
    tasks = raw.get("tasks")
    if not isinstance(tasks, dict):
        tasks = {}
    return {"schema_version": _STATE_SCHEMA_VERSION, "tasks": tasks}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(path.parent, 0o700)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".autonomous-limits-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            handle.flush()
            os.fsync(handle.fileno())
        with contextlib.suppress(OSError):
            os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _with_state_lock(path: Path, mutator: Callable[[dict[str, Any]], Any]) -> Any:
    """Apply *mutator* under process + file locks; persist atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _PROCESS_LOCK:
        lock_path.touch(exist_ok=True)
        with open(lock_path, "a+", encoding="utf-8") as lock_fh:
            if fcntl is not None:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                state = _read_state(path)
                result = mutator(state)
                _atomic_json(path, state)
                return result
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def record_limit_hit(
    task_key: str,
    limit_reason: str,
    *,
    state_path: Optional[Path] = None,
) -> tuple[int, str]:
    """Record a limit hit for *task_key*.

    Returns ``(limit_count, action)`` where the first hit permits one rescope
    and the second (and later) hits are ``stop-and-alert``. Does **not**
    auto-run a retry — only decides/records.
    """
    key = _safe_task_key(task_key)
    path = Path(state_path) if state_path is not None else default_state_path()

    def _mutate(state: dict[str, Any]) -> tuple[int, str]:
        tasks = state.setdefault("tasks", {})
        entry = tasks.get(key) if isinstance(tasks.get(key), dict) else {}
        prev = int(entry.get("limit_count") or 0)
        # Cap at 2: first hit permits one rescope; second is stop-and-alert.
        count = min(prev + 1, 2)
        action = (
            ACTION_STOP_AND_ALERT if count >= 2 else ACTION_RESCOPE_PERMITTED
        )
        tasks[key] = {
            "limit_count": count,
            "last_reason": limit_reason,
            "action": action,
            "updated_at": time.time(),
        }
        return count, action

    return _with_state_lock(path, _mutate)


def clear_limit_state(
    task_key: str,
    *,
    state_path: Optional[Path] = None,
) -> None:
    """Clear limit count after a successful completion."""
    key = _safe_task_key(task_key)
    path = Path(state_path) if state_path is not None else default_state_path()

    def _mutate(state: dict[str, Any]) -> None:
        tasks = state.setdefault("tasks", {})
        tasks.pop(key, None)

    _with_state_lock(path, _mutate)


def _api_call_count(agent: Any) -> int:
    if hasattr(agent, "get_activity_summary"):
        try:
            summary = agent.get_activity_summary() or {}
            if "api_call_count" in summary:
                return int(summary.get("api_call_count") or 0)
        except Exception:
            pass
    for attr in ("_api_call_count", "session_api_calls"):
        val = getattr(agent, attr, None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return 0


def _session_tokens(agent: Any) -> int:
    try:
        return int(getattr(agent, "session_total_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _session_cost_usd(agent: Any) -> float:
    try:
        return float(getattr(agent, "session_estimated_cost_usd", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def evaluate_thresholds(
    agent: Any,
    profile: AutonomousProfile,
    *,
    started_monotonic: float,
    now_monotonic: Optional[float] = None,
) -> Optional[str]:
    """Return a canonical limit_reason if any threshold is crossed, else None.

    Check order is deterministic: hard-timeout → max-turns → budget.
    Token or USD threshold both map to ``budget``.
    """
    now = time.monotonic() if now_monotonic is None else now_monotonic
    elapsed = now - started_monotonic
    if elapsed >= float(profile.timeout_seconds):
        return LIMIT_HARD_TIMEOUT
    if _api_call_count(agent) >= int(profile.max_turns):
        return LIMIT_MAX_TURNS
    if _session_tokens(agent) >= int(profile.token_budget):
        return LIMIT_BUDGET
    if _session_cost_usd(agent) >= float(profile.usd_budget):
        return LIMIT_BUDGET
    return None


def is_max_iterations_reached(result: Any) -> bool:
    """True when run_conversation exited because max iterations were hit."""
    if not isinstance(result, Mapping):
        return False
    reason = str(result.get("turn_exit_reason") or "")
    if reason.startswith("max_iterations_reached"):
        return True
    # Some paths only set completed=False without a precise reason string.
    if (
        result.get("completed") is False
        and result.get("failed") is not True
        and "max_iterations" in reason
    ):
        return True
    return False


def _raise_limit(
    limit_reason: str,
    *,
    task_key: str,
    state_path: Optional[Path],
    detail: str = "",
) -> None:
    count, action = record_limit_hit(
        task_key, limit_reason, state_path=state_path
    )
    raise AutonomousLimitError(
        limit_reason,
        limit_count=count,
        action=action,
        detail=detail,
    )


def run_with_autonomous_limits(
    agent: Any,
    profile: AutonomousProfile,
    *,
    prompt: str,
    task_key: str,
    task_id: Optional[str] = None,
    poll_interval: float = 1.0,
    state_path: Optional[Path] = None,
    heartbeat_fn: Optional[Callable[[], None]] = None,
    inactivity_limit: Optional[float] = None,
    context: Optional[contextvars.Context] = None,
    run_fn: Optional[Callable[..., Any]] = None,
) -> dict:
    """Run ``agent.run_conversation`` under the autonomous limit supervisor.

    On any hard limit (including ``max_iterations_reached`` from the agent)
    raises :class:`AutonomousLimitError` with ``limit_reason`` set. Successful
    completion clears the per-task limit count so a later failure again
    permits one rescope.
    """
    profile = validate_profile(profile)
    if poll_interval <= 0:
        poll_interval = 0.5

    path = Path(state_path) if state_path is not None else default_state_path()
    run_callable = run_fn or agent.run_conversation
    ctx = context if context is not None else contextvars.copy_context()

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    started = time.monotonic()
    try:
        # Mirror scheduler: Context.run(callable, *args, **kwargs).
        if task_id is not None:
            future = pool.submit(ctx.run, run_callable, prompt, task_id=task_id)
        else:
            future = pool.submit(ctx.run, run_callable, prompt)

        # Scheduler teardown waits on this future when present.
        with contextlib.suppress(Exception):
            setattr(agent, "_cron_worker_future", future)

        result: Any = None
        inactivity_timeout = False
        while True:
            done, _ = concurrent.futures.wait({future}, timeout=poll_interval)
            if done:
                result = future.result()
                break

            if heartbeat_fn is not None:
                with contextlib.suppress(Exception):
                    heartbeat_fn()

            reason = evaluate_thresholds(agent, profile, started_monotonic=started)
            if reason is not None:
                if hasattr(agent, "interrupt"):
                    with contextlib.suppress(Exception):
                        agent.interrupt(reason)
                _raise_limit(
                    reason,
                    task_key=task_key,
                    state_path=path,
                    detail=f"profile={profile.name}",
                )

            if inactivity_limit is not None and inactivity_limit > 0:
                idle_secs = 0.0
                if hasattr(agent, "get_activity_summary"):
                    try:
                        act = agent.get_activity_summary() or {}
                        idle_secs = float(act.get("seconds_since_activity", 0.0) or 0.0)
                    except Exception:
                        idle_secs = 0.0
                if idle_secs >= inactivity_limit:
                    inactivity_timeout = True
                    break

        if inactivity_timeout:
            activity: dict = {}
            if hasattr(agent, "get_activity_summary"):
                with contextlib.suppress(Exception):
                    activity = agent.get_activity_summary() or {}
            last_desc = activity.get("last_activity_desc", "unknown")
            secs_ago = activity.get("seconds_since_activity", 0)
            if hasattr(agent, "interrupt"):
                with contextlib.suppress(Exception):
                    agent.interrupt("Cron job timed out (inactivity)")
            raise TimeoutError(
                f"Cron job idle for {int(secs_ago)}s "
                f"(limit {int(inactivity_limit)}s) — last activity: {last_desc}"
            )

        if is_max_iterations_reached(result):
            # Agent self-stopped at max_iterations; treat as max-turns failure,
            # never as successful fallback delivery.
            _raise_limit(
                LIMIT_MAX_TURNS,
                task_key=task_key,
                state_path=path,
                detail=f"profile={profile.name} max_iterations_reached",
            )

        if not isinstance(result, dict):
            raise RuntimeError(
                f"agent.run_conversation returned {type(result).__name__} "
                f"instead of dict: {result!r}"
            )

        # Only a verified successful completion resets the rescope counter.
        # Failed/incomplete dict results are handled by scheduler.run_job and
        # must not erase a previous limit hit.
        if result.get("completed") is True and result.get("failed") is not True:
            clear_limit_state(task_key, state_path=path)
        return result
    except AutonomousLimitError:
        raise
    except TimeoutError:
        raise
    except Exception:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
