"""Reliable one-shot process control, preserving the vpsclone durable protocol."""
from __future__ import annotations
import logging
import os
import platform
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from tools.process_registry import ProcessSession
logger = logging.getLogger("tools.process_registry")
_IS_WINDOWS = platform.system() == "Windows"
RELIABLE_CONTROL_WATCH_PATTERN = "FABLE_WAKE"
RELIABLE_CONTROL_CLOSE_PATTERN = "FABLE_AUTO_CLOSE"
class ProcessCheckpointRecoveryError(RuntimeError):
    """Checkpoint state is unreadable, so gateway startup must fail closed."""


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_reliable_watch_event(
    event: Dict[str, Any],
    *,
    index: Optional[int] = None,
) -> None:
    """Validate the closed durable-outbox protocol before it affects state."""
    label = (
        f"process notification outbox entry {index}"
        if index is not None
        else "process notification event"
    )
    if not isinstance(event, dict):
        raise ValueError(f"{label} is not an object")
    if event.get("type") != "watch_match":
        raise ValueError(f"{label} has invalid type")
    if event.get("pattern") != RELIABLE_CONTROL_WATCH_PATTERN:
        raise ValueError(f"{label} has invalid control pattern")
    for field_name in ("delivery_id", "session_id", "platform"):
        if not _is_nonempty_string(event.get(field_name)):
            raise ValueError(f"{label} has invalid {field_name}")

    route_fields = ("chat_id", "session_key", "origin_session_id")
    for field_name in route_fields:
        value = event.get(field_name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{label} has non-string {field_name}")
    if not any(_is_nonempty_string(event.get(name)) for name in route_fields):
        raise ValueError(f"{label} has no durable delivery route")

    started_at = event.get("started_at")
    if (
        isinstance(started_at, bool)
        or not isinstance(started_at, (int, float))
        or not started_at > 0
    ):
        raise ValueError(f"{label} has invalid started_at")
    output = event.get("output")
    if (
        not isinstance(output, str)
        or RELIABLE_CONTROL_WATCH_PATTERN not in output
    ):
        raise ValueError(f"{label} has invalid control output")
    if not isinstance(event.get("command"), str):
        raise ValueError(f"{label} has invalid command")
    for field_name in (
        "task_id",
        "user_id",
        "user_name",
        "thread_id",
        "message_id",
        "parent_session_id",
        "termination_source",
        "control_reason",
    ):
        value = event.get(field_name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{label} has non-string {field_name}")


def _validate_reliable_watch_outbox(entries: List[Dict[str, Any]]) -> None:
    seen_delivery_ids: set[str] = set()
    for index, event in enumerate(entries):
        _validate_reliable_watch_event(event, index=index)
        delivery_id = event["delivery_id"]
        if delivery_id in seen_delivery_ids:
            raise ValueError(
                "process notification outbox contains duplicate delivery_id "
                f"{delivery_id!r}"
            )
        seen_delivery_ids.add(delivery_id)


def _validate_process_checkpoint(entries: List[Dict[str, Any]]) -> None:
    """Validate recovery rows while accepting safe pre-feature checkpoints.

    Older generic rows did not carry the newer Fable booleans, owner id, or
    pid-scope metadata, so those fields remain optional. A real process row
    still needs a positive PID. Only the explicit consumed-control tombstone
    may omit it because that row represents a suppression verdict, not a live
    process to adopt.
    """
    seen_session_ids: set[str] = set()
    boolean_fields = (
        "notify_on_complete",
        "notify_on_failure",
        "reliable_control_watch_delivered",
        "reliable_control_close_seen",
        "reliable_control_delivery_consumed",
    )
    string_fields = (
        "checkpoint_owner_id",
        "command",
        "pid_scope",
        "systemd_unit",
        "task_id",
        "session_key",
        "watcher_platform",
        "watcher_chat_id",
        "watcher_user_id",
        "watcher_user_name",
        "watcher_thread_id",
        "watcher_message_id",
        "parent_session_id",
    )
    reliable_state_fields = (
        "reliable_control_watch_delivered",
        "reliable_control_close_seen",
        "reliable_control_delivery_consumed",
    )

    for index, entry in enumerate(entries):
        label = f"process checkpoint entry {index}"
        if not isinstance(entry, dict):
            raise ValueError(f"{label} is not an object")
        session_id = entry.get("session_id")
        if not _is_nonempty_string(session_id):
            raise ValueError(f"{label} has invalid session_id")
        if session_id in seen_session_ids:
            raise ValueError(
                f"process checkpoint contains duplicate session_id {session_id!r}"
            )
        seen_session_ids.add(session_id)

        for field_name in boolean_fields:
            if field_name in entry and not isinstance(entry[field_name], bool):
                raise ValueError(f"{label} has non-boolean {field_name}")
        for field_name in string_fields:
            value = entry.get(field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{label} has non-string {field_name}")

        pid_scope = entry.get("pid_scope", "host")
        if pid_scope not in {"host", "sandbox"}:
            raise ValueError(f"{label} has invalid pid_scope")
        consumed_tombstone = entry.get(
            "reliable_control_delivery_consumed", False
        ) is True
        pid = entry.get("pid")
        if not consumed_tombstone:
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                raise ValueError(
                    f"{label} has no positive pid and is not a consumed tombstone"
                )
        elif pid is not None and (
            isinstance(pid, bool) or not isinstance(pid, int) or pid < 0
        ):
            raise ValueError(f"{label} has invalid tombstone pid")

        host_start_time = entry.get("host_start_time")
        if host_start_time is not None and (
            isinstance(host_start_time, bool)
            or not isinstance(host_start_time, (int, float))
            or host_start_time < 0
        ):
            raise ValueError(f"{label} has invalid host_start_time")
        started_at = entry.get("started_at")
        if started_at is not None and (
            isinstance(started_at, bool)
            or not isinstance(started_at, (int, float))
            or not started_at > 0
        ):
            raise ValueError(f"{label} has invalid started_at")
        watcher_interval = entry.get("watcher_interval")
        if watcher_interval is not None and (
            isinstance(watcher_interval, bool)
            or not isinstance(watcher_interval, int)
            or watcher_interval < 0
        ):
            raise ValueError(f"{label} has invalid watcher_interval")

        watch_patterns = entry.get("watch_patterns", [])
        if (
            not isinstance(watch_patterns, list)
            or any(
                not _is_nonempty_string(pattern)
                for pattern in watch_patterns
            )
        ):
            raise ValueError(f"{label} has invalid watch_patterns")
        reliable_protocol = watch_patterns == [RELIABLE_CONTROL_WATCH_PATTERN]
        if consumed_tombstone and not reliable_protocol:
            raise ValueError(
                f"{label} consumed tombstone is not a Fable control row"
            )
        if any(entry.get(name) is True for name in reliable_state_fields) and not reliable_protocol:
            raise ValueError(
                f"{label} has Fable control state without the exact pattern"
            )
        watcher_platform = entry.get("watcher_platform", "")
        if (
            reliable_protocol
            and _is_nonempty_string(watcher_platform)
            and not consumed_tombstone
            and not (
                _is_nonempty_string(entry.get("watcher_chat_id"))
                or _is_nonempty_string(entry.get("session_key"))
            )
        ):
            raise ValueError(f"{label} has no durable Fable delivery route")
        if (
            entry.get("reliable_control_watch_delivered") is True
            and not consumed_tombstone
            and not _is_nonempty_string(watcher_platform)
        ):
            raise ValueError(
                f"{label} marks a durable Fable wake without a platform"
            )


class _CheckpointFileLock:
    """Short cross-process lock for the shared process checkpoint."""

    def __init__(self, path: Path):
        self.path = path
        self._handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except Exception:
            self._handle.close()
            self._handle = None
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


@dataclass(frozen=True)
class ProcessNotificationConfig:
    """Immutable notification/routing data installed before output readers run.

    ``terminal_tool`` resolves async-delivery capability and flag conflicts
    before launching the child, then passes this value into the registry.  A
    frozen value prevents the old post-spawn mutation race from returning in a
    new call site.
    """

    watcher_platform: str = ""
    watcher_chat_id: str = ""
    watcher_user_id: str = ""
    watcher_user_name: str = ""
    watcher_thread_id: str = ""
    watcher_message_id: str = ""
    watcher_interval: int = 0
    parent_session_id: str = ""
    notify_on_complete: bool = False
    notify_on_failure: bool = False
    watch_patterns: tuple[str, ...] = ()


def process_completion_failed(session: ProcessSession) -> bool:
    """Return whether a terminal session ended abnormally.

    ``None`` remains non-failure for compatibility with detached processes
    whose exact status is unavailable, except when the registry explicitly
    classified the backend as lost/failed.
    """
    if getattr(session, "completion_reason", "exited") in {"lost", "failed_start"}:
        return True
    return getattr(session, "exit_code", None) not in {0, None}


def should_notify_process_completion(
    session: ProcessSession,
    *,
    notify_on_complete: Optional[bool] = None,
    notify_on_failure: Optional[bool] = None,
) -> bool:
    """Apply full-completion vs failure-only notification semantics."""
    reliable_control_protocol = (
        getattr(session, "watch_patterns", []) == [RELIABLE_CONTROL_WATCH_PATTERN]
    )
    reliable_wake_delivered = getattr(
        session, "_reliable_control_watch_delivered", False
    )
    reliable_close_seen = getattr(session, "_reliable_control_close_seen", False)
    complete_enabled = (
        bool(getattr(session, "notify_on_complete", False))
        if notify_on_complete is None
        else bool(notify_on_complete)
    )
    failure_enabled = (
        bool(getattr(session, "notify_on_failure", False))
        if notify_on_failure is None
        else bool(notify_on_failure)
    )
    if complete_enabled:
        return True
    if not failure_enabled:
        return False
    if reliable_control_protocol:
        # The wake marker is the primary notification. Completion is strictly
        # a fallback, even when the watcher exits non-zero immediately after
        # printing it. AUTO_CLOSE suppresses only a clean exit; an abnormal
        # exit after that sentinel is still noteworthy.
        if reliable_wake_delivered:
            return False
        if process_completion_failed(session):
            return True
        return not reliable_close_seen
    return process_completion_failed(session)


class ProcessControlMixin:
    def _check_watch_patterns(self, session: ProcessSession, new_text: str) -> None:
        """Scan new output for watch patterns and queue notifications.

        Called from reader threads with new_text being the freshly-read chunk.

        Per-session rate limit: at most ONE watch-match notification per
        WATCH_MIN_INTERVAL_SECONDS. Any match arriving inside the cooldown
        window is dropped and counts as ONE strike for that window. After
        WATCH_STRIKE_LIMIT consecutive strike windows, watch_patterns is
        disabled for this session and the session is promoted to
        notify_on_complete semantics — one notification when the process
        actually exits, no more mid-process spam.

        Independently, WATCH_LIFETIME_MAX_HITS caps the total number of
        matches ever delivered for a session, so a pattern that keeps
        recurring at a cadence just above the cooldown (e.g. a service
        restarted repeatedly over a day) still gets disabled instead of
        forcing a full-context agent turn indefinitely.
        """
        from tools.process_registry import WATCH_MIN_INTERVAL_SECONDS, WATCH_STRIKE_LIMIT, WATCH_LIFETIME_MAX_HITS, _redact_process_result
        if not session.watch_patterns or session._watch_disabled:
            return
        # Suppress-after-exit: once the reader loop has declared the process
        # exited, any late chunk we still see is post-exit noise. Dropping these
        # prevents the "stale notifications delivered minutes after the process
        # ended" spam when completion_queue consumers run async.
        if session.exited:
            return

        # Scan with a bounded overlap so a sentinel split across reader chunks
        # (e.g. ``FABLE_`` + ``WAKE``) is still recognized exactly once.
        control_protocol = session.watch_patterns == [RELIABLE_CONTROL_WATCH_PATTERN]
        scan_patterns = list(session.watch_patterns)
        if control_protocol:
            scan_patterns.append(RELIABLE_CONTROL_CLOSE_PATTERN)
        max_pattern_len = max((len(pattern) for pattern in scan_patterns), default=1)
        close_state_changed = False
        with session._lock:
            carry = session._watch_scan_carry
            scan_text = carry + new_text
            carry_len = len(carry)
            session._watch_scan_carry = (
                scan_text[-(max_pattern_len - 1):]
                if max_pattern_len > 1
                else ""
            )

            def _contains_new_occurrence(pattern: str) -> bool:
                start = 0
                while True:
                    pos = scan_text.find(pattern, start)
                    if pos < 0:
                        return False
                    occurrence_end = pos + max(1, len(pattern))
                    if occurrence_end > carry_len:
                        return True
                    start = pos + 1

            if (
                control_protocol
                and _contains_new_occurrence(RELIABLE_CONTROL_CLOSE_PATTERN)
                and not session._reliable_control_close_seen
            ):
                session._reliable_control_close_seen = True
                close_state_changed = True

        with self._lock:
            registered_session = self._running.get(session.id) is session
        if close_state_changed and registered_session:
            # Persist the terminal protocol state before the child can exit;
            # recovery can then distinguish a deliberate autonomous close
            # from a watcher that vanished without any sentinel.
            self._write_checkpoint()

        # Preserve the historical line-shaped notification payload while only
        # accepting occurrences that include at least one byte from this chunk.
        matched_lines = []
        matched_pattern = None
        line_offset = 0
        line_parts = scan_text.splitlines(keepends=True)
        if scan_text and not line_parts:
            line_parts = [scan_text]
        for raw_line in line_parts:
            line = raw_line.rstrip("\r\n")
            for pat in session.watch_patterns:
                search_from = 0
                matched_in_new_data = False
                while True:
                    pos = line.find(pat, search_from)
                    if pos < 0:
                        break
                    occurrence_end = line_offset + pos + max(1, len(pat))
                    if occurrence_end > carry_len:
                        matched_in_new_data = True
                        break
                    search_from = pos + 1
                if matched_in_new_data:
                    matched_lines.append(line.rstrip())
                    if matched_pattern is None:
                        matched_pattern = pat
                    break  # one match per line is enough
            line_offset += len(raw_line)

        if not matched_lines:
            return

        now = time.time()
        should_disable = False
        lifetime_exhausted = False
        with session._lock:
            # A kill/exit can race after the initial fast-path check but
            # before this state transition.  The completion path reserves its
            # fallback wake under the same lock; never emit a late marker on
            # top of that reserved one-shot event.
            if session.exited or session._watch_disabled:
                return
            # Case 1: still inside the cooldown from the last emission.
            # Count this as a strike for the current window (only once per window)
            # and drop the event. If we've hit the strike limit, disable watch
            # and promote to notify_on_complete.
            if session._watch_cooldown_until and now < session._watch_cooldown_until:
                session._watch_suppressed += len(matched_lines)
                if not session._watch_strike_candidate:
                    # First drop in this window — count one strike.
                    session._watch_strike_candidate = True
                    session._watch_consecutive_strikes += 1
                    if session._watch_consecutive_strikes >= WATCH_STRIKE_LIMIT:
                        session._watch_disabled = True
                        # Promote to notify_on_complete so the agent still gets
                        # exactly one notification when the process actually ends.
                        session.notify_on_complete = True
                        should_disable = True
                return_early = True
            else:
                # Case 2: cooldown has expired.
                # Decide whether this window was a "clean" one (no drops) or a
                # strike window. If no strike candidate was set during the prior
                # cooldown, reset the consecutive-strike counter — we're back to
                # healthy emission cadence.
                if (
                    session._watch_cooldown_until
                    and not session._watch_strike_candidate
                ):
                    session._watch_consecutive_strikes = 0
                session._watch_strike_candidate = False

                # Emit the notification and start a new cooldown window.
                session._watch_last_emit_at = now
                session._watch_cooldown_until = now + WATCH_MIN_INTERVAL_SECONDS
                session._watch_hits += 1
                suppressed = session._watch_suppressed
                session._watch_suppressed = 0
                return_early = False
                # Lifetime cap: this match is delivered (it already earned it),
                # but disable further ones regardless of how cleanly spaced
                # they are — see WATCH_LIFETIME_MAX_HITS above.
                lifetime_exhausted = session._watch_hits >= WATCH_LIFETIME_MAX_HITS
                if lifetime_exhausted:
                    session._watch_disabled = True
                    session.notify_on_complete = True

                reliable_control_match = (
                    matched_pattern == RELIABLE_CONTROL_WATCH_PATTERN
                    and session.watch_patterns == [RELIABLE_CONTROL_WATCH_PATTERN]
                    and session._watch_hits == 1
                    and not session._reliable_control_watch_delivered
                )
                if reliable_control_match:
                    # This is a protocol control edge, not a recurring log
                    # subscription. Guarantee it once, then close the watch so
                    # it cannot be used to bypass flood protection repeatedly.
                    session._reliable_control_watch_delivered = True
                    session._watch_disabled = True

        if return_early:
            if should_disable:
                # Emit exactly one "watch disabled, falling back to notify_on_complete"
                # summary event so the agent/user sees why things went quiet.
                self.completion_queue.put({
                    "session_id": session.id,
                    "session_key": session.session_key,
                    "task_id": session.task_id,
                    "owner_task_id": session.owner_task_id or session.task_id,
                    "command": session.command,
                    "type": "watch_disabled",
                    "suppressed": session._watch_suppressed,
                    "platform": session.watcher_platform,
                    "chat_id": session.watcher_chat_id,
                    "user_id": session.watcher_user_id,
                    "user_name": session.watcher_user_name,
                    "thread_id": session.watcher_thread_id,
                    "message_id": session.watcher_message_id,
                    "parent_session_id": session.parent_session_id,
                    "message": (
                        f"Watch patterns disabled for process {session.id} — "
                        f"{WATCH_STRIKE_LIMIT} consecutive rate-limit windows triggered "
                        f"(min spacing {WATCH_MIN_INTERVAL_SECONDS}s). "
                        f"Falling back to notify_on_complete semantics; you'll get "
                        f"exactly one notification when the process exits."
                    ),
                })
            return

        # Trim matched output to a reasonable size
        output = "\n".join(matched_lines[:20])
        if len(output) > 2000:
            output = output[:2000] + "\n...(truncated)"

        # Global circuit breaker — across all sessions (secondary safety net).
        if not reliable_control_match and not self._global_watch_admit(now):
            if lifetime_exhausted:
                # The final match was dropped by the global breaker, but the
                # session is already disabled — still tell the user why things
                # went quiet (the strike path emits its summary unconditionally
                # too).
                self._emit_lifetime_watch_disabled(session)
            return

        notification = {
            "session_id": session.id,
            "session_key": session.session_key,
            "task_id": session.task_id,
            "owner_task_id": session.owner_task_id or session.task_id,
            "command": session.command,
            "type": "watch_match",
            "pattern": matched_pattern,
            "output": output,
            "suppressed": suppressed,
            "platform": session.watcher_platform,
            "chat_id": session.watcher_chat_id,
            "user_id": session.watcher_user_id,
            "user_name": session.watcher_user_name,
            "thread_id": session.watcher_thread_id,
            "message_id": session.watcher_message_id,
            "parent_session_id": session.parent_session_id,
            "started_at": session.started_at,
        }
        _redact_process_result(notification)
        if (
            reliable_control_match
            and registered_session
            and session.watcher_platform
        ):
            notification["delivery_id"] = (
                f"watch:{session.id}:{session.started_at:.9f}"
            )
            # Linearize primary-marker persistence with explicit wait/kill
            # consumption.  The match was reserved under this same lock above,
            # but building/redacting the event happens outside it.  Re-check the
            # monotonic consume fence *before* disk I/O and keep the lock through
            # the outbox/checkpoint transaction so a consumer cannot observe an
            # empty outbox, remove its tombstone, and then lose a late producer
            # write in a crash.
            with session._lock:
                consumed_before_persist = self.is_completion_consumed(
                    session.id
                )
                if consumed_before_persist:
                    outbox_persisted = False
                    checkpoint_written = False
                else:
                    outbox_persisted = self._persist_reliable_watch_event(
                        notification
                    )
                    if outbox_persisted:
                        session._reliable_control_watch_persisted = True
                        checkpoint_written = self._write_checkpoint(
                            remove_session_ids={session.id}
                        )
                    else:
                        checkpoint_written = self._write_checkpoint(
                            extra_entries=[
                                self._checkpoint_entry_for_session(session)
                            ]
                        )
            if consumed_before_persist:
                self._durably_consume_reliable_control(session)
                return
            notification["checkpoint_confirmed"] = bool(
                outbox_persisted and checkpoint_written
            )
            if self.is_completion_consumed(session.id):
                # An explicit kill/wait may have consumed the process while the
                # marker transaction completed but before RAM publication.
                # Remove the durable side and never publish a stale wake.
                self._durably_consume_reliable_control(session)
                return
        self.completion_queue.put(notification)

        if lifetime_exhausted:
            # Same "why things went quiet" summary as the strike-limit path,
            # queued right after the final delivered match.
            self._emit_lifetime_watch_disabled(session)

    def _emit_lifetime_watch_disabled(self, session: ProcessSession) -> None:
        """Queue the watch_disabled summary for the lifetime-cap path (#93513)."""
        from tools.process_registry import WATCH_LIFETIME_MAX_HITS
        self.completion_queue.put({
            "session_id": session.id,
            "session_key": session.session_key,
            "task_id": session.task_id,
            "owner_task_id": session.owner_task_id or session.task_id,
            "command": session.command,
            "type": "watch_disabled",
            "suppressed": 0,
            "platform": session.watcher_platform,
            "chat_id": session.watcher_chat_id,
            "user_id": session.watcher_user_id,
            "user_name": session.watcher_user_name,
            "thread_id": session.watcher_thread_id,
            "message_id": session.watcher_message_id,
            "parent_session_id": session.parent_session_id,
            "message": (
                f"Watch patterns disabled for process {session.id} — "
                f"reached the lifetime cap of {WATCH_LIFETIME_MAX_HITS} delivered "
                f"matches. Falling back to notify_on_complete semantics; you'll get "
                f"exactly one notification when the process exits."
            ),
        })

    def _refresh_detached_session(
        self,
        session: Optional[ProcessSession],
        *,
        consume_output: bool = False,
    ) -> Optional[ProcessSession]:
        """Update a recovered host session through the finalization state machine."""
        from tools.process_registry import _stop_systemd_unit
        if session is None:
            return session
        if session.exited:
            if consume_output:
                self._consume_completion_result(session)
            return session
        if not session.detached or session.pid_scope != "host":
            return session

        # Identity-aware liveness: a recycled PID (alive but a different process
        # than we spawned) must be treated as "our process exited", so it is
        # moved to finished and can never be tree-killed by a later kill().
        if self._host_pid_is_ours(session.pid, session.host_start_time):
            return session

        recovered_systemd_unit = session.systemd_unit
        scope_reap_failed = bool(
            recovered_systemd_unit
            and not _stop_systemd_unit(recovered_systemd_unit)
        )
        already_exited = False
        with session._lock:
            if session.exited:
                already_exited = True
            else:
                session.exited = True
                # Recovered sessions no longer have a waitable handle, so the
                # real exit code is unavailable after the process object is gone.
                session.exit_code = None
                session.completion_reason = "lost"
                session.termination_source = (
                    "checkpoint_scope_reap_failed"
                    if scope_reap_failed
                    else "checkpoint_process_lost"
                )
                session._retain_checkpoint_for_scope_reap = scope_reap_failed
                if recovered_systemd_unit and not scope_reap_failed:
                    # A successful stop discharged the persisted scope cleanup
                    # obligation; avoid a second stop in kill_process's
                    # already-exited branch and omit it from future snapshots.
                    session.systemd_unit = ""

        if already_exited:
            if consume_output:
                self._consume_completion_result(session)
            return session
        self._move_to_finished(session, consume_output=consume_output)
        return session

    @staticmethod
    def _notification_session_kwargs(
        notification: Optional[ProcessNotificationConfig],
    ) -> Dict[str, Any]:
        """Convert immutable spawn configuration into session constructor data."""
        config = notification or ProcessNotificationConfig()
        return {
            "watcher_platform": config.watcher_platform,
            "watcher_chat_id": config.watcher_chat_id,
            "watcher_user_id": config.watcher_user_id,
            "watcher_user_name": config.watcher_user_name,
            "watcher_thread_id": config.watcher_thread_id,
            "watcher_message_id": config.watcher_message_id,
            "watcher_interval": config.watcher_interval,
            "parent_session_id": config.parent_session_id,
            "notify_on_complete": config.notify_on_complete,
            "notify_on_failure": config.notify_on_failure,
            "watch_patterns": list(config.watch_patterns),
        }

    @staticmethod
    def _pending_watcher_for_session(
        session: ProcessSession,
    ) -> Optional[Dict[str, Any]]:
        """Build the gateway completion watcher registered with a new session."""
        if session.watcher_interval <= 0 or not session.watcher_platform:
            return None
        return {
            "session_id": session.id,
            "check_interval": session.watcher_interval,
            "session_key": session.session_key,
            "platform": session.watcher_platform,
            "chat_id": session.watcher_chat_id,
            "user_id": session.watcher_user_id,
            "user_name": session.watcher_user_name,
            "thread_id": session.watcher_thread_id,
            "message_id": session.watcher_message_id,
            "notify_on_complete": session.notify_on_complete,
            "notify_on_failure": session.notify_on_failure,
            "parent_session_id": session.parent_session_id,
        }

    def _rollback_spawn_activation(self, session: ProcessSession) -> None:
        """Remove registration/checkpoint state after reader setup fails."""
        with self._lock:
            if self._running.get(session.id) is session:
                self._running.pop(session.id, None)
            self.pending_watchers = [
                watcher
                for watcher in self.pending_watchers
                if watcher.get("session_id") != session.id
            ]
        self._write_checkpoint()

    def _activate_spawned_session(
        self,
        session: ProcessSession,
        reader: threading.Thread,
    ) -> None:
        """Publish config durably before any Hermes output consumer starts.

        The child may already have written into its kernel pipe/PTY or remote
        log.  Starting the reader only after registration and checkpointing
        creates the required happens-before edge: Hermes cannot scan or emit
        those bytes with incomplete notification/routing metadata.
        """
        watcher = self._pending_watcher_for_session(session)
        with self._lock:
            self._prune_if_needed()
            self._running[session.id] = session
            if watcher is not None:
                self.pending_watchers.append(watcher)

        if not self._write_checkpoint():
            self._rollback_spawn_activation(session)
            raise RuntimeError(
                "Could not persist background process metadata before reader start"
            )

        try:
            reader.start()
        except BaseException:
            self._rollback_spawn_activation(session)
            raise

    @staticmethod
    def _uses_reliable_control_failure_fallback(
        session: ProcessSession,
    ) -> bool:
        """Whether completion is the Fable protocol's selective fail-safe."""
        return bool(
            session.notify_on_failure
            and not session.notify_on_complete
            and session.watch_patterns == [RELIABLE_CONTROL_WATCH_PATTERN]
        )

    def _build_reliable_control_failure_event(
        self,
        session: ProcessSession,
        *,
        control_reason: str = "missing_reliable_control_sentinel",
    ) -> Dict[str, Any]:
        """Build the durable wake used when no trustworthy close was seen."""
        from tools.process_registry import _redact_process_result
        exit_code = session.exit_code
        reason = session.completion_reason or "exited"
        event = {
            "type": "watch_match",
            "pattern": RELIABLE_CONTROL_WATCH_PATTERN,
            "control_reason": control_reason,
            "session_id": session.id,
            "session_key": session.session_key,
            "task_id": session.task_id,
            "owner_task_id": session.owner_task_id or session.task_id,
            "command": session.command,
            "termination_source": session.termination_source,
            "output": (
                f"{RELIABLE_CONTROL_WATCH_PATTERN} "
                f"reason={control_reason} "
                f"completion_reason={reason} exit_code={exit_code}"
            ),
            "suppressed": 0,
            "platform": session.watcher_platform,
            "chat_id": session.watcher_chat_id,
            "user_id": session.watcher_user_id,
            "user_name": session.watcher_user_name,
            "thread_id": session.watcher_thread_id,
            "message_id": session.watcher_message_id,
            "parent_session_id": session.parent_session_id,
            "started_at": session.started_at,
            "delivery_id": (
                f"watch-fallback:{session.id}:{session.started_at:.9f}"
            ),
        }
        _redact_process_result(event)
        return event

    def _move_to_finished(
        self,
        session: ProcessSession,
        *,
        consume_output: bool = False,
    ) -> None:
        """Atomically finalize one session and publish at most one result.

        Reader, explicit kill, detached refresh, and local reconciliation can
        all observe an exit concurrently.  The per-session finalization claim
        serializes their complete state/checkpoint/outbox/queue transaction;
        membership in ``_running`` alone is not a sufficient lock because a
        second caller could otherwise publish an empty checkpoint while the
        first caller is still committing its durable wake.

        ``consume_output`` is part of the same transition. Callers returning a
        terminal transcript inline set it before notification policy is
        evaluated, so abandoned-turn cleanup and wait cannot revive work via a
        failure-only Fable wake.
        """
        from tools.process_registry import _redact_process_result, save_completed_result
        with session._lock:
            if consume_output:
                self._completion_consumed.add(session.id)
                if session.watch_patterns == [RELIABLE_CONTROL_WATCH_PATTERN]:
                    # Make the consume verdict crash-durable before this
                    # finalizer can remove the producer checkpoint.
                    self._durably_consume_reliable_control(session)

            if session._finalization_started:
                # A consuming caller may arrive after the sole producer has
                # committed. Revoke its durable control row while still using
                # the consumed bit to suppress any RAM event already dequeued.
                if (
                    consume_output
                    and session.watch_patterns
                    == [RELIABLE_CONTROL_WATCH_PATTERN]
                ):
                    self._durably_consume_reliable_control(session)
                session._completion_event.set()
                return

            session._finalization_started = True
            with self._lock:
                # Keep the producer in ``_running`` until its durable verdict
                # is committed. Readers mark ``exited`` before reaching this
                # method, and unrelated process spawns/checkpoints can race in
                # that window. Removing it here would let their snapshot erase
                # the only recovery row before this finalizer writes an outbox
                # fallback. ``_write_checkpoint(remove_session_ids=...)`` is
                # the explicit commit point for terminal removal.
                was_running = session.id in self._running
                if was_running:
                    save_completed_result(session)
                if not was_running:
                    self._finished[session.id] = session
            if not was_running:
                if (
                    consume_output
                    and session.watch_patterns
                    == [RELIABLE_CONTROL_WATCH_PATTERN]
                ):
                    self._durably_consume_reliable_control(session)
                self._release_finished_handles(session)
                session._finalization_complete = True
                session._completion_event.set()
                return

            try:
                if self.is_completion_consumed(session.id):
                    # Explicit wait/log/kill (and abandoned-turn cleanup)
                    # already put the result in the caller's current turn.
                    if session.watch_patterns == [RELIABLE_CONTROL_WATCH_PATTERN]:
                        self._durably_consume_reliable_control(session)
                    else:
                        self._write_checkpoint(
                            remove_session_ids={session.id}
                        )
                    return

                had_unpersisted_primary_capture = bool(
                    session._reliable_control_watch_delivered
                    and not session._reliable_control_watch_persisted
                )
                notify_completion = bool(
                    should_notify_process_completion(session)
                )
                reliable_fallback = bool(
                    notify_completion
                    and self._uses_reliable_control_failure_fallback(session)
                )
                if reliable_fallback:
                    # Reserve the one-shot edge before disk I/O. A reader that
                    # was already scanning its final chunk observes this state
                    # under the same lock and cannot queue a second marker.
                    session._reliable_control_watch_delivered = True
                    session._watch_disabled = True
                unpersisted_reliable_capture = bool(
                    self._uses_reliable_control_failure_fallback(session)
                    and session.watcher_platform
                    and had_unpersisted_primary_capture
                )
                if unpersisted_reliable_capture:
                    # A primary FABLE_WAKE exists in RAM, but its outbox write
                    # is either failed or still resolving. Keep the false
                    # tombstone serialized with the primary path's update.
                    self._write_checkpoint(
                        extra_entries=[self._checkpoint_entry_for_session(session)]
                    )
                    return

                if reliable_fallback:
                    # Convert failure-only completion into the same durable,
                    # one-shot control event as a primary marker.
                    notification = self._build_reliable_control_failure_event(
                        session
                    )
                    durable_route = bool(session.watcher_platform)
                    if durable_route:
                        outbox_persisted = self._persist_reliable_watch_event(
                            notification
                        )
                    else:
                        notification.pop("delivery_id", None)
                        outbox_persisted = False
                    session._reliable_control_watch_persisted = outbox_persisted
                    if not durable_route or outbox_persisted:
                        if session._retain_checkpoint_for_scope_reap:
                            checkpoint_written = self._write_checkpoint(
                                extra_entries=[
                                    self._checkpoint_entry_for_session(session)
                                ]
                            )
                        else:
                            checkpoint_written = self._write_checkpoint(
                                remove_session_ids={session.id}
                            )
                    else:
                        # Keep a recovery tombstone when the outbox itself could
                        # not be written. A later process can conservatively
                        # recreate the wake instead of treating this as clean.
                        checkpoint_written = self._write_checkpoint(
                            extra_entries=[
                                self._checkpoint_entry_for_session(session)
                            ]
                        )
                    notification["checkpoint_confirmed"] = bool(
                        outbox_persisted and checkpoint_written
                    )
                    if self.is_completion_consumed(session.id):
                        self._durably_consume_reliable_control(session)
                        return
                    self.completion_queue.put(notification)
                    return

                if session._retain_checkpoint_for_scope_reap:
                    self._write_checkpoint(
                        extra_entries=[self._checkpoint_entry_for_session(session)]
                    )
                else:
                    self._write_checkpoint(remove_session_ids={session.id})

                if notify_completion:
                    from tools.ansi_strip import strip_ansi
                    output_tail = (
                        strip_ansi(session.output_buffer[-2000:])
                        if session.output_buffer
                        else ""
                    )
                    notification = {
                        "type": "completion",
                        "session_id": session.id,
                        "session_key": session.session_key,
                        "task_id": session.task_id,
                        "owner_task_id": session.owner_task_id or session.task_id,
                        **({"handoff_note": session.handoff_note} if session.handoff_note else {}),
                        "command": session.command,
                        "exit_code": session.exit_code,
                        "completion_reason": session.completion_reason,
                        "termination_source": session.termination_source,
                        "output": output_tail,
                        "started_at": session.started_at,
                    }
                    _redact_process_result(notification)
                    self.completion_queue.put(notification)
            finally:
                with self._lock:
                    self._running.pop(session.id, None)
                    self._finished[session.id] = session
                self._release_finished_handles(session)
                session._finalization_complete = True
                session._completion_event.set()

    def is_notification_consumed(self, event: Dict[str, Any]) -> bool:
        """Whether inline output consumption suppresses this queued event.

        Historically only ``completion`` rows used the consumed marker. The
        failure-only Fable protocol represents completion as a ``watch_match``
        instead, so every delivery rail must classify that exact control event
        the same way without suppressing ordinary recurring watch patterns.
        """
        session_id = str(event.get("session_id") or "")
        if not self.is_completion_consumed(session_id):
            return False
        return bool(
            event.get("type", "completion") == "completion"
            or (
                event.get("type") == "watch_match"
                and event.get("pattern") == RELIABLE_CONTROL_WATCH_PATTERN
            )
        )

    def _consume_completion_result(self, session: ProcessSession) -> None:
        """Record inline consumption and revoke any queued durable control wake."""
        with session._lock:
            self._completion_consumed.add(session.id)
            reliable_control = (
                session.watch_patterns == [RELIABLE_CONTROL_WATCH_PATTERN]
            )
        if not reliable_control:
            return
        self._durably_consume_reliable_control(session)

    def _durably_consume_reliable_control(
        self,
        session: ProcessSession,
    ) -> bool:
        """Fence a consumed FABLE_WAKE durably before deleting its outbox row.

        ``_completion_consumed`` is process-local. If the outbox rewrite fails,
        this checkpoint tombstone survives restart and prevents the stale wake
        from being restored while recovery retries its physical deletion.
        """
        self._completion_consumed.add(session.id)
        if session.watch_patterns != [RELIABLE_CONTROL_WATCH_PATTERN]:
            return True
        # This fence is deliberately monotonic for the registry process's
        # lifetime. A matching event can be queued, dequeued by a gateway, or
        # sitting in an adapter task; none of those states are represented by
        # the finished-session LRU, so only process teardown is a safe implicit
        # release point.
        self._reliable_control_consumed.add(session.id)

        consumed_entry = self._checkpoint_entry_for_session(session)
        consumed_entry["reliable_control_delivery_consumed"] = True
        tombstone_written = self._write_checkpoint(
            extra_entries=[consumed_entry]
        )
        outbox_discarded = self.discard_reliable_watch_events_for_session(
            session.id
        )
        if outbox_discarded and session.exited:
            if session._retain_checkpoint_for_scope_reap:
                # Consumption suppresses the control edge, not the independent
                # obligation to retry an owned systemd scope that failed reaping.
                retained_entry = self._checkpoint_entry_for_session(session)
                retained_entry["reliable_control_delivery_consumed"] = True
                self._write_checkpoint(extra_entries=[retained_entry])
            else:
                # Once the outbox is confirmed empty this tombstone has no
                # further recovery work. A failed cleanup write merely retains
                # a safe fence.
                self._write_checkpoint(remove_session_ids={session.id})
        if not tombstone_written and not outbox_discarded:
            logger.error(
                "Could not durably record or delete consumed executive watch "
                "event for %s; keeping the in-memory consume fence",
                session.id,
            )
        return bool(tombstone_written or outbox_discarded)
