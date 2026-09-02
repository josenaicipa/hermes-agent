"""Monotonic per-profile checkpoint and durable reliable-watch outbox."""
from __future__ import annotations
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from tools.process_registry import ProcessSession
from agent.redact import redact_sensitive_text
logger = logging.getLogger("tools.process_registry")
_IS_WINDOWS = platform.system() == "Windows"
RELIABLE_CONTROL_WATCH_PATTERN = "FABLE_WAKE"
RELIABLE_CONTROL_CLOSE_PATTERN = "FABLE_AUTO_CLOSE"
from tools.process_registry_control import (
    ProcessCheckpointRecoveryError, _CheckpointFileLock, _validate_process_checkpoint,
    _validate_reliable_watch_event, _validate_reliable_watch_outbox,
)

class ProcessCheckpointMixin:
    @staticmethod
    def _watch_outbox_path() -> Path:
        from tools.process_registry import _checkpoint_path
        return _checkpoint_path().with_name("process_notifications.json")

    def _checkpoint_entry_for_session(
        self, session: ProcessSession,
    ) -> Dict[str, Any]:
        """Return the durable recovery shape for one process session."""
        if (
            session.host_start_time is None
            and session.pid_scope == "host"
            and session.pid
        ):
            session.host_start_time = self._safe_host_start_time(session.pid)
        reliable_wake_is_durable = bool(
            session._reliable_control_watch_delivered
            and session._reliable_control_watch_persisted
        )
        return {
            "session_id": session.id,
            "checkpoint_owner_id": self._checkpoint_owner_id,
            # Recovery never re-runs this command, so masking is lossless.
            "command": redact_sensitive_text(session.command, code_file=True),
            "pid": session.pid,
            "pid_scope": session.pid_scope,
            "host_start_time": session.host_start_time,
            "systemd_unit": session.systemd_unit,
            "cwd": session.cwd,
            "started_at": session.started_at,
            "task_id": session.task_id,
            "owner_task_id": session.owner_task_id or session.task_id,
            "handoff_note": session.handoff_note,
            "session_key": session.session_key,
            "watcher_platform": session.watcher_platform,
            "watcher_chat_id": session.watcher_chat_id,
            "watcher_user_id": session.watcher_user_id,
            "watcher_user_name": session.watcher_user_name,
            "watcher_thread_id": session.watcher_thread_id,
            "watcher_message_id": session.watcher_message_id,
            "watcher_interval": session.watcher_interval,
            "parent_session_id": session.parent_session_id,
            "notify_on_complete": session.notify_on_complete,
            "notify_on_failure": session.notify_on_failure,
            "watch_patterns": list(session.watch_patterns),
            "reliable_control_watch_delivered": reliable_wake_is_durable,
            "reliable_control_close_seen": session._reliable_control_close_seen,
            # Crash-durable suppression fence for an event already returned
            # inline by wait/read_log/kill.
            "reliable_control_delivery_consumed": (
                self.is_completion_consumed(session.id)
            ),
        }

    @staticmethod
    def _read_watch_outbox(path: Path) -> List[Dict[str, Any]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("process notification outbox is not a list")
            if any(not isinstance(item, dict) for item in data):
                raise ValueError(
                    "process notification outbox contains a non-object entry"
                )
            _validate_reliable_watch_outbox(data)
            return data
        except FileNotFoundError:
            return []
        except Exception as exc:
            # Fail closed. Treating permission errors or corrupt JSON as an
            # empty outbox lets ACK/consume paths delete their checkpoint
            # tombstone while the unread event still exists on disk.
            raise RuntimeError(
                f"could not read process notification outbox {path}: {exc}"
            ) from exc

    def _persist_reliable_watch_event(self, event: Dict[str, Any]) -> bool:
        """Durably enqueue the one-shot executive wake before RAM delivery."""
        delivery_id = str(event.get("delivery_id") or "")
        if not delivery_id:
            return False
        path = self._watch_outbox_path()
        lock_path = path.with_name(f"{path.name}.lock")
        try:
            _validate_reliable_watch_event(event)
            with _CheckpointFileLock(lock_path):
                entries = self._read_watch_outbox(path)
                if not any(item.get("delivery_id") == delivery_id for item in entries):
                    entries.append(dict(event))
                    from utils import atomic_json_write
                    atomic_json_write(path, entries)
            self._restored_watch_delivery_ids.add(delivery_id)
            session_id = str(event.get("session_id") or "")
            self._durable_watch_session_ids.add(session_id)
            with self._lock:
                tombstone = self._checkpoint_tombstones.get(session_id)
                if not (
                    tombstone
                    and tombstone.get(
                        "reliable_control_delivery_consumed", False
                    )
                ):
                    self._checkpoint_tombstones.pop(session_id, None)
            return True
        except Exception as exc:
            logger.error("Could not persist executive watch event: %s", exc)
            return False

    def acknowledge_watch_event(self, event: Dict[str, Any]) -> bool:
        """Remove a durable executive wake after its synthetic turn commits."""
        delivery_id = str(event.get("delivery_id") or "")
        if not delivery_id:
            return True
        path = self._watch_outbox_path()
        lock_path = path.with_name(f"{path.name}.lock")
        try:
            with _CheckpointFileLock(lock_path):
                entries = self._read_watch_outbox(path)
                retained = [
                    item for item in entries
                    if item.get("delivery_id") != delivery_id
                ]
                if len(retained) != len(entries):
                    from utils import atomic_json_write
                    atomic_json_write(path, retained)
            self._restored_watch_delivery_ids.discard(delivery_id)
            session_id = str(event.get("session_id") or "")
            if not any(
                str(item.get("session_id") or "") == session_id
                for item in retained
            ):
                self._durable_watch_session_ids.discard(session_id)
            return True
        except Exception as exc:
            logger.error("Could not acknowledge executive watch event: %s", exc)
            return False

    def retarget_reliable_watch_event(
        self,
        delivery_id: str,
        *,
        parent_session_id: str,
    ) -> bool:
        """Durably update a control wake route without changing its identity.

        Compression exhaustion can reset the conversation before a synthetic
        wake succeeds. Clearing its old parent route *before* that reset lets
        the same delivery id retry against the fresh chat session; if this
        rewrite fails, the caller must defer the reset so the old route remains
        valid and crash recovery cannot terminally drop the only wake.
        """
        delivery_id = str(delivery_id or "")
        if not delivery_id:
            return False
        path = self._watch_outbox_path()
        lock_path = path.with_name(f"{path.name}.lock")
        try:
            with _CheckpointFileLock(lock_path):
                entries = self._read_watch_outbox(path)
                found = False
                updated: List[Dict[str, Any]] = []
                for item in entries:
                    copied = dict(item)
                    if str(copied.get("delivery_id") or "") == delivery_id:
                        copied["parent_session_id"] = str(
                            parent_session_id or ""
                        )
                        found = True
                    updated.append(copied)
                if not found:
                    return False
                from utils import atomic_json_write
                atomic_json_write(path, updated)
            return True
        except Exception as exc:
            logger.error(
                "Could not retarget executive watch event %s: %s",
                delivery_id,
                exc,
            )
            return False

    def discard_reliable_watch_events_for_session(self, session_id: str) -> bool:
        """Forget durable control wakes after intentional output consumption."""
        session_id = str(session_id or "")
        if not session_id:
            return True
        path = self._watch_outbox_path()
        lock_path = path.with_name(f"{path.name}.lock")
        try:
            with _CheckpointFileLock(lock_path):
                entries = self._read_watch_outbox(path)
                removed_delivery_ids = {
                    str(item.get("delivery_id") or "")
                    for item in entries
                    if str(item.get("session_id") or "") == session_id
                }
                retained = [
                    item
                    for item in entries
                    if str(item.get("session_id") or "") != session_id
                ]
                if len(retained) != len(entries):
                    from utils import atomic_json_write
                    atomic_json_write(path, retained)
            with self._lock:
                self._checkpoint_tombstones.pop(session_id, None)
            self._restored_watch_delivery_ids.difference_update(
                removed_delivery_ids
            )
            self._durable_watch_session_ids.discard(session_id)
            return True
        except Exception as exc:
            logger.error(
                "Could not discard consumed executive watch event for %s: %s",
                session_id,
                exc,
            )
            return False

    def _load_reliable_watch_events(self) -> List[Dict[str, Any]]:
        """Read valid executive wakes and expose their producer identities."""
        path = self._watch_outbox_path()
        lock_path = path.with_name(f"{path.name}.lock")
        with _CheckpointFileLock(lock_path):
            entries = self._read_watch_outbox(path)

        return [dict(event) for event in entries]

    def _restore_reliable_watch_events(
        self,
        events: Optional[List[Dict[str, Any]]] = None,
        *,
        checkpoint_confirmed: bool = True,
    ) -> int:
        """Requeue durable, unacknowledged executive wakes at gateway startup."""
        if events is None:
            try:
                events = self._load_reliable_watch_events()
            except Exception as exc:
                logger.error(
                    "Could not restore process notification outbox: %s", exc
                )
                return 0

        restored = 0
        for event in events:
            delivery_id = str(event.get("delivery_id") or "")
            if (
                not delivery_id
                or delivery_id in self._restored_watch_delivery_ids
                or event.get("type") != "watch_match"
                or event.get("pattern") != RELIABLE_CONTROL_WATCH_PATTERN
            ):
                continue
            restored_event = dict(event)
            restored_event["restored"] = True
            restored_event["checkpoint_confirmed"] = checkpoint_confirmed
            self.completion_queue.put(restored_event)
            self._restored_watch_delivery_ids.add(delivery_id)
            self._durable_watch_session_ids.add(
                str(event.get("session_id") or "")
            )
            restored += 1
        return restored

    def _write_checkpoint(
        self,
        extra_entries: Optional[List[Dict[str, Any]]] = None,
        remove_session_ids: Optional[set[str]] = None,
    ) -> bool:
        """Write a monotonic running-process snapshot atomically.

        Returns ``True`` on success.  Spawn activation treats failure as fatal
        because starting a reader without the durable routing/config snapshot
        would recreate the lost-notification race.
        """
        from tools.process_registry import _checkpoint_path
        try:
            with self._checkpoint_lock:
                remove_ids = {
                    str(session_id) for session_id in (remove_session_ids or ())
                }
                extra_ids = {
                    str(item.get("session_id") or "")
                    for item in (extra_entries or ())
                    if item.get("session_id")
                }
                # The file is shared by gateway/CLI/worker processes under one
                # HERMES_HOME. Serialize read+merge+replace across processes
                # and replace only this producer's prior slice; otherwise two
                # disjoint registries erase one another's recoverable sessions.
                checkpoint_file_lock = _checkpoint_path().with_name(
                    f"{_checkpoint_path().name}.lock"
                )
                with _CheckpointFileLock(checkpoint_file_lock):
                    try:
                        existing = json.loads(
                            _checkpoint_path().read_text(encoding="utf-8")
                        )
                    except FileNotFoundError:
                        existing = []
                    if not isinstance(existing, list):
                        raise ValueError("process checkpoint is not a list")
                    if any(not isinstance(item, dict) for item in existing):
                        raise ValueError(
                            "process checkpoint contains a non-object entry"
                        )
                    _validate_process_checkpoint(existing)

                    # Validate the shared snapshot before changing any local
                    # tombstone state. If the file is semantically corrupt,
                    # this write must be a complete no-op rather than partly
                    # consuming/removing a recovery decision in RAM.
                    with self._lock:
                        planned_tombstones = dict(
                            self._checkpoint_tombstones
                        )
                        for session_id in remove_ids - extra_ids:
                            planned_tombstones.pop(session_id, None)
                        entries = []
                        for s in self._running.values():
                            # ``exited`` is observed before the reader can enter
                            # the serialized finalizer. Keep such rows until
                            # that finalizer commits a clean/consumed verdict or
                            # a durable fallback outbox row.
                            if (
                                s.id not in remove_ids
                                or not s._finalization_started
                            ):
                                entries.append(
                                    self._checkpoint_entry_for_session(s)
                                )
                        tracked_ids = {
                            item.get("session_id") for item in entries
                        }
                        if extra_entries:
                            for item in extra_entries:
                                session_id = str(item.get("session_id") or "")
                                if not session_id:
                                    continue
                                owned_item = dict(item)
                                owned_item["checkpoint_owner_id"] = (
                                    self._checkpoint_owner_id
                                )
                                planned_tombstones[session_id] = owned_item
                        for session_id, item in planned_tombstones.items():
                            if session_id in tracked_ids:
                                continue
                            entries.append(dict(item))

                        replacement_ids = {
                            str(item.get("session_id") or "")
                            for item in entries
                        }
                        preserved = []
                        seen_ids = set()
                        for item in existing:
                            session_id = str(item.get("session_id") or "")
                            if (
                                item.get("checkpoint_owner_id")
                                == self._checkpoint_owner_id
                                or session_id in remove_ids
                                or session_id in replacement_ids
                                or session_id in seen_ids
                            ):
                                continue
                            preserved.append(item)
                            seen_ids.add(session_id)
                        merged_entries = preserved + entries
                        _validate_process_checkpoint(merged_entries)

                        # Atomic write prevents partial files;
                        # _checkpoint_lock prevents an older in-process
                        # snapshot from publishing after a newer one, while
                        # the file lock protects the cross-process merge.
                        from utils import atomic_json_write
                        atomic_json_write(_checkpoint_path(), merged_entries)
                        self._checkpoint_tombstones.clear()
                        self._checkpoint_tombstones.update(
                            planned_tombstones
                        )

                        # A terminal removal and its in-memory ownership
                        # transfer must be one checkpoint transaction. If the
                        # session stayed in ``_running`` until after this lock
                        # was released, an unrelated writer could immediately
                        # snapshot it back into processes.json, reviving a
                        # clean AUTO_CLOSE on restart. Non-finalizing remove
                        # requests deliberately keep live sessions tracked.
                        for session_id in remove_ids:
                            session = self._running.get(session_id)
                            if (
                                session is None
                                or not session._finalization_started
                            ):
                                continue
                            self._running.pop(session_id, None)
                            self._finished[session_id] = session
            return True
        except Exception as e:
            logger.debug("Failed to write checkpoint file: %s", e, exc_info=True)
            return False

    @staticmethod
    def _session_from_checkpoint_entry(
        entry: Dict[str, Any],
        *,
        detached: bool = True,
        exited: bool = False,
        completion_reason: str = "exited",
    ) -> ProcessSession:
        from tools.process_registry import ProcessSession
        return ProcessSession(
            id=entry["session_id"],
            command=entry.get("command", "unknown"),
            task_id=entry.get("task_id", ""),
            owner_task_id=entry.get("owner_task_id", "") or entry.get("task_id", ""),
            handoff_note=entry.get("handoff_note", ""),
            session_key=entry.get("session_key", ""),
            pid=entry.get("pid"),
            host_start_time=entry.get("host_start_time"),
            pid_scope=entry.get("pid_scope", "host"),
            systemd_unit=entry.get("systemd_unit", ""),
            cwd=entry.get("cwd"),
            started_at=entry.get("started_at", time.time()),
            detached=detached,
            exited=exited,
            completion_reason=completion_reason,
            watcher_platform=entry.get("watcher_platform", ""),
            watcher_chat_id=entry.get("watcher_chat_id", ""),
            watcher_user_id=entry.get("watcher_user_id", ""),
            watcher_user_name=entry.get("watcher_user_name", ""),
            watcher_thread_id=entry.get("watcher_thread_id", ""),
            watcher_message_id=entry.get("watcher_message_id", ""),
            watcher_interval=entry.get("watcher_interval", 0),
            parent_session_id=entry.get("parent_session_id", ""),
            notify_on_complete=entry.get("notify_on_complete", False),
            notify_on_failure=entry.get("notify_on_failure", False),
            watch_patterns=entry.get("watch_patterns", []),
            _reliable_control_watch_delivered=entry.get(
                "reliable_control_watch_delivered", False
            ),
            _reliable_control_watch_persisted=entry.get(
                "reliable_control_watch_delivered", False
            ),
            _reliable_control_close_seen=entry.get(
                "reliable_control_close_seen", False
            ),
        )

    def _checkpoint_requires_control_fallback(
        self,
        entry: Dict[str, Any],
        *,
        force_failure: bool = False,
        outbox_state_unknown: bool = False,
    ) -> bool:
        """Return whether an unrecoverable process row still owes Fable a wake."""
        session_id = str(entry.get("session_id") or "")
        return bool(
            entry.get("notify_on_failure", False)
            and entry.get("watcher_platform", "")
            and entry.get("watch_patterns", [])
            == [RELIABLE_CONTROL_WATCH_PATTERN]
            and not entry.get("reliable_control_delivery_consumed", False)
            and (
                outbox_state_unknown
                or not entry.get("reliable_control_watch_delivered", False)
            )
            and (
                force_failure
                or not entry.get("reliable_control_close_seen", False)
            )
            and (
                outbox_state_unknown
                or session_id not in self._durable_watch_session_ids
            )
        )

    def _persist_checkpoint_loss_fallback(
        self,
        entry: Dict[str, Any],
        *,
        termination_source: str,
        control_reason: str = "missing_reliable_control_sentinel",
    ) -> tuple[Dict[str, Any], bool]:
        """Create the durable failure-only wake for an unrecoverable row."""
        failed_session = self._session_from_checkpoint_entry(
            entry,
            exited=True,
            completion_reason="lost",
        )
        failed_session.termination_source = termination_source
        fallback_event = self._build_reliable_control_failure_event(
            failed_session,
            control_reason=control_reason,
        )
        return (
            fallback_event,
            self._persist_reliable_watch_event(fallback_event),
        )

    def recover_from_checkpoint(self) -> int:
        """
        On gateway startup, probe PIDs from checkpoint file.

        Returns the number of processes recovered as detached.
        """
        from tools.process_registry import _stop_systemd_unit, _checkpoint_path
        # Load (but do not enqueue) the durable outbox first.  Reconcile its
        # producer state with the process checkpoint before a gateway consumer
        # can acknowledge the event; otherwise a crash between adapter accept
        # and checkpoint repair could resurrect a duplicate fallback wake.
        try:
            durable_events = self._load_reliable_watch_events()
        except Exception as exc:
            # Unknown is not empty. In particular, an outbox-only primary may
            # be the last recovery source after its producer checkpoint was
            # cleanly retired. Starting the gateway without being able to read
            # that source would silently lose FABLE_WAKE, so refuse startup
            # until storage is readable instead of serving in a degraded mode.
            raise ProcessCheckpointRecoveryError(
                "could not restore process notification outbox: "
                f"{exc}"
            ) from exc
        outbox_state_unknown = False

        checkpoint_file_lock = _checkpoint_path().with_name(
            f"{_checkpoint_path().name}.lock"
        )
        try:
            with _CheckpointFileLock(checkpoint_file_lock):
                entries = json.loads(
                    _checkpoint_path().read_text(encoding="utf-8")
                )
        except FileNotFoundError:
            self._restore_reliable_watch_events(
                durable_events, checkpoint_confirmed=True
            )
            return 0
        except Exception as exc:
            raise ProcessCheckpointRecoveryError(
                f"could not read process checkpoint {_checkpoint_path()}: {exc}"
            ) from exc
        if not isinstance(entries, list):
            raise ProcessCheckpointRecoveryError(
                f"process checkpoint {_checkpoint_path()} is not a list"
            )
        if any(not isinstance(entry, dict) for entry in entries):
            raise ProcessCheckpointRecoveryError(
                f"process checkpoint {_checkpoint_path()} contains a non-object entry"
            )
        try:
            _validate_process_checkpoint(entries)
        except Exception as exc:
            raise ProcessCheckpointRecoveryError(
                f"process checkpoint {_checkpoint_path()} is invalid: {exc}"
            ) from exc

        # Both durable inputs are now fully validated. Only now let outbox
        # producer identities influence recovery decisions.
        self._durable_watch_session_ids.update(
            str(event["session_id"]) for event in durable_events
        )

        # Rehydrate consumption before exposing any durable outbox event. If
        # physical deletion still fails, the checkpoint row remains a
        # monotonic suppression tombstone for the next restart.
        consumed_session_ids = {
            str(entry.get("session_id") or "")
            for entry in entries
            if entry.get("reliable_control_delivery_consumed", False)
            and entry.get("session_id")
        }
        consumed_discard_failed: set[str] = set()
        for session_id in consumed_session_ids:
            self._completion_consumed.add(session_id)
            self._reliable_control_consumed.add(session_id)
            if not self.discard_reliable_watch_events_for_session(session_id):
                consumed_discard_failed.add(session_id)
        if consumed_session_ids:
            durable_events = [
                event
                for event in durable_events
                if str(event.get("session_id") or "")
                not in consumed_session_ids
            ]

        checkpoint_session_ids = {
            str(entry.get("session_id") or "")
            for entry in entries
            if isinstance(entry, dict)
        }

        recovered = 0
        unresolved_scope_entries: List[Dict[str, Any]] = []
        generated_fallback_events: List[tuple[Dict[str, Any], bool]] = []
        for entry in entries:
            session_id = str(entry.get("session_id") or "")
            with self._lock:
                already_tracked = session_id in self._running
            if already_tracked:
                continue
            pid = entry.get("pid")
            if not pid:
                if session_id in consumed_discard_failed:
                    unresolved_scope_entries.append(entry)
                continue

            pid_scope = entry.get("pid_scope", "host")
            if pid_scope != "host":
                # Sandbox-backed processes keep only in-sandbox PIDs in the
                # checkpoint, which are not meaningful to the restarted host
                # process once the original environment handle is gone.
                logger.info(
                    "Skipping recovery for non-host process: %s (pid=%s, scope=%s)",
                    entry.get("command", "unknown")[:60],
                    pid,
                    pid_scope,
                )
                if self._checkpoint_requires_control_fallback(
                    entry,
                    force_failure=True,
                    outbox_state_unknown=outbox_state_unknown,
                ):
                    # The restarted process no longer owns the environment
                    # handle needed to probe a sandbox/SSH/container-local PID.
                    # Treat that unobservable watcher as lost without applying
                    # host PID ownership checks, and persist its selective wake
                    # before allowing the checkpoint row to disappear.
                    fallback_event, fallback_persisted = (
                        self._persist_checkpoint_loss_fallback(
                            entry,
                            termination_source="checkpoint_backend_unrecoverable",
                            control_reason=(
                                "checkpoint_exit_status_unavailable"
                                if entry.get("reliable_control_close_seen", False)
                                else "missing_reliable_control_sentinel"
                            ),
                        )
                    )
                    generated_fallback_events.append(
                        (fallback_event, fallback_persisted)
                    )
                    if not fallback_persisted:
                        unresolved_scope_entries.append(entry)
                    logger.warning(
                        "Recovered unobservable %s Fable watcher %s; "
                        "queued durable failure-only wake",
                        pid_scope,
                        entry.get("session_id", "?"),
                    )
                elif session_id in consumed_discard_failed:
                    unresolved_scope_entries.append(entry)
                continue

            # The PID must be alive AND still the same process we spawned. A
            # bare liveness check is unsafe: across a restart (especially a
            # reboot or long uptime) the kernel may have recycled this number
            # onto an unrelated process — adopting it would let a later kill or
            # watcher tree-kill a stranger (e.g. a browser). Re-validate the
            # kernel start time recorded in the checkpoint.
            recorded_start = entry.get("host_start_time")
            if not self._host_pid_is_ours(pid, recorded_start):
                if self._is_host_pid_alive(pid):
                    logger.info(
                        "Not recovering session %s: pid %d is alive but its "
                        "start time no longer matches — PID was recycled onto "
                        "an unrelated process; refusing to adopt it.",
                        entry.get("session_id", "?"), pid,
                    )
                systemd_unit = entry.get("systemd_unit", "")
                if systemd_unit and not _stop_systemd_unit(systemd_unit):
                    retained_entry = dict(entry)
                    session_id = str(entry.get("session_id") or "")
                    if self._checkpoint_requires_control_fallback(
                        entry,
                        force_failure=True,
                        outbox_state_unknown=outbox_state_unknown,
                    ):
                        # Failure to reap an owned scope after its wrapper died
                        # is itself an abnormal terminal state. Wake Fable even
                        # when the wrapper had printed AUTO_CLOSE, while keeping
                        # the row so a later startup can retry scope cleanup.
                        fallback_event, fallback_persisted = (
                            self._persist_checkpoint_loss_fallback(
                                entry,
                                termination_source="checkpoint_scope_reap_failed",
                                control_reason="checkpoint_scope_reap_failed",
                            )
                        )
                        generated_fallback_events.append(
                            (fallback_event, fallback_persisted)
                        )
                        if fallback_persisted:
                            retained_entry[
                                "reliable_control_watch_delivered"
                            ] = True
                    elif session_id in self._durable_watch_session_ids:
                        # Repair an outbox-only half transaction before the
                        # retained cleanup row can outlive acknowledgement.
                        retained_entry[
                            "reliable_control_watch_delivered"
                        ] = True
                    logger.warning(
                        "Could not reap persisted scope %s for dead wrapper pid %s; "
                        "retaining checkpoint entry for the next startup",
                        systemd_unit,
                        pid,
                    )
                    unresolved_scope_entries.append(retained_entry)
                    continue

                if self._checkpoint_requires_control_fallback(
                    entry,
                    force_failure=True,
                    outbox_state_unknown=outbox_state_unknown,
                ):
                    # The gateway died before observing a terminal sentinel,
                    # or the watchdog exited while it was down. Persist a
                    # synthetic FABLE_WAKE before deleting the dead process
                    # row; a RAM-only pending watcher would be lost on another
                    # gateway crash.
                    fallback_event, fallback_persisted = (
                        self._persist_checkpoint_loss_fallback(
                            entry,
                            termination_source="checkpoint_process_lost",
                            control_reason=(
                                "checkpoint_exit_status_unavailable"
                                if entry.get("reliable_control_close_seen", False)
                                else "missing_reliable_control_sentinel"
                            ),
                        )
                    )
                    generated_fallback_events.append(
                        (fallback_event, fallback_persisted)
                    )
                    if not fallback_persisted:
                        # Retain the old recovery row when the durable outbox
                        # cannot be written.  Its no-sentinel state is the
                        # final fallback on the next startup.
                        unresolved_scope_entries.append(entry)
                    logger.warning(
                        "Recovered missing Fable control sentinel for %s; "
                        "queued durable failure-only wake",
                        entry.get("session_id", "?"),
                    )
                elif session_id in consumed_discard_failed:
                    unresolved_scope_entries.append(entry)
                continue

            live_entry = entry
            if self._checkpoint_requires_control_fallback(
                entry,
                force_failure=True,
                outbox_state_unknown=outbox_state_unknown,
            ):
                # A host PID can survive the gateway, but its inherited
                # stdout/PTY cannot be reattached by this registry. Waiting for
                # the PID to exit would silently miss every later FABLE_WAKE
                # (and may wait forever). Fail closed immediately and durably;
                # still adopt the PID below so normal kill/liveness/scope
                # cleanup remains available.
                fallback_event, fallback_persisted = (
                    self._persist_checkpoint_loss_fallback(
                        entry,
                        termination_source=(
                            "checkpoint_output_stream_unrecoverable"
                        ),
                        control_reason="output_stream_unrecoverable",
                    )
                )
                generated_fallback_events.append(
                    (fallback_event, fallback_persisted)
                )
                if fallback_persisted:
                    live_entry = dict(entry)
                    live_entry["reliable_control_watch_delivered"] = True
                logger.warning(
                    "Recovered live Fable watcher %s without a reattachable "
                    "output stream; queued durable failure-only wake",
                    entry.get("session_id", "?"),
                )

            session = self._session_from_checkpoint_entry(live_entry)
            if (
                session._reliable_control_watch_delivered
                or session.id in self._durable_watch_session_ids
            ):
                # Either side of the durable protocol transaction is evidence
                # that this marker was captured. In particular, repair an
                # outbox-only half-transaction before exposing its restored
                # event to a consumer.
                session._reliable_control_watch_delivered = True
                session._reliable_control_watch_persisted = True
                session._watch_disabled = True
            with self._lock:
                self._running[session.id] = session
            recovered += 1
            logger.info("Recovered detached process: %s (pid=%d)", session.command[:60], pid)

            # Re-enqueue the same watcher shape used by fresh activation.
            watcher = self._pending_watcher_for_session(session)
            if watcher is not None:
                self.pending_watchers.append(watcher)

        checkpoint_written = self._write_checkpoint(
            extra_entries=unresolved_scope_entries,
            remove_session_ids=checkpoint_session_ids,
        )
        self._restore_reliable_watch_events(
            durable_events,
            checkpoint_confirmed=checkpoint_written,
        )
        for event, outbox_persisted in generated_fallback_events:
            queued_event = dict(event)
            queued_event["restored"] = True
            queued_event["checkpoint_confirmed"] = bool(
                outbox_persisted and checkpoint_written
            )
            self.completion_queue.put(queued_event)

        return recovered
