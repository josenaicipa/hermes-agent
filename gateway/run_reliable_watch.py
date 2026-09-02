"""Crash-durable selective process notifications, separate from adapter admission."""
from __future__ import annotations

import asyncio
import logging
from gateway.platforms.event import MessageEvent
from gateway.platforms.event_receipts import has_gateway_delivery_receipt, resolve_gateway_delivery_receipt

logger = logging.getLogger("gateway.run")
_GATEWAY_OWNER_RECEIPT_OUTCOME_KEY = "_gateway_owner_receipt_outcome"

def _gateway_delivery_receipt_outcome(agent_result: dict) -> str:
    """ACK a synthetic control turn only after a genuinely successful run.

    Persisting a user row for a transient/partial failure is useful transcript
    continuity, but it does not mean Fable processed the control wake. Receipt
    ACK is stricter than the legacy resume marker: the runner must explicitly
    report ``completed is True``. Missing/None, failures, interrupts, partial
    results, and explicit errors all remain retryable in the durable outbox.
    """
    if isinstance(agent_result, dict):
        owner_outcome = agent_result.get(_GATEWAY_OWNER_RECEIPT_OUTCOME_KEY)
        if owner_outcome in {"durable", "retry"}:
            return owner_outcome

    completed_successfully = bool(
        isinstance(agent_result, dict)
        and agent_result.get("completed") is True
        and not agent_result.get("failed")
        and not agent_result.get("partial")
        and not agent_result.get("interrupted")
        and not agent_result.get("error")
    )
    return "durable" if completed_successfully else "retry"

def _preserve_queued_followup_history_offset(
    current_result: dict,
    followup_result: dict,
) -> dict:
    """Carry the outer history offset through queued follow-up drains.

    ``_process_message_background()`` persists transcript rows only once, after the
    entire in-band queued-follow-up chain returns.  Each recursive ``_run_agent()``
    call advances ``history_offset`` to the history it received, so without
    correction the outermost persistence step sees only the *last* queued turn as
    "new" and silently drops earlier turns from the same drain chain.

    Preserve the earliest (outermost) history offset so the final transcript slice
    still includes every queued turn that ran during the chain.
    """
    if not isinstance(followup_result, dict):
        return followup_result
    if not isinstance(current_result, dict):
        return followup_result

    merged = dict(followup_result)
    # A queued user follow-up is a distinct logical turn. The MessageEvent
    # receipt still belongs to the outer synthetic FABLE_WAKE, so its ACK
    # verdict must be frozen before the recursive turn can replace the public
    # result shape. Otherwise failure->success loses the wake, while
    # success->failure replays an already handled wake.
    merged[_GATEWAY_OWNER_RECEIPT_OUTCOME_KEY] = (
        _gateway_delivery_receipt_outcome(current_result)
    )

    current_offset = current_result.get("history_offset")
    followup_offset = followup_result.get("history_offset")
    if (
        isinstance(current_offset, int)
        and not (
            isinstance(followup_offset, int)
            and followup_offset <= current_offset
        )
    ):
        merged["history_offset"] = current_offset
    return merged


class GatewayReliableWatchMixin:
    def _prepare_reliable_watch_receipt(self, event, evt, adapter):
        from tools.process_registry import RELIABLE_CONTROL_WATCH_PATTERN
        if not (evt.get("delivery_id") and evt.get("type") == "watch_match"
                and evt.get("pattern") == RELIABLE_CONTROL_WATCH_PATTERN
                and getattr(adapter, "supports_durable_delivery_receipts", False)):
            return None
        receipt = asyncio.get_running_loop().create_future()
        event._gateway_durable_delivery_receipt = receipt

        def settle(done):
            try:
                outcome = str(done.result() or "retry")
            except BaseException:
                outcome = "retry"
            settlement_event = evt
            if outcome == "retry" and (event.metadata or {}).get("gateway_retry_without_parent_session"):
                settlement_event = {**evt, "parent_session_id": ""}
            self._settle_reliable_watch_receipt(settlement_event, outcome)

        receipt.add_done_callback(settle)
        return receipt

    async def _deliver_watch_events(self, watch_events):
        retry = []
        for evt in watch_events:
            async with self._completion_event_scope(evt):
                retry.extend(await self._deliver_scoped_watch_events([evt]))
        return retry

    def _settle_reliable_watch_receipt(
        self,
        evt: dict,
        outcome: str,
    ) -> None:
        """Commit or retry a receipt-aware control wake after real processing."""
        from tools.process_registry import process_registry as _process_registry

        if outcome == "durable":
            if evt.get("checkpoint_confirmed") is False:
                logger.warning(
                    "Keeping processed watch event %s in durable outbox: "
                    "process checkpoint not confirmed",
                    evt.get("delivery_id"),
                )
                return
            if not _process_registry.acknowledge_watch_event(evt):
                # Processing already completed. Keeping the durable row for
                # at-least-once replay is safer than an ACK-shaped loss; do not
                # hot-loop another model turn merely because deletion failed.
                logger.warning(
                    "Processed watch event %s remains in durable outbox; "
                    "acknowledgement will retry after restart",
                    evt.get("delivery_id"),
                )
            return

        if outcome == "consumed":
            _process_registry.discard_reliable_watch_events_for_session(
                str(evt.get("session_id") or "")
            )
            return

        # Busy adapter, cancellation, handler exception, or any unknown result:
        # retain the outbox and re-offer the same event in this live process.
        _process_registry.completion_queue.put(evt)

    def _finish_consumed_reliable_watch(self, event: MessageEvent) -> bool:
        """Suppress a receipt event consumed inline before its own turn starts."""
        if not has_gateway_delivery_receipt(event):
            return False
        metadata = getattr(event, "metadata", None) or {}
        process_session_id = str(
            metadata.get("gateway_process_session_id") or ""
        )
        if not process_session_id:
            resolve_gateway_delivery_receipt(event, "retry")
            return True

        from tools.process_registry import process_registry as _process_registry

        if not _process_registry.is_completion_consumed(process_session_id):
            return False
        if _process_registry.discard_reliable_watch_events_for_session(
            process_session_id
        ):
            resolve_gateway_delivery_receipt(event, "consumed")
        else:
            # Do not run the turn while deletion is uncertain. The durable
            # consumed checkpoint tombstone fences restart; retry live cleanup.
            resolve_gateway_delivery_receipt(event, "retry")
        return True

    async def _deliver_scoped_watch_events(self, watch_events: list[dict]) -> list[dict]:
        """Deliver a detached watch batch and return retryable failures.

        Both the post-turn drain and the persistent idle watcher use this one
        delivery seam. Adapter rejection/exception requeues the original event
        instead of silently losing a one-shot control wake.
        """
        from gateway.run import _format_gateway_process_notification

        if not watch_events:
            return []

        from tools.process_registry import (
            RELIABLE_CONTROL_WATCH_PATTERN,
            process_registry as _process_registry,
        )

        if self._load_background_notifications_mode() == "off":
            # Display preferences may suppress log/status chatter, but cannot
            # disable the executive control plane. Keep the exemption exact
            # and one-shot (the producer closes the watch after first match).
            watch_events = [
                evt
                for evt in watch_events
                if evt.get("type") == "watch_match"
                and evt.get("pattern") == RELIABLE_CONTROL_WATCH_PATTERN
            ]

        retry_events: list[dict] = []
        for evt in watch_events:
            is_control_wake = (
                evt.get("type") == "watch_match"
                and evt.get("pattern") == RELIABLE_CONTROL_WATCH_PATTERN
            )
            if (
                is_control_wake
                and _process_registry.is_completion_consumed(
                    str(evt.get("session_id") or "")
                )
            ):
                # Explicit wait/log/kill already surfaced the process result.
                # In particular, abandoned-turn cleanup must not resurrect the
                # cancelled work through its failure-only control fallback.
                if not _process_registry.discard_reliable_watch_events_for_session(
                    str(evt.get("session_id") or "")
                ):
                    retry_events.append(evt)
                continue
            parent_session_id = str(
                evt.get("parent_session_id") or ""
            ).strip()
            if parent_session_id:
                # Watch matches are delayed process results too. Apply the
                # same spawn-session boundary policy as completion events so a
                # durable FABLE_WAKE from before /new cannot enter the new
                # conversation. A terminal verdict is an honest durable drop;
                # transient DB uncertainty remains retryable.
                verdict = await self._classify_completion_target(
                    parent_session_id
                )
                if verdict == "terminal":
                    logger.warning(
                        "Background process %s watch event targets "
                        "permanently-gone session %s; dropping notification",
                        evt.get("session_id") or "<unknown>",
                        parent_session_id,
                    )
                    if evt.get("delivery_id"):
                        if not _process_registry.acknowledge_watch_event(evt):
                            retry_events.append(evt)
                    continue
                if verdict == "retry":
                    retry_events.append(evt)
                    continue

            synth_text = _format_gateway_process_notification(evt)
            if not synth_text:
                continue
            try:
                delivered = await self._inject_watch_notification(synth_text, evt)
                if delivered is False or (is_control_wake and delivered is None):
                    retry_events.append(evt)
                elif delivered is True and evt.get("delivery_id"):
                    if evt.get("checkpoint_confirmed") is False:
                        # Adapter acceptance happened, but the producer's
                        # process checkpoint was not durably reconciled. Keep
                        # the outbox row for at-least-once replay after restart
                        # instead of creating an ack-without-recovery gap.
                        logger.warning(
                            "Keeping delivered watch event %s in durable "
                            "outbox: process checkpoint not confirmed",
                            evt.get("delivery_id"),
                        )
                    else:
                        _process_registry.acknowledge_watch_event(evt)
            except Exception as exc:
                logger.error("Watch notification injection error: %s", exc)
                retry_events.append(evt)
        return retry_events
