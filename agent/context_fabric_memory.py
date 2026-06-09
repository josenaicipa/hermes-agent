"""Optional Context Fabric filtering for external memory prefetch context.

This module is deliberately fail-open: if Context Fabric is disabled,
misconfigured, unavailable, or returns an error, Hermes keeps the original
memory provider context rather than dropping recall.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)
_MARKER = "[Memory Fabric scoped context]"
_CHANNEL_RE = re.compile(r"#[\w-]+")
_SCOPE_RE = re.compile(r"^##\s+scope:\s*([^\n\r]+)", re.IGNORECASE | re.MULTILINE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def extract_memory_fabric_payload(raw_context: str) -> dict[str, Any] | None:
    """Extract the JSON payload following the Memory Fabric context marker."""
    if not raw_context or _MARKER not in raw_context:
        return None
    after = raw_context.split(_MARKER, 1)[1].lstrip()
    try:
        payload, _idx = json.JSONDecoder().raw_decode(after)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_context_fabric_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        config = load_config()
        cf = config.get("context_fabric") if isinstance(config, dict) else None
        return dict(cf) if isinstance(cf, dict) else {}
    except Exception:
        return {}


def _session_value(name: str) -> str:
    try:
        from gateway.session_context import get_session_env

        return get_session_env(name, "") or ""
    except Exception:
        return os.getenv(name, "") or ""


def _current_channel(config: Mapping[str, Any]) -> str:
    default = str(config.get("default_channel") or "")
    explicit = _session_value("HERMES_SESSION_CHAT_NAME")
    match = _CHANNEL_RE.search(explicit)
    if match:
        return match.group(0)
    return explicit or default


def _slug(value: str) -> str:
    cleaned = _SLUG_RE.sub("-", value.strip().lower().lstrip("#")).strip("-")
    return cleaned or "default"


def infer_project_from_payload(payload: Mapping[str, Any]) -> str | None:
    """Infer a Memory Fabric project from scoped rendered context when present."""
    for key in ("scope_project", "project"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    rendered = payload.get("rendered")
    if isinstance(rendered, str):
        match = _SCOPE_RE.search(rendered)
        if match:
            scope = match.group(1).strip()
            # Common rendered scope is the project slug itself. If a future
            # renderer emits comma-separated scope fields, keep the first token.
            return scope.split(",", 1)[0].strip()
    return None


def _project_from_channel(channel: str) -> str:
    return _slug(channel)


def _resolve_project_channel(
    config: Mapping[str, Any],
    payload: Mapping[str, Any] | None = None,
) -> tuple[str, str, str | None]:
    channel = _current_channel(config)
    chat_id = _session_value("HERMES_SESSION_CHAT_ID")
    routes = config.get("channel_routes")
    route: Mapping[str, Any] = {}
    if isinstance(routes, Mapping):
        maybe = routes.get(channel) or routes.get(chat_id)
        if isinstance(maybe, Mapping):
            route = maybe
    if not route and _truthy(config.get("restrict_to_routes")):
        return "", "", None
    resolved_channel = str(route.get("channel") or config.get("default_channel") or channel or "")
    project = str(
        route.get("project")
        or config.get("default_project")
        or (infer_project_from_payload(payload) if payload is not None else "")
        or _project_from_channel(resolved_channel)
    )
    workspace = route.get("workspace") or config.get("default_workspace")
    return project, resolved_channel, str(workspace) if workspace else None


def _command_from_config(config: Mapping[str, Any]) -> list[str]:
    command = config.get("command")
    if isinstance(command, list) and command:
        return [str(part) for part in command]
    if isinstance(command, str) and command.strip():
        return [command]
    script = config.get("preflight_script")
    if script:
        return [sys.executable, str(script)]
    return []


def maybe_filter_memory_context(
    raw_context: str,
    query: str,
    *,
    config: Mapping[str, Any] | None = None,
) -> str:
    """Return Context-Fabric-filtered memory context when enabled.

    The configured command must accept the core preflight script arguments:
    ``--message``, ``--channel``, ``--project``, ``--memory-fabric-json`` and
    optional ``--workspace``. Its stdout replaces raw memory context.
    """
    cf_config = dict(config) if config is not None else _load_context_fabric_config()
    if not _truthy(cf_config.get("enabled")):
        return raw_context
    payload = extract_memory_fabric_payload(raw_context)
    if payload is None:
        return raw_context
    command = _command_from_config(cf_config)
    if not command:
        return raw_context
    project, channel, workspace = _resolve_project_channel(cf_config, payload)
    if not project or not channel:
        return raw_context
    timeout = float(cf_config.get("timeout_seconds") or 10)
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as fh:
            json.dump(payload, fh, ensure_ascii=False)
            payload_path = fh.name
        try:
            args = [
                *command,
                "--message",
                query,
                "--channel",
                channel,
                "--project",
                project,
                "--memory-fabric-json",
                payload_path,
            ]
            if workspace:
                args.extend(["--workspace", workspace])
            completed = subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        finally:
            try:
                Path(payload_path).unlink(missing_ok=True)
            except Exception:
                pass
        if completed.returncode != 0 or not completed.stdout.strip():
            logger.debug("Context Fabric memory filter failed: rc=%s stderr=%s", completed.returncode, completed.stderr[:500])
            return raw_context
        return completed.stdout.strip()
    except Exception as exc:
        logger.debug("Context Fabric memory filter unavailable: %s", exc)
        return raw_context


__all__ = ["extract_memory_fabric_payload", "infer_project_from_payload", "maybe_filter_memory_context"]
