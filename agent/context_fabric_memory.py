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
_CHANNEL_MENTION_RE = re.compile(r"<#(?P<id>\d+)>")
_CHANNEL_SCOPE_INTENT_RE = re.compile(
    r"\b(context\s*fabric|memory|memoria|scoped?|scope|canal|channel|audita|audit|revisa|review|diagn[oó]stic|fix|arregl|para|for)\b",
    re.IGNORECASE,
)
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


def _same_channel(left: str, right: str) -> bool:
    return _slug(left) == _slug(right)


def _infer_project_from_graph(payload: Mapping[str, Any], channel: str = "") -> str | None:
    graph = payload.get("graph_context")
    if not isinstance(graph, Mapping):
        return None
    relations = graph.get("semantic_relations")
    if not isinstance(relations, list):
        return None
    fallback: str | None = None
    for rel in relations:
        if not isinstance(rel, Mapping):
            continue
        relation = str(rel.get("relation") or "").strip().lower()
        if relation not in {"routes_to", "route_to", "maps_to"}:
            continue
        source = str(rel.get("from") or "").strip()
        target = str(rel.get("to") or "").strip()
        if not target:
            continue
        fallback = fallback or target
        if channel and source and _same_channel(source, channel):
            return target
    return fallback


def infer_project_from_payload(payload: Mapping[str, Any], *, channel: str = "") -> str | None:
    """Infer a Memory Fabric project from scoped rendered/graph context when present."""
    for key in ("scope_project", "project"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    inferred = _infer_project_from_graph(payload, channel=channel)
    if inferred:
        return inferred
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


def _route_lookup(routes: Mapping[str, Any], key: str | None) -> tuple[Mapping[str, Any], str | None]:
    """Return a route mapping for ``key`` without inventing a fallback route."""
    if not key:
        return {}, None
    candidates = [str(key).strip()]
    if candidates[0].startswith("#"):
        candidates.append(candidates[0].lstrip("#"))
    elif not candidates[0].isdigit():
        candidates.append(f"#{candidates[0]}")
    for candidate in candidates:
        maybe = routes.get(candidate)
        if isinstance(maybe, Mapping):
            return maybe, candidate
    if candidates[0].isdigit():
        for route_key, maybe in routes.items():
            if not isinstance(maybe, Mapping):
                continue
            route_channel_id = maybe.get("channel_id") or maybe.get("channelId")
            if route_channel_id is not None and str(route_channel_id).strip() == candidates[0]:
                return maybe, str(route_key)
    return {}, None


def _route_from_query_mentions(query: str, routes: Mapping[str, Any], *, allow_bare_channel_names: bool = False) -> tuple[Mapping[str, Any], str | None]:
    """Resolve an explicitly mentioned Discord channel in the user query.

    The active session channel is not always the channel being audited. A user may
    ask from #context-fabric about ``<#123>``; in that case Context Fabric must
    scope the Memory Fabric preflight to the mentioned channel's configured route
    instead of the current chat. Bare ``#name`` routing is opt-in only because it
    is common incidental prose and can otherwise create silent cross-project
    scope bleed. Only configured routes are honored, so arbitrary text cannot
    create a new scope.
    """
    if not query:
        return {}, None
    for match in _CHANNEL_MENTION_RE.finditer(query):
        # Explicit Discord mentions are only scope selectors when the nearby text
        # shows the user is asking about channel-scoped context. Pasted quotes or
        # casual references to another channel must not silently hijack this
        # turn's Memory Fabric scope.
        window = query[max(0, match.start() - 80) : min(len(query), match.end() + 80)]
        if not _CHANNEL_SCOPE_INTENT_RE.search(window):
            continue
        channel_id = match.group("id")
        route, _key = _route_lookup(routes, channel_id)
        if route:
            return route, channel_id
    if not allow_bare_channel_names:
        return {}, None
    for match in _CHANNEL_RE.finditer(query):
        route, key = _route_lookup(routes, match.group(0))
        if route:
            channel_id = route.get("channel_id") or route.get("channelId")
            return route, str(channel_id) if channel_id else (key if key and key.isdigit() else None)
    return {}, None


def _resolve_project_channel(
    config: Mapping[str, Any],
    payload: Mapping[str, Any] | None = None,
    *,
    query: str = "",
) -> tuple[str, str, str | None, Mapping[str, Any], str | None, str | None]:
    channel = _current_channel(config)
    chat_id = _session_value("HERMES_SESSION_CHAT_ID")
    session_channel_name = _session_value("HERMES_SESSION_CHAT_NAME")
    routes = config.get("channel_routes")
    route: Mapping[str, Any] = {}
    resolved_channel_id: str | None = chat_id or None
    resolved_channel_name: str | None = session_channel_name or None
    if isinstance(routes, Mapping):
        route_query = query or (str(payload.get("query") or "") if isinstance(payload, Mapping) else "")
        route, mentioned_channel_id = _route_from_query_mentions(
            route_query,
            routes,
            allow_bare_channel_names=_truthy(config.get("allow_bare_query_channel_route")),
        )
        if route:
            resolved_channel_id = mentioned_channel_id or None
            resolved_channel_name = str(route.get("channel") or "") or resolved_channel_name
        else:
            route, key = _route_lookup(routes, channel)
            if not route:
                route, key = _route_lookup(routes, chat_id)
            if route:
                resolved_channel_id = chat_id or (key if key and key.isdigit() else None)
    if not route and _truthy(config.get("restrict_to_routes")):
        return "", "", None, route, None, None
    resolved_channel = str(route.get("channel") or config.get("default_channel") or channel or "")
    if route and not resolved_channel_name:
        resolved_channel_name = resolved_channel
    project = str(
        route.get("project")
        or config.get("default_project")
        or (infer_project_from_payload(payload, channel=resolved_channel) if payload is not None else "")
        or _project_from_channel(resolved_channel)
    )
    workspace = route.get("workspace") or config.get("default_workspace")
    return project, resolved_channel, str(workspace) if workspace else None, route, resolved_channel_id, resolved_channel_name


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


def _csv_route_values(route: Mapping[str, Any], key: str) -> str:
    values = route.get(key)
    if isinstance(values, str):
        return values.strip()
    if isinstance(values, (list, tuple)):
        return ",".join(str(value).strip() for value in values if str(value).strip())
    return ""


def _append_route_options(args: list[str], route: Mapping[str, Any]) -> None:
    for key, flag in (
        ("required_skills", "--required-skills"),
        ("fallback_skills", "--fallback-skills"),
        ("enabled_toolsets", "--enabled-toolsets"),
    ):
        value = _csv_route_values(route, key)
        if value:
            args.extend([flag, value])
    task_type = str(route.get("task_type") or "").strip()
    if task_type:
        args.extend(["--task-type", task_type])
    budget_profile = str(route.get("budget_profile") or "").strip()
    if budget_profile:
        args.extend(["--budget-profile", budget_profile])


def maybe_filter_memory_context(
    raw_context: str,
    query: str,
    *,
    config: Mapping[str, Any] | None = None,
    compaction_count: int = 0,
) -> str:
    """Return Context-Fabric-filtered memory context when enabled.

    The configured command must accept the core preflight script arguments:
    ``--message``, ``--channel``, ``--project``, ``--memory-fabric-json`` and
    optional ``--workspace``. Newer preflight scripts also accept current
    Discord/session identity and compaction count; those flags are appended so
    AGENT_CONTEXT can distinguish exact-channel scope from ambiguous fallbacks.
    Its stdout replaces raw memory context.
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
    project, channel, workspace, route, resolved_channel_id, resolved_channel_name = _resolve_project_channel(cf_config, payload, query=query)
    logger.debug(
        "Context Fabric memory route decision: project=%s channel=%s workspace=%s route_matched=%s channel_id_present=%s bare_query_route_enabled=%s",
        project,
        channel,
        workspace or "",
        bool(route),
        bool(resolved_channel_id),
        _truthy(cf_config.get("allow_bare_query_channel_route")),
    )
    if not project or not channel:
        return raw_context
    # This runs synchronously on the turn preflight path. Keep it deliberately
    # tight and fail-open; Context Fabric is helpful context, not a reason to
    # stall the agent when its preflight command is slow or wedged.
    timeout = min(max(float(cf_config.get("timeout_seconds") or 3), 0.1), 3.0)
    try:
        fd, payload_path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
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
            guild_id = _session_value("HERMES_SESSION_GUILD_ID")
            channel_id = resolved_channel_id or _session_value("HERMES_SESSION_CHAT_ID")
            channel_name = resolved_channel_name or _session_value("HERMES_SESSION_CHAT_NAME")
            if guild_id:
                args.extend(["--guild-id", guild_id])
            if channel_id:
                args.extend(["--channel-id", channel_id])
            if channel_name:
                args.extend(["--channel-name", channel_name])
            if compaction_count > 0:
                args.extend(["--compaction-count", str(compaction_count)])
            _append_route_options(args, route)
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
            logger.debug("Context Fabric memory filter failed: rc=%s stderr_len=%s", completed.returncode, len(completed.stderr or ""))
            return raw_context
        return completed.stdout.strip()
    except Exception as exc:
        logger.debug("Context Fabric memory filter unavailable: %s", exc)
        return raw_context


__all__ = ["extract_memory_fabric_payload", "infer_project_from_payload", "maybe_filter_memory_context"]
