"""Minimal three-column agentic-efficiency SQLite ledger for cron runs.

Exactly one append per agentic model-path attempt. Schema is intentionally
tiny: tipo, resultado_material, tokens — nothing else.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from hermes_constants import get_hermes_home

AGENTIC_EFFICIENCY_FILE = get_hermes_home().resolve() / "cron" / "agentic_efficiency.db"

VALID_TIPOS = frozenset({"nueva", "retry", "gate"})
VALID_RESULTADOS = frozenset({"sí", "no"})

_lock = threading.RLock()
logger = logging.getLogger(__name__)

# Conservative: only treat these as structured mutation evidence.
_WRITE_TOOLS = frozenset({"write_file"})
_PATCH_TOOLS = frozenset({"patch"})
_TERMINAL_TOOLS = frozenset({"terminal"})
_DATA_TOOLS = frozenset({"web_search", "web_extract"})
_EXTERNAL_ACTION_TOOLS = frozenset(
    {
        "cronjob",
        "skill_manage",
        "memory",
        "memory_fabric_propose",
        "image_generate",
        "text_to_speech",
        "tool_call",
    }
)
_MUTATING_ACTIONS = frozenset(
    {"create", "update", "pause", "resume", "remove", "run", "patch", "edit", "delete", "write_file", "add", "replace"}
)
_MUTATION_VERBS = frozenset(
    {"create", "update", "delete", "remove", "send", "post", "publish", "upload", "write", "mutate", "run"}
)
_RECEIPT_KEYS = frozenset(
    {"id", "job_id", "url", "path", "resolved_path", "status", "state", "image", "media", "created", "updated", "deleted", "removed", "sha", "commit"}
)
_FAILED_STATUSES = frozenset(
    {"failed", "failure", "error", "cancelled", "canceled", "rejected"}
)
_GIT_COMMIT_RE = re.compile(
    r"(?:^|[;&|]\s*|\n\s*)git\s+commit\b",
    re.IGNORECASE,
)


class AgenticTipoError(ValueError):
    """Job declared an invalid agentic_execution_type; fail closed."""


def _connect() -> sqlite3.Connection:
    AGENTIC_EFFICIENCY_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AGENTIC_EFFICIENCY_FILE), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS agentic_efficiency (
             tipo TEXT NOT NULL CHECK(tipo IN ('nueva','retry','gate')),
             resultado_material TEXT NOT NULL CHECK(resultado_material IN ('sí','no')),
             tokens INTEGER NOT NULL CHECK(tokens >= 0)
           )"""
    )
    return conn


def append_agentic_efficiency(
    tipo: str, resultado_material: str, tokens: int
) -> None:
    """Validate and append exactly one row to the agentic_efficiency table."""
    if tipo not in VALID_TIPOS:
        raise ValueError(f"tipo must be one of {sorted(VALID_TIPOS)}, got {tipo!r}")
    if resultado_material not in VALID_RESULTADOS:
        raise ValueError(
            "resultado_material must be one of "
            f"{sorted(VALID_RESULTADOS)}, got {resultado_material!r}"
        )
    try:
        tokens_i = int(tokens)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tokens must be a non-negative integer, got {tokens!r}") from exc
    if tokens_i < 0:
        raise ValueError(f"tokens must be >= 0, got {tokens_i}")

    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO agentic_efficiency (tipo, resultado_material, tokens) "
            "VALUES (?, ?, ?)",
            (tipo, resultado_material, tokens_i),
        )
        conn.commit()


def resolve_agentic_tipo(
    job: Optional[Mapping[str, Any]] = None,
    *,
    profile_name: Optional[str] = None,
) -> str:
    """Resolve tipo for a cron job.

    Precedence:
      1. Explicit job['agentic_execution_type'] when present (must be valid).
      2. autonomous profile named ``retry`` (job field or resolved profile_name).
      3. Default ``nueva``.
    """
    job = job or {}
    if "agentic_execution_type" in job and job.get("agentic_execution_type") is not None:
        raw = job.get("agentic_execution_type")
        if not isinstance(raw, str) or raw not in VALID_TIPOS:
            raise AgenticTipoError(
                f"invalid agentic_execution_type {raw!r}; "
                f"expected one of {sorted(VALID_TIPOS)}"
            )
        return raw

    profile = str(job.get("autonomous_profile") or profile_name or "").strip()
    if profile == "retry":
        return "retry"
    return "nueva"


def _parse_tool_json(content: Any) -> Optional[dict]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _tool_call_map(messages: Sequence[Any]) -> dict[str, tuple[str, str]]:
    """Map tool_call id → (function_name, arguments_str)."""
    out: dict[str, tuple[str, str]] = {}
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, Mapping):
                continue
            tid = tc.get("id")
            fn = tc.get("function") if isinstance(tc.get("function"), Mapping) else {}
            name = str(fn.get("name") or "")
            args = fn.get("arguments")
            if isinstance(args, Mapping):
                args_s = json.dumps(args, ensure_ascii=False)
            else:
                args_s = str(args or "")
            if tid:
                out[str(tid)] = (name, args_s)
    return out


def _command_from_args(args_s: str) -> str:
    try:
        parsed = json.loads(args_s) if args_s else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return args_s
    if isinstance(parsed, Mapping):
        return str(parsed.get("command") or "")
    return args_s


def _is_material_write(payload: Mapping[str, Any], workdir: Optional[Path]) -> bool:
    if payload.get("error"):
        return False
    try:
        bytes_written = int(payload.get("bytes_written", 0) or 0)
    except (TypeError, ValueError):
        return False
    if bytes_written <= 0:
        return False
    resolved = payload.get("resolved_path") or payload.get("path")
    if not resolved:
        return False
    try:
        path = Path(str(resolved))
        if not path.is_absolute():
            if workdir is None:
                return False
            path = workdir / path
        return path.is_file()
    except (TypeError, ValueError, OSError):
        return False


def _is_material_patch(payload: Mapping[str, Any]) -> bool:
    if payload.get("success") is not True:
        return False
    files = payload.get("files_modified")
    return isinstance(files, list) and len(files) > 0


def _is_material_git_commit(
    payload: Mapping[str, Any], command: str
) -> bool:
    if not command or not _GIT_COMMIT_RE.search(command):
        return False
    try:
        exit_code = int(payload.get("exit_code"))
    except (TypeError, ValueError):
        return False
    return exit_code == 0


def _parse_args(args_s: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(args_s) if args_s else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _has_nonempty_data(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_has_nonempty_data(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nonempty_data(v) for v in value)
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None and value is not False


def _is_material_data(name: str, payload: Mapping[str, Any]) -> bool:
    if name not in _DATA_TOOLS or payload.get("error"):
        return False
    if name == "web_search":
        return _has_nonempty_data(payload.get("data") or payload.get("results"))
    results = payload.get("results")
    if not isinstance(results, list):
        return False
    return any(
        isinstance(item, Mapping)
        and not item.get("error")
        and _has_nonempty_data(item.get("content"))
        for item in results
    )


def _has_receipt(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    for key, item in value.items():
        if key in _RECEIPT_KEYS and _has_nonempty_data(item):
            return True
        if isinstance(item, Mapping) and _has_receipt(item):
            return True
        if isinstance(item, list) and any(_has_receipt(v) for v in item):
            return True
    return False


def _payload_failed(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("success") is False or value.get("ok") is False:
            return True
        if any(
            value.get(flag) is True
            for flag in ("failed", "cancelled", "canceled")
        ):
            return True
        if bool(value.get("error")) or bool(value.get("errors")):
            return True
        raw_status = value.get("status") or value.get("state") or ""
        if isinstance(raw_status, int) and raw_status >= 400:
            return True
        status = str(raw_status).lower()
        if status in _FAILED_STATUSES:
            return True
        exit_code = value.get("exit_code")
        if isinstance(exit_code, int) and exit_code != 0:
            return True
        return any(_payload_failed(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_payload_failed(item) for item in value)
    return False


def _is_material_external_action(
    name: str, payload: Mapping[str, Any], args_s: str
) -> bool:
    if name not in _EXTERNAL_ACTION_TOOLS or _payload_failed(payload):
        return False
    args = _parse_args(args_s)
    if name in {"image_generate", "text_to_speech"}:
        return _has_receipt(payload)
    if name == "memory_fabric_propose" and args.get("write") is not True:
        return False
    if name == "tool_call":
        underlying = str(args.get("name") or "").lower()
        if not any(verb in underlying for verb in _MUTATION_VERBS):
            return False
    else:
        action = str(args.get("action") or "").lower()
        if action and action not in _MUTATING_ACTIONS:
            return False
        if name in {"cronjob", "skill_manage", "memory"} and not action:
            return False
    return _has_receipt(payload)


def classify_resultado_material(
    messages: Any,
    *,
    workdir: Optional[Any] = None,
) -> str:
    """Conservative structured-material classifier.

    Defaults to ``no``. Only strict tool-result shapes (write_file bytes,
    successful patch with files_modified, terminal git commit exit 0) count.
    Final narrative, free-form claims, and generic ``success: true`` never count.
    """
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return "no"

    workdir_path: Optional[Path] = None
    if workdir is not None:
        try:
            workdir_path = Path(workdir)
        except (TypeError, ValueError):
            workdir_path = None

    call_map = _tool_call_map(messages)

    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        if msg.get("role") != "tool":
            continue
        tid = msg.get("tool_call_id")
        mapped_name, args_s = call_map.get(str(tid), ("", "")) if tid else ("", "")
        name = str(msg.get("name") or mapped_name or "")
        payload = _parse_tool_json(msg.get("content"))
        if payload is None:
            continue

        if name in _WRITE_TOOLS and _is_material_write(payload, workdir_path):
            return "sí"

        if name in _PATCH_TOOLS and _is_material_patch(payload):
            return "sí"

        if name in _TERMINAL_TOOLS:
            command = ""
            if tid and str(tid) in call_map:
                tname, terminal_args_s = call_map[str(tid)]
                if tname in _TERMINAL_TOOLS:
                    command = _command_from_args(terminal_args_s)
            if _is_material_git_commit(payload, command):
                return "sí"

        if _is_material_data(name, payload):
            return "sí"

        if _is_material_external_action(name, payload, args_s):
            return "sí"

    return "no"


class OnceOnlyAgenticRecorder:
    """Record at most one agentic_efficiency row for a single run_job attempt."""

    def __init__(self, tipo: str) -> None:
        if tipo not in VALID_TIPOS:
            raise AgenticTipoError(
                f"invalid tipo {tipo!r}; expected one of {sorted(VALID_TIPOS)}"
            )
        self.tipo = tipo
        self._done = False
        self._guard = threading.Lock()

    @property
    def recorded(self) -> bool:
        return self._done

    def record(
        self,
        *,
        messages: Any = None,
        tokens: Any = 0,
        workdir: Optional[Any] = None,
    ) -> None:
        """Append once. Failures propagate; subsequent calls are no-ops."""
        with self._guard:
            if self._done:
                return
            # Mark before the insert so a failed write cannot be retried into
            # a duplicate row on a later path (fail closed, no duplicates).
            self._done = True
            try:
                tokens_i = int(tokens or 0)
            except (TypeError, ValueError):
                tokens_i = 0
            if tokens_i < 0:
                tokens_i = 0
            resultado = classify_resultado_material(messages, workdir=workdir)
            append_agentic_efficiency(self.tipo, resultado, tokens_i)


def record_agentic_efficiency_once(
    recorder: Optional[OnceOnlyAgenticRecorder],
    *,
    messages: Any = None,
    tokens: Any = 0,
    workdir: Optional[Any] = None,
) -> None:
    """Scheduler helper: no-op when recorder is None (non-model paths)."""
    if recorder is None:
        return
    recorder.record(messages=messages, tokens=tokens, workdir=workdir)


def record_agentic_efficiency_after_worker(
    recorder: Optional[OnceOnlyAgenticRecorder],
    *,
    agent: Any,
    fallback_result: Optional[Mapping[str, Any]],
    workdir: Optional[Any] = None,
    on_error: Optional[Callable[[BaseException], None]] = None,
) -> bool:
    """Record from the worker's final snapshot, immediately or via callback.

    Returns ``True`` when recorded synchronously and ``False`` when a callback
    was attached to an in-flight worker.  This avoids partial tokens/evidence
    when a hard limit returns control before the worker finishes unwinding.
    """
    if recorder is None:
        return True
    future = getattr(agent, "_cron_worker_future", None) if agent is not None else None

    def finalize(done_future: Any = None) -> None:
        result: Any = fallback_result
        if done_future is not None:
            try:
                candidate = done_future.result()
                if isinstance(candidate, Mapping):
                    result = candidate
            except BaseException:
                pass
        messages = result.get("messages") if isinstance(result, Mapping) else None
        tokens = int(getattr(agent, "session_total_tokens", 0) or 0)
        recorder.record(messages=messages, tokens=tokens, workdir=workdir)

    def callback(done_future: Any) -> None:
        try:
            finalize(done_future)
        except BaseException as exc:
            if on_error is not None:
                on_error(exc)
            else:
                logger.exception("Deferred agentic-efficiency ledger write failed")

    if future is not None and callable(getattr(future, "done", None)):
        if not future.done():
            future.add_done_callback(callback)
            return False
        finalize(future)
        return True
    finalize()
    return True
