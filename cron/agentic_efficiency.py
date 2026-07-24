"""Minimal three-column agentic-efficiency SQLite ledger for cron runs.

Exactly one append per agentic model-path attempt. Schema is intentionally
tiny: tipo, resultado_material, tokens — nothing else.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from hermes_constants import get_hermes_home

AGENTIC_EFFICIENCY_FILE = get_hermes_home().resolve() / "cron" / "agentic_efficiency.db"

VALID_TIPOS = frozenset({"nueva", "retry", "gate"})
VALID_RESULTADOS = frozenset({"sí", "no"})

_lock = threading.RLock()

# Conservative: only treat these as structured mutation evidence.
_WRITE_TOOLS = frozenset({"write_file"})
_PATCH_TOOLS = frozenset({"patch"})
_TERMINAL_TOOLS = frozenset({"terminal"})
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
    # Optional verification when a concrete path is claimed and workdir is known.
    # Absence of path is fine: structured bytes_written alone is enough.
    resolved = payload.get("resolved_path") or payload.get("path")
    if resolved and workdir is not None:
        try:
            path = Path(str(resolved))
            if path.is_absolute() and not path.exists():
                # Structured tool still reported bytes_written; keep accepting
                # the structured evidence (tool shape), not prose claims.
                pass
        except (TypeError, ValueError, OSError):
            pass
    return True


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
        name = str(msg.get("name") or "")
        payload = _parse_tool_json(msg.get("content"))
        if payload is None:
            continue

        if name in _WRITE_TOOLS and _is_material_write(payload, workdir_path):
            return "sí"

        if name in _PATCH_TOOLS and _is_material_patch(payload):
            return "sí"

        if name in _TERMINAL_TOOLS:
            tid = msg.get("tool_call_id")
            command = ""
            if tid and str(tid) in call_map:
                tname, args_s = call_map[str(tid)]
                if tname in _TERMINAL_TOOLS:
                    command = _command_from_args(args_s)
            if _is_material_git_commit(payload, command):
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
