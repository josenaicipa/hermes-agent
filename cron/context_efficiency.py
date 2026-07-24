"""Deterministic context-efficiency gates for cron execution.

This module deliberately contains no delivery adapters and no model prompt policy.
It persists profile-local state, returns decisions to ``cron.scheduler``, and keeps
all behavioural limits in code rather than in LLM instructions.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None


_GEMINI_PROVIDER = "google-gemini-cli"
_GEMINI_MODEL = "Gemini 3.5 Flash (Medium)"
_SCHEMA_VERSION = 1
_PROCESS_LOCK = threading.RLock()


class EfficiencyStateError(RuntimeError):
    """The durable gate state could not be trusted; callers must fail closed."""


@dataclass(frozen=True)
class EfficiencyConfig:
    enabled: bool = False
    compaction_tokens: int = 70_000
    blocker_alert_hours: float = 6.0
    redesign_primary_url: str = "http://127.0.0.1:4318/v1/chat/completions"
    redesign_backup_url: str = "http://127.0.0.1:4319/v1/chat/completions"
    redesign_project: str = "cron-context-efficiency"
    redesign_timeout_seconds: float = 120.0

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]]) -> "EfficiencyConfig":
        raw = raw if isinstance(raw, Mapping) else {}
        redesign = raw.get("redesign") if isinstance(raw.get("redesign"), Mapping) else {}
        enabled = raw.get("enabled", False) is True
        try:
            tokens = int(raw.get("compaction_tokens", 70_000))
        except (TypeError, ValueError) as exc:
            raise ValueError("cron.context_efficiency.compaction_tokens must be an integer") from exc
        if tokens < 16_000 or tokens > 10_000_000:
            raise ValueError("cron.context_efficiency.compaction_tokens must be between 16000 and 10000000")
        try:
            alert_hours = float(raw.get("blocker_alert_hours", 6.0))
            timeout = float(redesign.get("timeout_seconds", 120.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("context-efficiency durations must be numeric") from exc
        if alert_hours <= 0 or timeout <= 0:
            raise ValueError("context-efficiency durations must be positive")
        project = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(redesign.get("project") or "cron-context-efficiency")).strip("-")
        return cls(
            enabled=enabled,
            compaction_tokens=tokens,
            blocker_alert_hours=alert_hours,
            redesign_primary_url=str(redesign.get("primary_url") or cls.redesign_primary_url),
            redesign_backup_url=str(redesign.get("backup_url") or cls.redesign_backup_url),
            redesign_project=project or "cron-context-efficiency",
            redesign_timeout_seconds=timeout,
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(now: Optional[datetime] = None) -> str:
    current = now or _utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).isoformat()


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _safe_job_id(job: Mapping[str, Any]) -> str:
    raw = str(job.get("id") or "unknown")
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw)[:96] or "unknown"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd, tmp_name = tempfile.mkstemp(prefix=".context-efficiency-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
        os.chmod(path, 0o600)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd, tmp_name = tempfile.mkstemp(prefix=".cron-log-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
        os.chmod(path, 0o600)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def format_script_context(
    job: Mapping[str, Any],
    success: bool,
    output: str,
    *,
    hermes_home: Path,
    run_id: Optional[str] = None,
) -> str:
    """Persist complete redacted script output and return asymmetric model context."""
    raw = str(output or "")
    job_id = _safe_job_id(job)
    stable_run = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(run_id or hashlib.sha256(raw.encode()).hexdigest()[:20]))
    artifact = Path(hermes_home) / "cron" / "context-efficiency" / "logs" / job_id / f"{stable_run}.log"
    _atomic_text(artifact, raw)
    lines = raw.splitlines()
    if success:
        preview = lines[:20]
        clipped = "\n".join(preview)[:2000]
        return (
            "Script succeeded. Deterministic summary: "
            f"{len(lines)} lines, {len(raw)} characters.\n"
            f"First lines (max 20 / 2000 chars):\n{clipped}\n"
            f"Full redacted output: {artifact}"
        )

    stderr_lines: list[str] = []
    if "stderr:\n" in raw:
        stderr_part = raw.split("stderr:\n", 1)[1].split("\nstdout:\n", 1)[0]
        stderr_lines = stderr_part.splitlines()[:150]
    elif lines:
        stderr_lines = lines[:150]
    tail = lines[-80:]
    return (
        "Script failed. Raw stderr (max 150 lines):\n"
        + "\n".join(stderr_lines)
        + "\n\nUnfiltered tail of combined log (max 80 lines):\n"
        + "\n".join(tail)
        + f"\n\nFull redacted log (use read/grep/sed): {artifact}"
    )


def apply_cron_compaction_ceiling(agent: Any, config: EfficiencyConfig) -> Optional[int]:
    """Apply an absolute ceiling to the built-in cron compressor only."""
    if not config.enabled or getattr(agent, "platform", None) != "cron":
        return None
    if not getattr(agent, "compression_enabled", False):
        raise RuntimeError("cron context-efficiency requires compression_enabled")
    compressor = getattr(agent, "context_compressor", None)
    if getattr(compressor, "compression_enabled", True) is False:
        raise RuntimeError("cron context-efficiency compressor is disabled")
    current = getattr(compressor, "threshold_tokens", None)
    if not isinstance(current, int) or current <= 0:
        raise RuntimeError("cron context engine has no compatible threshold_tokens API")
    ceiling = min(current, config.compaction_tokens)
    compressor.threshold_tokens = ceiling
    ratio = getattr(compressor, "summary_target_ratio", None)
    if isinstance(ratio, (int, float)) and hasattr(compressor, "tail_token_budget"):
        compressor.tail_token_budget = int(ceiling * ratio)
    return ceiling


def _workspace_evidence(workdir: Optional[str]) -> Mapping[str, Any]:
    if not workdir:
        return {}
    root = Path(workdir)
    if not root.is_dir():
        return {"workdir_present": False}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root, capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        return {"git_head": head, "dirty_digest": hashlib.sha256(dirty.encode()).hexdigest()}
    except (OSError, subprocess.SubprocessError):
        return {"git_unavailable": True}


def boundary_fingerprint(job: Mapping[str, Any], dynamic_input: str) -> str:
    safe = {
        "job_id": str(job.get("id") or ""),
        "objective": str(job.get("prompt") or ""),
        "dynamic_input": str(dynamic_input or ""),
        "workspace": _workspace_evidence(job.get("workdir")),
    }
    canonical = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class GateDecision:
    allow: bool
    fingerprint: str
    dynamic_append: Optional[str] = None
    redesign_requested: bool = False
    blocker_alert: Optional[str] = None
    reason: Optional[str] = None


class OpusRedesignBackend:
    """Private HTTP backend with exactly one primary and one backup attempt."""

    def __init__(
        self,
        config: EfficiencyConfig,
        *,
        http_post: Optional[Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]] = None,
    ) -> None:
        self.config = config
        self.http_post = http_post or self._post

    @staticmethod
    def _post(url: str, body: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
        encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        request = urllib.request.Request(url, data=encoded, headers={"content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured local bridge
            return json.loads(response.read().decode("utf-8"))

    def __call__(self, package: Mapping[str, Any]) -> str:
        canonical = json.dumps(package, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        body = {
            "profile": "review_bundle",
            "project": self.config.redesign_project,
            "candidate_sha": hashlib.sha256(canonical.encode()).hexdigest(),
            "model": "claude-opus-4-8",
            "effort": "max",
            "messages": [{"role": "user", "content": canonical}],
        }
        last_error: Optional[BaseException] = None
        for url in (self.config.redesign_primary_url, self.config.redesign_backup_url):
            try:
                response = self.http_post(url, body, self.config.redesign_timeout_seconds)
                content = response["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("empty redesign response")
                return content.strip()
            except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = exc
        raise RuntimeError(f"Opus redesign unavailable after primary and backup: {type(last_error).__name__}")


class ContextEfficiencyGate:
    def __init__(
        self,
        hermes_home: Path,
        config: EfficiencyConfig,
        *,
        redesign_backend: Optional[Callable[[Mapping[str, Any]], str]] = None,
    ) -> None:
        self.home = Path(hermes_home)
        self.config = config
        self.root = self.home / "cron" / "context-efficiency"
        self.state_path = self.root / "ledger.json"
        self.lock_path = self.root / ".ledger.lock"
        self.redesign_backend = redesign_backend

    @contextlib.contextmanager
    def _locked_state(self):
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        with _PROCESS_LOCK:
            with open(self.lock_path, "a+", encoding="utf-8") as lock_handle:
                os.chmod(self.lock_path, 0o600)
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    if self.state_path.exists():
                        try:
                            state = json.loads(self.state_path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
                            raise EfficiencyStateError("context-efficiency ledger is corrupt or unreadable") from exc
                        if not isinstance(state, dict) or not isinstance(state.get("jobs", {}), dict):
                            raise EfficiencyStateError("context-efficiency ledger has invalid schema")
                    else:
                        state = {"version": _SCHEMA_VERSION, "jobs": {}}
                    yield state
                    _atomic_json(self.state_path, state)
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _job_state(state: dict, job_id: str) -> dict:
        jobs = state.setdefault("jobs", {})
        return jobs.setdefault(job_id, {"failures": [], "records": []})

    def before_run(
        self,
        job: Mapping[str, Any],
        *,
        dynamic_input: str,
        now: Optional[datetime] = None,
    ) -> GateDecision:
        fingerprint = boundary_fingerprint(job, dynamic_input)
        job_id = _safe_job_id(job)
        package: Optional[dict[str, Any]] = None

        # Reserve the single redesign attempt under lock, but never hold the
        # global ledger lock while performing a network call.
        with self._locked_state() as state:
            item = self._job_state(state, job_id)
            if item.get("fingerprint") != fingerprint:
                item.update({"fingerprint": fingerprint, "failures": []})
                item.pop("redesign", None)
                item.pop("blocker", None)
            pending = item.get("redesign") if isinstance(item.get("redesign"), dict) else None
            if pending and pending.get("result") and not pending.get("consumed"):
                pending["consumed"] = True
                return GateDecision(True, fingerprint, dynamic_append=str(pending["result"]))
            failures = list(item.get("failures") or [])
            if len(failures) < 2:
                return GateDecision(True, fingerprint)
            if pending:
                return GateDecision(False, fingerprint, reason="failure_streak_blocked")
            package = {
                "objective": str(job.get("prompt") or ""),
                "dynamic_input": str(dynamic_input or ""),
                "failures": [
                    {
                        "error": str(row.get("error") or "")[:2000],
                        "attempted": str(row.get("attempted") or "")[:2000],
                    }
                    for row in failures[-2:]
                ],
            }
            item["redesign"] = {"attempted": True, "status": "in_progress", "at": _iso(now)}

        backend = self.redesign_backend or OpusRedesignBackend(self.config)
        try:
            result = backend(package)
        except Exception as exc:
            with self._locked_state() as state:
                item = self._job_state(state, job_id)
                if item.get("fingerprint") == fingerprint:
                    item["redesign"] = {
                        "attempted": True,
                        "status": "failed",
                        "failed": type(exc).__name__,
                        "at": _iso(now),
                    }
                    self._record_blocker_item(item, "Opus redesign unavailable", fingerprint, now)
            return GateDecision(False, fingerprint, redesign_requested=True, reason="redesign_unavailable")

        with self._locked_state() as state:
            item = self._job_state(state, job_id)
            if item.get("fingerprint") != fingerprint:
                return GateDecision(False, fingerprint, reason="fingerprint_changed_during_redesign")
            item["redesign"] = {
                "attempted": True,
                "status": "prepared",
                "result": result,
                "consumed": False,
                "at": _iso(now),
            }
        return GateDecision(False, fingerprint, redesign_requested=True, reason="redesign_prepared")

    def record_outcome(
        self,
        job: Mapping[str, Any],
        fingerprint: str,
        outcome: str,
        *,
        error: str = "",
        attempted: str = "",
        material: bool = False,
        usage: Optional[Mapping[str, Any]] = None,
        duration_seconds: Optional[float] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> None:
        allowed = {"success", "failure", "valid_noop", "blocker", "material_outcome"}
        if outcome not in allowed:
            raise ValueError(f"unsupported context-efficiency outcome: {outcome}")
        job_id = _safe_job_id(job)
        with self._locked_state() as state:
            item = self._job_state(state, job_id)
            if item.get("fingerprint") != fingerprint:
                item.update({"fingerprint": fingerprint, "failures": []})
                item.pop("redesign", None)
            if outcome == "failure":
                item.setdefault("failures", []).append({
                    "error": str(error)[:4000], "attempted": str(attempted)[:4000], "at": _iso(now),
                })
                item["failures"] = item["failures"][-2:]
            elif material or outcome == "material_outcome":
                item["failures"] = []
                item.pop("redesign", None)
                item.pop("blocker", None)
            row: dict[str, Any] = {
                "at": _iso(now), "fingerprint": fingerprint, "outcome": outcome,
                "material": bool(material or outcome == "material_outcome"),
            }
            if usage is not None:
                row["usage"] = dict(usage)
            if duration_seconds is not None:
                row["duration_seconds"] = duration_seconds
            if model:
                row["model"] = model
            if provider:
                row["provider"] = provider
            item.setdefault("records", []).append(row)
            item["records"] = item["records"][-1000:]

    @staticmethod
    def _record_blocker_item(item: dict, summary: str, fingerprint: str, now: Optional[datetime]) -> None:
        current = item.get("blocker") if isinstance(item.get("blocker"), dict) else None
        if current and current.get("fingerprint") == fingerprint:
            current["summary"] = str(summary)[:1000]
            return
        item["blocker"] = {
            "fingerprint": fingerprint, "summary": str(summary)[:1000], "since": _iso(now), "alerted": False,
        }

    def record_blocker(
        self, job: Mapping[str, Any], summary: str, fingerprint: str, *, now: Optional[datetime] = None
    ) -> None:
        with self._locked_state() as state:
            item = self._job_state(state, _safe_job_id(job))
            self._record_blocker_item(item, summary, fingerprint, now)

    def blocker_alert(
        self, job: Mapping[str, Any], fingerprint: str, *, now: Optional[datetime] = None
    ) -> Optional[str]:
        current_time = now or _utc_now()
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        with self._locked_state() as state:
            item = self._job_state(state, _safe_job_id(job))
            blocker = item.get("blocker") if isinstance(item.get("blocker"), dict) else None
            if not blocker or blocker.get("fingerprint") != fingerprint or blocker.get("alerted"):
                return None
            age_hours = (current_time.astimezone(timezone.utc) - _parse_time(blocker["since"])).total_seconds() / 3600
            if age_hours <= self.config.blocker_alert_hours:
                return None
            blocker["alerted"] = True
            name = str(job.get("name") or job.get("id") or "cron")
            return f"⚠ Cron blocker envejecido: {name} lleva {age_hours:.1f}h bloqueado — {blocker.get('summary', '')}"

    def weekly_records(self, job_id: str) -> list[dict[str, Any]]:
        with self._locked_state() as state:
            item = self._job_state(state, _safe_job_id({"id": job_id}))
            return list(item.get("records") or [])


GEMINI_COMPRESSION_PROVIDER = _GEMINI_PROVIDER
GEMINI_COMPRESSION_MODEL = _GEMINI_MODEL
