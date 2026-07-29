"""Fail-closed execution envelopes for the local Claude Agent SDK bridge.

The bridge on 127.0.0.1:4318/4319 rejects unprofiled requests.  Keep the
validation client-side so dispatcher mistakes fail before an HTTP request or
subagent spawn.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

CLAUDE_EXECUTION_PROFILES = ("probe", "review_bundle", "code")
CLAUDE_BRIDGE_PROVIDERS = frozenset({"claude-sdk-local", "claude-sdk-team-local"})
_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")


class ClaudeExecutionProfileError(ValueError):
    """Raised before dispatch when the Claude bridge envelope is invalid."""


def is_claude_sdk_bridge(*, provider: Any = None, base_url: Any = None) -> bool:
    if str(provider or "").strip().lower() in CLAUDE_BRIDGE_PROVIDERS:
        return True
    raw = str(base_url or "").strip()
    if not raw:
        return False
    try:
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
        return host in {"127.0.0.1", "localhost", "::1"} and parsed.port in {4318, 4319}
    except ValueError:
        return False


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ClaudeExecutionProfileError(f"{field} is required for this execution profile")
    return text


def _string_list(value: Any, field: str, *, maximum: int, max_chars: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > maximum:
        raise ClaudeExecutionProfileError(f"{field} must be a list with at most {maximum} entries")
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text or len(text) > max_chars:
            raise ClaudeExecutionProfileError(
                f"{field} entries must be non-empty strings of at most {max_chars} characters"
            )
        result.append(text)
    return result


def normalize_claude_execution_envelope(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical top-level body fields accepted by the bridge."""
    profile = str(values.get("execution_profile") or values.get("profile") or "").strip()
    if not profile:
        raise ClaudeExecutionProfileError("execution_profile is required for the Claude SDK bridge")
    if profile not in CLAUDE_EXECUTION_PROFILES:
        allowed = "|".join(CLAUDE_EXECUTION_PROFILES)
        raise ClaudeExecutionProfileError(f"execution_profile must be one of {allowed}")

    if profile == "probe":
        return {"profile": profile}

    project = _required_text(values.get("project"), "project")
    candidate_sha = _required_text(values.get("candidate_sha"), "candidate_sha")
    if not _SHA_RE.fullmatch(candidate_sha):
        raise ClaudeExecutionProfileError("candidate_sha must be a 40- or 64-character hexadecimal Git SHA")

    if profile == "review_bundle":
        if values.get("cwd") or values.get("skills") or values.get("rules"):
            raise ClaudeExecutionProfileError(
                "review_bundle does not accept cwd, skills, or rules"
            )
        return {
            "profile": profile,
            "project": project,
            "candidate_sha": candidate_sha.lower(),
        }

    cwd_raw = _required_text(values.get("cwd"), "cwd")
    cwd = Path(cwd_raw).expanduser().resolve()
    if not cwd.is_dir():
        raise ClaudeExecutionProfileError("cwd must be an existing Git worktree root")
    try:
        git_root = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ClaudeExecutionProfileError("cwd must be an existing Git worktree root") from exc
    if Path(git_root).resolve() != cwd:
        raise ClaudeExecutionProfileError("cwd must be the exact Git worktree root")
    if head.lower() != candidate_sha.lower():
        raise ClaudeExecutionProfileError("candidate_sha must equal HEAD of cwd")

    return {
        "profile": profile,
        "project": project,
        "candidate_sha": candidate_sha.lower(),
        "cwd": str(cwd),
        "skills": _string_list(values.get("skills"), "skills", maximum=12, max_chars=128),
        "rules": _string_list(values.get("rules"), "rules", maximum=20, max_chars=500),
    }


def require_claude_execution_profile(
    *, provider: Any = None, base_url: Any = None, request_overrides: Any = None
) -> dict[str, Any]:
    """Validate a bridge request and return its canonical profile envelope.

    Non-bridge routes return an empty mapping and are deliberately unaffected.
    """
    if not is_claude_sdk_bridge(provider=provider, base_url=base_url):
        return {}
    values = request_overrides if isinstance(request_overrides, Mapping) else {}
    return normalize_claude_execution_envelope(values)
