"""Mandatory Linear issue linking for network dispatch.

A job is not dispatchable without a Linear identifier. This module parses
and normalizes issue ids; it does not put API tokens into prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse


_ISSUE_RE = re.compile(r"^[A-Z][A-Z0-9]{0,10}-\d+$")
_ISSUE_IN_TEXT_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9]{0,10}-\d+)\b")
_LINEAR_HOSTS = frozenset({"linear.app", "www.linear.app"})


class LinearLinkError(ValueError):
    """Raised when a Linear issue id is missing or malformed."""


@dataclass(frozen=True)
class LinearLink:
    """A required Linear issue binding for a dispatched job."""

    identifier: str
    url: str
    team_key: str = "NAI"

    def to_dict(self) -> dict:
        return {
            "identifier": self.identifier,
            "url": self.url,
            "team_key": self.team_key,
        }


def parse_linear_issue(raw: str, *, default_team: str = "NAI") -> str:
    """Return a normalized Linear identifier (e.g. ``NAI-68``).

    Accepts a bare id, a Linear URL, or text containing exactly one id.
    """
    text = (raw or "").strip()
    if not text:
        raise LinearLinkError("Linear issue is required")

    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"}:
        host = parsed.netloc.lower()
        if host not in _LINEAR_HOSTS:
            raise LinearLinkError(f"invalid Linear issue {raw!r}")
        parts = [p for p in parsed.path.split("/") if p]
        # /<workspace>/issue/<ID>/optional-slug
        if "issue" in parts:
            idx = parts.index("issue")
            if idx + 1 < len(parts):
                text = parts[idx + 1]
        else:
            raise LinearLinkError(f"invalid Linear issue {raw!r}")

    candidate = text.strip().upper()
    if _ISSUE_RE.match(candidate):
        return candidate

    matches = _ISSUE_IN_TEXT_RE.findall(text)
    unique = {m.upper() for m in matches if _ISSUE_RE.match(m.upper())}
    if len(unique) == 1:
        return next(iter(unique))

    raise LinearLinkError(f"invalid Linear issue {raw!r}")


def linear_url(identifier: str, *, workspace: str = "naicipa") -> str:
    """Return the canonical Linear issue URL for ``identifier``."""
    return f"https://linear.app/{workspace}/issue/{identifier}"


def require_linear_issue(
    raw: Optional[str],
    *,
    workspace: str = "naicipa",
    default_team: str = "NAI",
) -> LinearLink:
    """Parse ``raw`` or raise :class:`LinearLinkError`."""
    identifier = parse_linear_issue(raw or "", default_team=default_team)
    team_key = identifier.split("-", 1)[0]
    return LinearLink(
        identifier=identifier,
        url=linear_url(identifier, workspace=workspace),
        team_key=team_key,
    )
