"""Brokered 1Password credential access for isolated profiles.

Profiles request named secrets through this broker. The broker fetches via
the existing 1Password secret source, records an audit row, and returns a
receipt that never contains secret values. Granted values stay in a
process-local map used only to populate a child process environment —
they are never written to prompts, memory, job files, or receipts.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence

from hermes_cli.independent_network.store import append_jsonl, audit_dir, network_root, write_json


Fetcher = Callable[[str, str], str]
# (op_reference, env_name) -> secret value


class SecretRevealedError(RuntimeError):
    """Raised when a receipt, audit row, or job payload would contain a secret."""


_OP_REF_RE = re.compile(r"^op://[A-Za-z0-9._-]+/.+$")
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")

# Catalog of named secrets the fleet may request. Values are 1Password
# *references*, never the credentials themselves.
DEFAULT_REFERENCES: Dict[str, str] = {
    "LINEAR_API_KEY": "op://Naicipa/Linear/credential",
    "OPENAI_API_KEY": "op://Naicipa/OpenAI/credential",
    "ANTHROPIC_API_KEY": "op://Naicipa/Anthropic/credential",
    "XAI_API_KEY": "op://Naicipa/xAI/credential",
}

def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _request_id() -> str:
    return uuid.uuid4().hex


def is_op_reference(value: str) -> bool:
    return bool(_OP_REF_RE.match((value or "").strip()))


def assert_no_secret_values(payload: object, secrets: Sequence[str]) -> None:
    """Raise if any granted secret value appears in a serializable payload."""
    if not secrets:
        return
    blob = json.dumps(payload, default=str)
    for secret in secrets:
        if secret and secret in blob:
            raise SecretRevealedError("secret value leaked into broker payload")


@dataclass(frozen=True)
class CredentialReceipt:
    """Public record of a credential request. Contains no secret values."""

    request_id: str
    profile: str
    secret_name: str
    granted: bool
    source: str
    reference: str
    timestamp: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "profile": self.profile,
            "secret_name": self.secret_name,
            "granted": self.granted,
            "source": self.source,
            "reference": self.reference,
            "timestamp": self.timestamp,
            "error": self.error,
        }


@dataclass
class CredentialBroker:
    """On-demand 1Password broker with scope + audit and no value leakage."""

    home: Optional[Path] = None
    references: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_REFERENCES))
    allowlist: Optional[Mapping[str, Sequence[str]]] = None
    fetcher: Optional[Fetcher] = None
    source: str = "onepassword"
    _grants: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def catalog(self) -> Dict[str, str]:
        """Return env-name → op:// reference mappings (no secret values)."""
        return {name: ref for name, ref in self.references.items() if is_op_reference(ref)}

    def allowed_names(self, profile: str) -> frozenset[str]:
        catalog = set(self.catalog())
        if self.allowlist is None:
            return frozenset(catalog)
        scoped = self.allowlist.get(profile) or self.allowlist.get(profile.lower())
        if scoped is None:
            return frozenset(catalog)
        return frozenset(n for n in scoped if n in catalog)

    def request(self, profile: str, secret_name: str) -> CredentialReceipt:
        """Request a named secret for ``profile``.

        On success the value is stored only in the process-local grant map
        under ``request_id``. The returned receipt never includes it.
        """
        name = (secret_name or "").strip()
        profile_id = (profile or "").strip().lower()
        request_id = _request_id()
        timestamp = _utcnow()

        if not profile_id:
            return self._finish(
                CredentialReceipt(
                    request_id=request_id,
                    profile=profile_id,
                    secret_name=name,
                    granted=False,
                    source=self.source,
                    reference="",
                    timestamp=timestamp,
                    error="profile is required",
                )
            )
        if not _ENV_NAME_RE.match(name):
            return self._finish(
                CredentialReceipt(
                    request_id=request_id,
                    profile=profile_id,
                    secret_name=name,
                    granted=False,
                    source=self.source,
                    reference="",
                    timestamp=timestamp,
                    error="invalid secret name",
                )
            )
        if name not in self.allowed_names(profile_id):
            return self._finish(
                CredentialReceipt(
                    request_id=request_id,
                    profile=profile_id,
                    secret_name=name,
                    granted=False,
                    source=self.source,
                    reference=self.catalog().get(name, ""),
                    timestamp=timestamp,
                    error="secret not in profile scope",
                )
            )

        reference = self.catalog()[name]
        try:
            value = self._fetch(reference, name)
        except Exception as exc:
            return self._finish(
                CredentialReceipt(
                    request_id=request_id,
                    profile=profile_id,
                    secret_name=name,
                    granted=False,
                    source=self.source,
                    reference=reference,
                    timestamp=timestamp,
                    error=_safe_error(exc),
                )
            )

        if not isinstance(value, str) or not value:
            return self._finish(
                CredentialReceipt(
                    request_id=request_id,
                    profile=profile_id,
                    secret_name=name,
                    granted=False,
                    source=self.source,
                    reference=reference,
                    timestamp=timestamp,
                    error="empty credential",
                )
            )

        self._grants[request_id] = value
        receipt = CredentialReceipt(
            request_id=request_id,
            profile=profile_id,
            secret_name=name,
            granted=True,
            source=self.source,
            reference=reference,
            timestamp=timestamp,
        )
        self._finish(receipt, secrets=(value,))
        return receipt

    def take_grant(self, request_id: str) -> Optional[str]:
        """Pop a granted value out of process memory. Not for serialization."""
        return self._grants.pop(request_id, None)

    def peek_grant(self, request_id: str) -> Optional[str]:
        """Read a granted value without removing it. Not for serialization."""
        return self._grants.get(request_id)

    def collect_for_profile(self, profile: str) -> tuple[Dict[str, str], list[CredentialReceipt]]:
        """Request every in-scope secret and return env overlay + receipts.

        The env overlay is for the child process only. Receipts stay
        value-free. Failed requests are audited and omitted from the overlay.
        """
        env: Dict[str, str] = {}
        receipts: list[CredentialReceipt] = []
        for name in sorted(self.allowed_names(profile)):
            receipt = self.request(profile, name)
            receipts.append(receipt)
            if receipt.granted:
                value = self.peek_grant(receipt.request_id)
                if value:
                    env[name] = value
        return env, receipts

    def _fetch(self, reference: str, secret_name: str) -> str:
        if self.fetcher is not None:
            return self.fetcher(reference, secret_name)
        return _default_onepassword_fetch(reference, secret_name, home=self.home)

    def _finish(
        self,
        receipt: CredentialReceipt,
        *,
        secrets: Sequence[str] = (),
    ) -> CredentialReceipt:
        payload = receipt.to_dict()
        assert_no_secret_values(payload, secrets)
        path = audit_dir(self.home) / "credentials.jsonl"
        append_jsonl(path, payload)
        return receipt


def _safe_error(exc: BaseException) -> str:
    """Surface a coarse error kind without echoing possible secret material."""
    name = type(exc).__name__
    return f"{name}: unavailable"


def _default_onepassword_fetch(
    reference: str,
    secret_name: str,
    *,
    home: Optional[Path],
) -> str:
    """Resolve one ``op://`` reference through the existing 1Password source."""
    from agent.secret_sources.onepassword import fetch_onepassword_secrets

    secrets, warnings = fetch_onepassword_secrets(
        references={secret_name: reference},
        use_cache=False,
        home_path=home,
    )
    value = secrets.get(secret_name)
    if not value:
        detail = "; ".join(warnings) if warnings else "not granted"
        raise RuntimeError(detail)
    return value


def write_default_catalog(home: Optional[Path] = None) -> Path:
    """Persist the reference catalog (references only) next to the job store."""
    path = network_root(home) / "credentials.yaml"
    payload = {
        "source": "onepassword",
        "references": dict(DEFAULT_REFERENCES),
    }
    write_json(path.with_suffix(".json"), payload)
    return path.with_suffix(".json")
