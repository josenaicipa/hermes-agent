"""Helpers for reading the effective fallback provider chain from config.

This module is also the single owner of one fallback-entry *policy* question:
**which model does a chain entry actually request?**  By default it is the
entry's own ``model`` — but an entry may opt in to keeping the model the
caller asked for while only the provider / endpoint changes.

Why the opt-in exists (live V2.9 routing incident): the primary route ran
``claude-opus-5`` on one local Claude bridge, and the inherited chain entry
for the sibling bridge was configured with a cheaper ``model``.  When the
primary reported exhausted credits, Hermes activated that entry verbatim, so
the sibling bridge spawned its child process with the *cheaper* model — a
silent downgrade of the model the caller requested.  Chains that never opt in
keep the historical "configured model wins" behavior exactly.

The policy is deliberately generic: no provider label, port, or model name is
special cased.  Anything that needs the model an entry will actually request
must go through :func:`resolve_fallback_model` /
:func:`effective_fallback_entry` instead of reading ``entry["model"]``, and
anything that could rewrite a preserved model must gate on
:func:`require_preserved_model` so a substitution fails closed rather than
running a different model behind the caller's back.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

#: Per-entry opt-in: keep the requested model, swap only provider/base_url.
PRESERVE_REQUESTED_MODEL_KEY = "preserve_requested_model"

_OPT_IN_TRUE = frozenset({"1", "true", "yes", "on", "enabled"})
_OPT_IN_FALSE = frozenset({"0", "false", "no", "off", "disabled"})


class FallbackModelPolicyError(ValueError):
    """Raised when an entry's model policy cannot be honored exactly.

    Callers must treat this as "skip this entry" — never as "use the
    configured model instead".  Silently substituting a different model is
    the exact defect this policy exists to prevent.
    """


def entry_preserves_requested_model(entry: Any) -> bool:
    """Return the strict opt-in state of one fallback entry.

    Missing key → ``False`` (every pre-existing chain keeps its behavior).
    Recognized booleans / YAML-ish strings decide.  Anything else raises
    :class:`FallbackModelPolicyError`: an operator who typo'd a
    routing-safety flag must get a loud skip, not a silent downgrade.
    """
    if not isinstance(entry, Mapping) or PRESERVE_REQUESTED_MODEL_KEY not in entry:
        return False
    value = entry[PRESERVE_REQUESTED_MODEL_KEY]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _OPT_IN_TRUE:
            return True
        if lowered in _OPT_IN_FALSE:
            return False
    raise FallbackModelPolicyError(
        f"{PRESERVE_REQUESTED_MODEL_KEY} must be a boolean "
        f"(true/false); got {value!r}"
    )


@dataclass(frozen=True)
class FallbackModelDecision:
    """The model one fallback entry will request, and why.

    ``model`` is the effective model — the only value downstream code may
    use.  ``configured_model`` is kept for diagnostics so logs can show what
    the entry declared, and ``preserved`` marks entries whose model must not
    be rewritten later.
    """

    model: str
    configured_model: str
    preserved: bool


def resolve_fallback_model(entry: Any, *, requested_model: Any) -> FallbackModelDecision:
    """Decide which model ``entry`` requests, honoring the opt-in.

    ``requested_model`` is the model the caller actually asked for (the
    primary route's model), passed in by the caller so this module stays
    free of agent-runtime knowledge.

    Raises:
        FallbackModelPolicyError: invalid opt-in value, or opt-in with no
            requested model to preserve.  Both must fail closed.
    """
    configured = ""
    if isinstance(entry, Mapping):
        configured = str(entry.get("model") or "").strip()
    if not entry_preserves_requested_model(entry):
        return FallbackModelDecision(
            model=configured, configured_model=configured, preserved=False
        )
    preserved = str(requested_model or "").strip()
    if not preserved:
        raise FallbackModelPolicyError(
            f"{PRESERVE_REQUESTED_MODEL_KEY} needs a requested model to "
            "preserve, but none is known for this route"
        )
    return FallbackModelDecision(
        model=preserved, configured_model=configured, preserved=True
    )


def effective_fallback_entry(
    entry: Mapping[str, Any], decision: FallbackModelDecision
) -> dict[str, Any]:
    """Return a copy of ``entry`` whose ``model`` is the effective model.

    Every entry-shaped consumer (dedup/skip keys, local availability checks,
    backend identity, credential hints, logging) should be handed this copy
    so they all agree on the model that will be requested.  The source entry
    is never mutated: chains are shared with delegate children, and mutating
    one would rewrite a sibling agent's routing policy.
    """
    effective = dict(entry)
    effective["model"] = decision.model
    return effective


def _canonical_model_key(value: Any) -> str:
    """Fold formatting-only differences (vendor prefix, dot/hyphen) away."""
    text = str(value or "").strip().lower()
    if "/" in text:
        text = text.rsplit("/", 1)[1].strip()
    return text.replace(".", "-")


def require_preserved_model(
    decision: FallbackModelDecision, actual_model: Any, *, source: str
) -> None:
    """Fail closed when a preserved model was rewritten into another model.

    Provider routers and per-provider normalizers legitimately *reformat* a
    slug (``claude-opus-5`` → ``anthropic/claude-opus-5``); that is accepted.
    Landing on a different model is a substitution and raises, so the caller
    skips the entry instead of quietly running the wrong model.  Entries that
    did not opt in are never constrained.
    """
    if not decision.preserved:
        return
    if _canonical_model_key(actual_model) == _canonical_model_key(decision.model):
        return
    raise FallbackModelPolicyError(
        f"{source} cannot serve the preserved model {decision.model!r} "
        f"without substituting {actual_model!r}"
    )


def _normalized_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def resolve_entry_api_key(entry: dict[str, Any] | None) -> str | None:
    """API key for one fallback entry: inline ``api_key``, else ``key_env``.

    Mirrors the custom-provider convention (``key_env`` names the env var
    holding the key; ``api_key_env`` accepted as an alias). Returns None when
    neither yields a non-empty value, letting ``resolve_runtime_provider``
    fall through to the provider's standard credential resolution.
    """
    if not isinstance(entry, dict):
        return None
    inline = str(entry.get("api_key") or "").strip()
    if inline:
        return inline
    key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
    if key_env:
        return os.getenv(key_env, "").strip() or None
    return None


def _iter_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        return []

    entries: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue

        normalized = dict(entry)
        normalized["provider"] = provider
        normalized["model"] = model

        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url

        entries.append(normalized)
    return entries


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str, bool]:
    # The model policy is part of the route identity: the same
    # provider/model/base_url with and without the preserve opt-in are two
    # different routes (one pinned to the entry's model, one to the caller's),
    # so deduping them together would silently drop one of the operator's
    # declared hops.  An invalid opt-in value must not break chain loading —
    # activation is where it fails closed — so it is treated as "not opted in"
    # for identity purposes only.
    try:
        preserves = entry_preserves_requested_model(entry)
    except FallbackModelPolicyError:
        preserves = False
    return (
        str(entry.get("provider") or "").strip().lower(),
        str(entry.get("model") or "").strip().lower(),
        _normalized_base_url(entry.get("base_url")).lower(),
        preserves,
    )


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its
    order. Legacy ``fallback_model`` entries are appended afterwards unless
    they target the same provider/model/base_url route (and the same model
    policy) as an earlier entry. Every key an entry declares — including
    ``preserve_requested_model`` — is carried through untouched.
    The returned list always contains fresh dict copies.
    """

    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, bool]] = set()

    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key)):
            identity = _entry_identity(entry)
            if identity in seen:
                continue
            seen.add(identity)
            chain.append(entry)

    return chain
