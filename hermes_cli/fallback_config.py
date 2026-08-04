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

Three rules make this flag safe to own here, and all three are strict:

1. **Only an actual YAML boolean opts in.**  No truthy coercion — not
   ``"yes"``, not ``1``, not ``2``, never a collection or an empty value.
   Unquoted ``true``/``yes``/``on`` still work because YAML 1.1 itself
   resolves them to booleans; the flag adds no coercion of its own.  A flag
   that decides *which model runs* must not be guessable.
2. **Only an eligible model is preserved.**  An entry declaring a model of a
   different family (or an explicit ``models:`` list without the model) can
   never legitimately answer the request, so preserving is refused instead of
   sending a Claude request to a GPT endpoint (see
   :func:`model_family_key`).
3. **Only inside a fallback entry does the flag exist.**  Written anywhere
   else it is a silent no-op that leaves the mirror entry downgrading models,
   so :func:`preserve_requested_model_issues` reports the misplacement (it
   feeds ``validate_config_structure`` / ``hermes doctor``).

Diagnostics from this module are printed at startup, written to logs and
pasted into bug reports, so they name paths, providers and model ids — never
credentials, endpoints, or raw operator-supplied values.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

#: Per-entry opt-in: keep the requested model, swap only provider/base_url.
PRESERVE_REQUESTED_MODEL_KEY = "preserve_requested_model"

#: Copy-pasteable fix shown with every rejected opt-in.  Deliberately free of
#: user-supplied text so it can never carry a secret into a log line.
PRESERVE_REQUESTED_MODEL_HINT = (
    "Write it as an unquoted YAML boolean on the fallback entry itself:\n"
    "  fallback_providers:\n"
    "    - provider: my-bridge-b\n"
    "      model: claude-opus-5\n"
    "      base_url: https://<mirror-endpoint>/v1\n"
    f"      {PRESERVE_REQUESTED_MODEL_KEY}: true\n"
    "Quoted values ('true', \"yes\"), numbers, empty values and lists are "
    "refused. Remove the key to keep the entry's configured model."
)

#: Words operators reach for instead of a boolean.  Echoing one back is safe
#: (closed set, cannot be a secret) and makes the fix obvious.
_BOOLEANISH_WORDS = frozenset({
    "true", "false", "yes", "no", "on", "off", "1", "0",
    "y", "n", "t", "f", "enabled", "disabled", "none", "null",
})

#: Config locations that actually read the flag.  Kept next to the walker that
#: enforces it so "where is this legal?" has exactly one answer.
_MAIN_CHAIN_ROOT_KEYS = ("fallback_providers", "fallback_model")
_AUXILIARY_CHAIN_KEY = "fallback_chain"

# Model ids separate their tokens with -, ., /, :, _ and @ depending on vendor.
_MODEL_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

# Config trees are shallow; the bound only exists so a pathological/aliased
# document cannot turn a startup warning pass into a hang.
_MAX_CONFIG_SCAN_DEPTH = 12


class FallbackModelPolicyError(ValueError):
    """Raised when an entry's model policy cannot be honored exactly.

    Callers must treat this as "skip this entry" — never as "use the
    configured model instead".  Silently substituting a different model is
    the exact defect this policy exists to prevent.
    """


def describe_rejected_opt_in_value(value: Any) -> str:
    """Describe a rejected opt-in value without echoing operator input.

    Operators do paste tokens into the wrong key, and these strings end up in
    logs and issue reports, so the value itself is never reflected: the reader
    gets its YAML *shape*, plus the literal word when it comes from the closed
    set of boolean-ish spellings.
    """
    if value is None:
        return "an empty value (YAML null)"
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _BOOLEANISH_WORDS:
            return f"the quoted string {word!r}"
        return "a string"
    if isinstance(value, Mapping):
        return "a mapping"
    if isinstance(value, (list, tuple, set, frozenset)):
        return "a list"
    if isinstance(value, float):
        return "a number"
    if isinstance(value, int):  # bools already returned above
        return "an integer"
    return f"a {type(value).__name__} value"


def validate_preserve_requested_model(value: Any, *, where: str = "") -> bool:
    """Return the boolean meaning of one opt-in value, or fail closed.

    ``where`` is a config path used only for diagnostics (never a value).
    """
    if value is True:
        return True
    if value is False:
        return False
    location = f" at {where}" if where else ""
    raise FallbackModelPolicyError(
        f"{PRESERVE_REQUESTED_MODEL_KEY}{location} must be an unquoted YAML "
        f"boolean (true or false), not {describe_rejected_opt_in_value(value)} "
        "— the entry is skipped instead of silently requesting a different "
        "model"
    )


def entry_preserves_requested_model(entry: Any, *, where: str = "") -> bool:
    """Return the strict opt-in state of one fallback entry.

    Missing key → ``False`` (every pre-existing chain keeps its behavior).
    Only a real YAML boolean decides; anything else raises
    :class:`FallbackModelPolicyError`, because an operator who typo'd a
    routing-safety flag must get a loud skip, not a silent downgrade.
    """
    if not isinstance(entry, Mapping) or PRESERVE_REQUESTED_MODEL_KEY not in entry:
        return False
    return validate_preserve_requested_model(
        entry[PRESERVE_REQUESTED_MODEL_KEY], where=where
    )


def _canonical_model_key(value: Any) -> str:
    """Fold formatting-only differences (vendor prefix, dot/hyphen) away."""
    text = str(value or "").strip().lower()
    if "/" in text:
        text = text.rsplit("/", 1)[1].strip()
    return text.replace(".", "-")


def model_family_key(value: Any) -> str:
    """The model's family token (``claude``, ``gpt``, ``gemini``, …) or ``""``.

    The leading identifier token of a model id is its family on every provider
    catalog Hermes talks to (``claude-opus-5``, ``anthropic/claude-opus-5``,
    ``openai/gpt-5.4``, ``gemini-3-pro``, ``glm-4.7``, ``qwen3:8b``), so this
    stays vendor-agnostic: nothing here enumerates families, and an id with no
    alphabetic token (``"1234"``, ``"-"``) has no family at all — which is the
    signal that it is not a usable wire id.
    """
    for token in _MODEL_TOKEN_SPLIT.split(_canonical_model_key(value)):
        if token and any(ch.isalpha() for ch in token):
            return token
    return ""


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


def _entry_declared_models(entry: Any) -> set[str]:
    """Canonical keys of the models an entry says its endpoint serves.

    ``models:`` is the documented per-provider shape (list, or the mapping
    form used for per-model settings).  Empty set = "the entry makes no claim",
    which is not the same as "serves nothing".
    """
    if not isinstance(entry, Mapping):
        return set()
    raw = entry.get("models")
    if isinstance(raw, Mapping):
        candidates: Sequence[Any] = list(raw.keys())
    elif isinstance(raw, (list, tuple, set, frozenset)):
        candidates = list(raw)
    else:
        return set()
    keys = {_canonical_model_key(c) for c in candidates if isinstance(c, str)}
    return {key for key in keys if key}


def _require_eligible_preserved_model(
    entry: Any, preserved: str, configured: str
) -> None:
    """Refuse to preserve a model the entry could not honestly serve.

    Two failure classes, both fail closed so the chain walks on:

    * **unsupported** — the model id carries no recognizable family (so it is
      not a usable wire id), or the entry declares an explicit ``models:``
      allowlist that does not contain it;
    * **cross-family** — the entry declares a model from another family, i.e.
      it is not a mirror/sibling endpoint of the requested model at all.
      Preserving there would aim a Claude request at a GPT deployment.

    An entry that declares no model makes no family claim (auxiliary chains
    legitimately omit it), so it is allowed: the operator's opt-in is the only
    claim available, and any substitution is still caught downstream by
    :func:`require_preserved_model`.
    """
    family = model_family_key(preserved)
    if not family:
        raise FallbackModelPolicyError(
            f"{PRESERVE_REQUESTED_MODEL_KEY} cannot preserve the unsupported "
            f"model id {preserved!r}: it carries no recognizable model family"
        )
    entry_family = model_family_key(configured)
    if entry_family and entry_family != family:
        raise FallbackModelPolicyError(
            f"{PRESERVE_REQUESTED_MODEL_KEY} cannot preserve {preserved!r} on "
            f"an entry declaring {configured!r}: cross-family substitution "
            f"({family} → {entry_family}) is refused"
        )
    declared = _entry_declared_models(entry)
    if declared and _canonical_model_key(preserved) not in declared:
        raise FallbackModelPolicyError(
            f"{PRESERVE_REQUESTED_MODEL_KEY} cannot preserve the unsupported "
            f"model {preserved!r}: this entry only declares "
            f"{sorted(declared)}"
        )


def resolve_fallback_model(entry: Any, *, requested_model: Any) -> FallbackModelDecision:
    """Decide which model ``entry`` requests, honoring the opt-in.

    ``requested_model`` is the model the caller actually asked for (the
    primary route's model for the main agent, the model the failing auxiliary
    call was running for aux tasks), passed in by the caller so this module
    stays free of agent-runtime knowledge.

    Raises:
        FallbackModelPolicyError: invalid opt-in value, opt-in with no
            requested model to preserve, or an opt-in whose requested model is
            not eligible for the entry (cross-family / unsupported).  All fail
            closed — the caller skips the entry and never falls back to the
            configured model.
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
    _require_eligible_preserved_model(entry, preserved, configured)
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


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str, Any]:
    # The model policy is part of the route identity: the same
    # provider/model/base_url with and without the preserve opt-in are two
    # different routes (one pinned to the entry's model, one to the caller's),
    # so deduping them together would silently drop one of the operator's
    # declared hops.  An invalid opt-in value must not break chain loading
    # (activation is where it fails closed), and it gets its own third identity
    # rather than being folded into "not opted in": collapsing it into a valid
    # plain twin would delete the broken entry before anything could report it,
    # leaving the operator with a silent downgrade and no log line.
    try:
        preserves: Any = entry_preserves_requested_model(entry)
    except FallbackModelPolicyError:
        preserves = None
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
    ``preserve_requested_model`` — is carried through untouched, *verbatim*:
    loading never repairs or drops a malformed opt-in, because the operator has
    to be able to see the broken entry (``hermes fallback list``,
    ``hermes doctor``).  Failing closed happens where the model is decided
    (:func:`resolve_fallback_model`), not here.
    """

    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, Any]] = set()

    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key)):
            identity = _entry_identity(entry)
            if identity in seen:
                continue
            seen.add(identity)
            chain.append(entry)

    return chain


# ── Config-time diagnostics: value, placement and use ─────────────────────


@dataclass(frozen=True)
class PreserveModelIssue:
    """One config problem with ``preserve_requested_model``.

    ``path`` is the YAML location (``fallback_providers[0]``), ``message``
    says what is wrong and ``hint`` how to fix it.  All three are safe to
    print: they carry paths, provider labels and model ids — never
    credentials, endpoints or raw operator-supplied values.
    """

    path: str
    message: str
    hint: str


def _format_config_path(path: tuple[Any, ...]) -> str:
    out = ""
    for part in path:
        if isinstance(part, int):
            out += f"[{part}]"
        elif out:
            out += f".{part}"
        else:
            out = str(part)
    return out or "the config root"


def _iter_key_locations(
    node: Any,
    key: str,
    *,
    path: tuple[Any, ...] = (),
    depth: int = 0,
    seen: set[int] | None = None,
) -> Iterator[tuple[tuple[Any, ...], Mapping[str, Any]]]:
    """Yield ``(path, mapping)`` for every mapping in ``node`` holding ``key``."""
    if depth > _MAX_CONFIG_SCAN_DEPTH:
        return
    if seen is None:
        seen = set()
    if id(node) in seen:
        return  # YAML anchors can alias the same object into two places
    if isinstance(node, Mapping):
        seen.add(id(node))
        if key in node:
            yield path, node
        for child_key, value in node.items():
            if isinstance(value, (Mapping, list, tuple)):
                yield from _iter_key_locations(
                    value, key,
                    path=path + (str(child_key),), depth=depth + 1, seen=seen,
                )
    elif isinstance(node, (list, tuple)):
        seen.add(id(node))
        for index, item in enumerate(node):
            if isinstance(item, (Mapping, list, tuple)):
                yield from _iter_key_locations(
                    item, key, path=path + (index,), depth=depth + 1, seen=seen,
                )


def _is_main_chain_entry_path(path: tuple[Any, ...]) -> bool:
    # Both root keys accept a single dict as well as a list of dicts (see
    # _iter_fallback_entries), so both shapes are legal locations.
    if len(path) == 1 and path[0] in _MAIN_CHAIN_ROOT_KEYS:
        return True
    return (
        len(path) == 2
        and path[0] in _MAIN_CHAIN_ROOT_KEYS
        and isinstance(path[1], int)
    )


def _is_auxiliary_chain_entry_path(path: tuple[Any, ...]) -> bool:
    return (
        len(path) == 4
        and path[0] == "auxiliary"
        and isinstance(path[1], str)
        and path[2] == _AUXILIARY_CHAIN_KEY
        and isinstance(path[3], int)
    )


def _configured_primary_model(config: Mapping[str, Any]) -> str:
    model_cfg = config.get("model")
    if isinstance(model_cfg, str):
        return model_cfg.strip()
    if isinstance(model_cfg, Mapping):
        return str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
    return ""


def _anchor_model_for_path(
    config: Mapping[str, Any], path: tuple[Any, ...]
) -> str:
    """The model an opt-in at ``path`` would preserve at runtime.

    Auxiliary task chains are anchored to that task's own model when it pins
    one — an aux mirror is a mirror of the aux route, not of the main route.
    """
    if _is_auxiliary_chain_entry_path(path):
        auxiliary = config.get("auxiliary")
        task_cfg = auxiliary.get(path[1]) if isinstance(auxiliary, Mapping) else None
        if isinstance(task_cfg, Mapping):
            task_model = str(task_cfg.get("model") or "").strip()
            if task_model:
                return task_model
    return _configured_primary_model(config)


def preserve_requested_model_issues(
    config: Mapping[str, Any] | None,
) -> list[PreserveModelIssue]:
    """Report every ``preserve_requested_model`` problem in a config tree.

    Feeds ``validate_config_structure`` (and through it ``hermes doctor`` and
    the startup warning banner).  Three problem classes, each of which used to
    be completely silent while the mirrored endpoint kept downgrading models:

    * **bad value** — anything that is not a real YAML boolean;
    * **bad placement** — the flag anywhere other than a fallback chain entry.
      At the config root it is worse than ignored: top-level scalars are
      bridged into ``os.environ``, so it becomes an env var;
    * **bad use** — a well-typed opt-in that can never fire (no ``provider``;
      no ``model`` on a main-chain entry, which the chain loader drops; or a
      declared model from another family, which the runtime always refuses).
    """
    if not isinstance(config, Mapping):
        return []

    issues: list[PreserveModelIssue] = []
    for path, entry in _iter_key_locations(config, PRESERVE_REQUESTED_MODEL_KEY):
        where = _format_config_path(path)
        is_main = _is_main_chain_entry_path(path)
        is_auxiliary = _is_auxiliary_chain_entry_path(path)
        if not (is_main or is_auxiliary):
            extra = (
                " A top-level scalar is exported into the environment instead."
                if not path else ""
            )
            issues.append(PreserveModelIssue(
                where,
                f"{PRESERVE_REQUESTED_MODEL_KEY} at {where} is never read — the "
                "flag only applies inside a fallback chain entry "
                "(fallback_providers[i], fallback_model, or "
                f"auxiliary.<task>.{_AUXILIARY_CHAIN_KEY}[i]), so the entry it "
                f"was meant for keeps using its configured model.{extra}",
                PRESERVE_REQUESTED_MODEL_HINT,
            ))
            continue

        try:
            enabled = validate_preserve_requested_model(
                entry[PRESERVE_REQUESTED_MODEL_KEY], where=where
            )
        except FallbackModelPolicyError as exc:
            issues.append(PreserveModelIssue(
                where, str(exc), PRESERVE_REQUESTED_MODEL_HINT
            ))
            continue
        if not enabled:
            continue

        if not str(entry.get("provider") or "").strip():
            issues.append(PreserveModelIssue(
                where,
                f"{where} sets {PRESERVE_REQUESTED_MODEL_KEY}: true but declares "
                "no 'provider', so the entry is dropped and the model is never "
                "preserved",
                "Add the provider whose endpoint mirrors your model, e.g.\n"
                "  - provider: my-bridge-b",
            ))
            continue

        configured = str(entry.get("model") or "").strip()
        if not configured and is_main:
            issues.append(PreserveModelIssue(
                where,
                f"{where} sets {PRESERVE_REQUESTED_MODEL_KEY}: true but declares "
                "no 'model', and fallback chain entries without a model are "
                "dropped before the flag can apply",
                "Declare the model the mirrored endpoint serves; the preserved "
                "request keeps the model you actually asked for:\n"
                "  - provider: my-bridge-b\n"
                "    model: claude-opus-5\n"
                f"    {PRESERVE_REQUESTED_MODEL_KEY}: true",
            ))
            continue

        anchor = _anchor_model_for_path(config, path)
        anchor_family = model_family_key(anchor)
        entry_family = model_family_key(configured)
        if anchor_family and entry_family and anchor_family != entry_family:
            issues.append(PreserveModelIssue(
                where,
                f"{where} sets {PRESERVE_REQUESTED_MODEL_KEY}: true but declares "
                f"{configured!r} while the route requests {anchor!r} — "
                f"cross-family substitution ({anchor_family} → {entry_family}) "
                "is always refused, so this entry can only ever be skipped",
                "Point the entry at a mirror of the model you request (same "
                f"family), or drop {PRESERVE_REQUESTED_MODEL_KEY} to use the "
                "entry's own model.",
            ))

    return issues
