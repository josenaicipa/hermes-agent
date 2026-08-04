"""Fallback entries that must keep the requested model across a provider swap.

Live incident (V2.9 routing): the primary route was ``claude-opus-5`` on
``claude-sdk-local`` (127.0.0.1:4318) and the inherited chain entry for
``claude-sdk-team-local`` (127.0.0.1:4319) was configured with
``model: claude-sonnet-5``.  When 4318 reported exhausted credits Hermes
activated that entry verbatim, so the 4319 child process was spawned with
``--model claude-sonnet-5`` — a silent downgrade of the Opus 5 request.

The correction is generic and opt-in: an entry may declare
``preserve_requested_model: true`` to say "switch provider/base_url, keep the
model the caller actually asked for".  These tests pin the policy owned by
``hermes_cli.fallback_config`` — no port numbers or model names are special
cased in the implementation.
"""

from __future__ import annotations

from hermes_cli.fallback_config import (
    PRESERVE_REQUESTED_MODEL_KEY,
    FallbackModelPolicyError,
    effective_fallback_entry,
    entry_preserves_requested_model,
    get_fallback_chain,
    require_preserved_model,
    resolve_fallback_model,
)

PRIMARY_MODEL = "claude-opus-5"
PRIMARY_PROVIDER = "claude-sdk-local"
PRIMARY_URL = "http://127.0.0.1:4318/v1"
TEAM_PROVIDER = "claude-sdk-team-local"
TEAM_URL = "http://127.0.0.1:4319/v1"


def _team_entry(**overrides):
    """The inherited 4319 entry exactly as the live config declares it."""
    entry = {
        "provider": TEAM_PROVIDER,
        "model": "claude-sonnet-5",
        "base_url": TEAM_URL,
    }
    entry.update(overrides)
    return entry


def _raises(exc_type, fn, *args, **kwargs):
    """Return the expected exception, or fail loudly when none is raised."""
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


# ── Opt-in parsing: strict, backward compatible, fail-closed ──────────────


class TestOptInParsing:
    def test_absent_key_is_not_opt_in(self):
        assert entry_preserves_requested_model(_team_entry()) is False

    def test_boolean_true_opts_in(self):
        entry = _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})
        assert entry_preserves_requested_model(entry) is True

    def test_boolean_false_stays_opt_out(self):
        entry = _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: False})
        assert entry_preserves_requested_model(entry) is False

    def test_yaml_string_truths_opt_in(self):
        for raw in ("true", "TRUE", "yes", "on", "1", "enabled"):
            entry = _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: raw})
            assert entry_preserves_requested_model(entry) is True, raw

    def test_yaml_string_falsehoods_stay_opt_out(self):
        for raw in ("false", "no", "off", "0", "disabled"):
            entry = _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: raw})
            assert entry_preserves_requested_model(entry) is False, raw

    def test_unrecognized_value_fails_closed(self):
        entry = _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: "maybe"})
        exc = _raises(FallbackModelPolicyError, entry_preserves_requested_model, entry)
        assert PRESERVE_REQUESTED_MODEL_KEY in str(exc)

    def test_non_scalar_value_fails_closed(self):
        entry = _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: {"nested": True}})
        _raises(FallbackModelPolicyError, entry_preserves_requested_model, entry)

    def test_non_mapping_entry_is_not_opt_in(self):
        assert entry_preserves_requested_model(None) is False
        assert entry_preserves_requested_model("claude-sonnet-5") is False


# ── Effective model resolution ────────────────────────────────────────────


class TestResolveFallbackModel:
    def test_without_opt_in_the_configured_model_wins(self):
        """Regression guard for the pre-existing (documented) behavior.

        This is the exact shape that produced the live Opus→Sonnet
        downgrade; entries that never opt in must keep it so no existing
        chain changes meaning.
        """
        decision = resolve_fallback_model(
            _team_entry(), requested_model=PRIMARY_MODEL
        )
        assert decision.model == "claude-sonnet-5"
        assert decision.preserved is False
        assert decision.configured_model == "claude-sonnet-5"

    def test_opt_in_preserves_the_requested_model(self):
        decision = resolve_fallback_model(
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True}),
            requested_model=PRIMARY_MODEL,
        )
        assert decision.model == PRIMARY_MODEL
        assert decision.preserved is True
        # The configured value is retained for diagnostics only.
        assert decision.configured_model == "claude-sonnet-5"

    def test_opt_in_without_a_requested_model_fails_closed(self):
        exc = _raises(
            FallbackModelPolicyError,
            resolve_fallback_model,
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True}),
            requested_model="   ",
        )
        assert "requested model" in str(exc).lower()

    def test_opt_in_never_downgrades_to_the_configured_model(self):
        """Fail closed: silently using ``claude-sonnet-5`` is forbidden."""
        entry = _team_entry(
            model="claude-sonnet-5", **{PRESERVE_REQUESTED_MODEL_KEY: True}
        )
        decision = resolve_fallback_model(entry, requested_model=PRIMARY_MODEL)
        assert decision.model != entry["model"]

    def test_invalid_opt_in_value_fails_closed(self):
        _raises(
            FallbackModelPolicyError,
            resolve_fallback_model,
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: 2}),
            requested_model=PRIMARY_MODEL,
        )

    def test_requested_model_is_trimmed(self):
        decision = resolve_fallback_model(
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True}),
            requested_model="  claude-opus-5  ",
        )
        assert decision.model == PRIMARY_MODEL

    def test_entry_without_model_or_opt_in_yields_empty_model(self):
        decision = resolve_fallback_model(
            {"provider": TEAM_PROVIDER}, requested_model=PRIMARY_MODEL
        )
        assert decision.model == ""
        assert decision.preserved is False


# ── Effective entry: one object every downstream check consumes ───────────


class TestEffectiveFallbackEntry:
    def test_effective_entry_carries_the_preserved_model(self):
        entry = _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})
        decision = resolve_fallback_model(entry, requested_model=PRIMARY_MODEL)
        effective = effective_fallback_entry(entry, decision)
        assert effective["model"] == PRIMARY_MODEL
        assert effective["provider"] == TEAM_PROVIDER
        assert effective["base_url"] == TEAM_URL

    def test_effective_entry_never_mutates_the_source(self):
        """Chains are shared with delegate children — copy, never mutate."""
        entry = _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})
        decision = resolve_fallback_model(entry, requested_model=PRIMARY_MODEL)
        effective = effective_fallback_entry(entry, decision)
        assert effective is not entry
        assert entry["model"] == "claude-sonnet-5"

    def test_effective_entry_keeps_credential_hints(self):
        entry = _team_entry(
            api_key="team-key",
            key_env="TEAM_KEY_ENV",
            **{PRESERVE_REQUESTED_MODEL_KEY: True},
        )
        decision = resolve_fallback_model(entry, requested_model=PRIMARY_MODEL)
        effective = effective_fallback_entry(entry, decision)
        assert effective["api_key"] == "team-key"
        assert effective["key_env"] == "TEAM_KEY_ENV"

    def test_effective_entry_without_opt_in_is_unchanged(self):
        entry = _team_entry()
        decision = resolve_fallback_model(entry, requested_model=PRIMARY_MODEL)
        assert effective_fallback_entry(entry, decision) == entry


# ── Same-backend loop prevention keeps working on the effective model ─────


class TestLoopPreventionUsesEffectiveModel:
    def _skip(self, entry, *, current_provider, current_url):
        from agent.backend_identity import BackendIdentity, should_skip_candidate

        decision = resolve_fallback_model(entry, requested_model=PRIMARY_MODEL)
        effective = effective_fallback_entry(entry, decision)
        candidate = BackendIdentity.build(
            provider=effective.get("provider"),
            model=effective.get("model"),
            base_url=effective.get("base_url"),
        )
        current = BackendIdentity.build(
            provider=current_provider,
            model=PRIMARY_MODEL,
            base_url=current_url,
        )
        return should_skip_candidate(candidate, current)

    def test_exact_same_backend_is_still_skipped(self):
        """Preserving the model must not defeat the anti-loop guard."""
        entry = {
            "provider": PRIMARY_PROVIDER,
            "model": "claude-sonnet-5",
            "base_url": PRIMARY_URL,
            PRESERVE_REQUESTED_MODEL_KEY: True,
        }
        assert self._skip(
            entry, current_provider=PRIMARY_PROVIDER, current_url=PRIMARY_URL
        ) is True

    def test_different_provider_and_endpoint_is_not_skipped(self):
        entry = _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})
        assert self._skip(
            entry, current_provider=PRIMARY_PROVIDER, current_url=PRIMARY_URL
        ) is False

    def test_opt_out_entry_on_the_same_backend_is_not_skipped(self):
        """Unchanged legacy behavior: a sibling model is its own deployment."""
        entry = {
            "provider": PRIMARY_PROVIDER,
            "model": "claude-sonnet-5",
            "base_url": PRIMARY_URL,
        }
        assert self._skip(
            entry, current_provider=PRIMARY_PROVIDER, current_url=PRIMARY_URL
        ) is False


# ── Fail closed when the preserved model would be substituted ────────────


class TestRequirePreservedModel:
    def _decision(self):
        return resolve_fallback_model(
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True}),
            requested_model=PRIMARY_MODEL,
        )

    def test_identical_model_passes(self):
        require_preserved_model(self._decision(), PRIMARY_MODEL, source="test")

    def test_vendor_prefixed_variant_passes(self):
        require_preserved_model(
            self._decision(), "anthropic/claude-opus-5", source="test"
        )

    def test_dot_notation_variant_passes(self):
        decision = resolve_fallback_model(
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True}),
            requested_model="claude-opus-5.1",
        )
        require_preserved_model(decision, "claude-opus-5-1", source="test")

    def test_substituted_model_fails_closed(self):
        exc = _raises(
            FallbackModelPolicyError,
            require_preserved_model,
            self._decision(),
            "claude-sonnet-5",
            source="team-bridge",
        )
        assert "claude-sonnet-5" in str(exc)
        assert "team-bridge" in str(exc)

    def test_missing_resolved_model_fails_closed(self):
        _raises(
            FallbackModelPolicyError,
            require_preserved_model,
            self._decision(),
            "",
            source="test",
        )

    def test_opt_out_decision_is_never_constrained(self):
        decision = resolve_fallback_model(
            _team_entry(), requested_model=PRIMARY_MODEL
        )
        require_preserved_model(decision, "claude-sonnet-5", source="test")


# ── The opt-in must survive config loading ────────────────────────────────


class TestOptInSurvivesConfigLoading:
    def test_get_fallback_chain_keeps_the_opt_in_key(self):
        config = {
            "fallback_providers": [
                _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})
            ]
        }
        chain = get_fallback_chain(config)
        assert len(chain) == 1
        assert chain[0][PRESERVE_REQUESTED_MODEL_KEY] is True

    def test_legacy_fallback_model_key_keeps_the_opt_in_key(self):
        config = {
            "fallback_model": _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: "yes"})
        }
        chain = get_fallback_chain(config)
        assert chain[0][PRESERVE_REQUESTED_MODEL_KEY] == "yes"

    def test_preserving_entry_is_not_deduped_against_its_plain_twin(self):
        """Same route, two different model policies = two distinct routes."""
        config = {
            "fallback_providers": [
                _team_entry(),
                _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True}),
            ]
        }
        chain = get_fallback_chain(config)
        assert len(chain) == 2
        assert entry_preserves_requested_model(chain[0]) is False
        assert entry_preserves_requested_model(chain[1]) is True

    def test_identical_plain_entries_are_still_deduped(self):
        config = {"fallback_providers": [_team_entry(), _team_entry()]}
        assert len(get_fallback_chain(config)) == 1
