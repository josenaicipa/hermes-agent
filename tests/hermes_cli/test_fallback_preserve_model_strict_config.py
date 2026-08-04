"""Strict config semantics for ``preserve_requested_model`` (Nemo HIGH).

``preserve_requested_model`` is a *routing-safety* flag: when it is on, Hermes
sends the model the caller asked for to a different provider/endpoint.  Reading
it with truthy coercion is therefore a security defect, not a convenience:

* ``preserve_requested_model: "yes"`` (a **quoted string**) used to opt in.  An
  operator who quotes the value, or a template that renders every scalar as a
  string, silently changed which model a request runs on.
* ``preserve_requested_model: 2`` / ``1`` / ``1.0`` used to be rejected only by
  accident of the string branch — any truthy coercion here is a downgrade
  vector in the other direction (a typo enabling or disabling the flag).
* ``preserve_requested_model:`` with an empty value (YAML ``null``) and
  collections must never be interpreted at all.

The rule these tests pin: **only an actual YAML boolean counts.**  Unquoted
``true``/``yes``/``on`` are booleans *at YAML parse time* (YAML 1.1), which is
fine — the flag never does its own coercion.  Everything else fails closed with
an actionable, secret-free error so the entry is skipped instead of running a
different model behind the caller's back.

The second half pins model *eligibility*: preserving is only legal when the
entry's declared model is in the same family as the model being preserved (a
mirror/sibling endpoint).  Cross-family and unsupported ids are refused.
"""

from __future__ import annotations

import yaml

from hermes_cli.fallback_config import (
    PRESERVE_REQUESTED_MODEL_HINT,
    PRESERVE_REQUESTED_MODEL_KEY,
    FallbackModelPolicyError,
    entry_preserves_requested_model,
    get_fallback_chain,
    model_family_key,
    preserve_requested_model_issues,
    resolve_fallback_model,
)

PRIMARY_MODEL = "claude-opus-5"
TEAM_PROVIDER = "claude-sdk-team-local"
TEAM_URL = "http://127.0.0.1:4319/v1"

# A value that must never be echoed back by an error message.  Operators do
# paste tokens into the wrong key, and config diagnostics land in logs, chat
# transcripts and bug reports.
SECRETISH = "sk-live-8f3a9d2c-not-a-real-key"


def _team_entry(**overrides):
    entry = {
        "provider": TEAM_PROVIDER,
        "model": "claude-sonnet-5",
        "base_url": TEAM_URL,
        "api_key": SECRETISH,
    }
    entry.update(overrides)
    return entry


def _raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


def _yaml_entry(value_literal: str) -> dict:
    """Parse one fallback entry from real YAML text.

    Going through YAML (instead of hand-building the dict) is the point: it is
    the only way to prove which literals PyYAML turns into booleans and which
    ones reach Hermes as strings.
    """
    document = (
        "provider: claude-sdk-team-local\n"
        "model: claude-sonnet-5\n"
        "base_url: http://127.0.0.1:4319/v1\n"
        f"{PRESERVE_REQUESTED_MODEL_KEY}: {value_literal}\n"
    )
    return yaml.safe_load(document)


# ── Only real YAML booleans opt in ────────────────────────────────────────


class TestOnlyRealBooleansOptIn:
    def test_missing_key_is_opt_out(self):
        assert entry_preserves_requested_model(_team_entry()) is False

    def test_boolean_true_opts_in(self):
        assert entry_preserves_requested_model(
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})
        ) is True

    def test_boolean_false_opts_out(self):
        assert entry_preserves_requested_model(
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: False})
        ) is False

    def test_unquoted_yaml_truths_are_booleans_before_hermes_sees_them(self):
        """``true``/``yes``/``on`` unquoted are booleans at YAML parse time.

        Hermes must accept them because YAML — not Hermes — did the
        conversion.  This is what keeps hand-written configs ergonomic without
        any coercion inside the flag.
        """
        for literal in ("true", "True", "TRUE", "yes", "on", "y"):
            entry = _yaml_entry(literal)
            value = entry[PRESERVE_REQUESTED_MODEL_KEY]
            if not isinstance(value, bool):
                # YAML 1.1 spells only some of these as booleans; anything that
                # arrives as a string must fail closed (covered below).
                _raises(
                    FallbackModelPolicyError,
                    entry_preserves_requested_model,
                    entry,
                )
                continue
            assert entry_preserves_requested_model(entry) is value, literal

    def test_quoted_yes_no_longer_opts_in(self):
        """The core regression: a quoted string must not enable the flag."""
        entry = _yaml_entry("'yes'")
        assert entry[PRESERVE_REQUESTED_MODEL_KEY] == "yes"
        exc = _raises(
            FallbackModelPolicyError, entry_preserves_requested_model, entry
        )
        assert PRESERVE_REQUESTED_MODEL_KEY in str(exc)

    def test_every_quoted_boolean_word_fails_closed(self):
        for literal in ("'true'", '"true"', "'yes'", "'on'", "'1'", "'enabled'",
                        "'false'", "'no'", "'off'", "'0'", "'disabled'"):
            entry = _yaml_entry(literal)
            assert isinstance(entry[PRESERVE_REQUESTED_MODEL_KEY], str), literal
            _raises(
                FallbackModelPolicyError,
                entry_preserves_requested_model,
                entry,
            )

    def test_integer_two_fails_closed(self):
        exc = _raises(
            FallbackModelPolicyError,
            entry_preserves_requested_model,
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: 2}),
        )
        assert "boolean" in str(exc).lower()

    def test_integer_one_and_zero_fail_closed(self):
        """``1``/``0`` are the classic truthy-coercion trap — both are refused."""
        for raw in (1, 0):
            _raises(
                FallbackModelPolicyError,
                entry_preserves_requested_model,
                _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: raw}),
            )

    def test_float_fails_closed(self):
        _raises(
            FallbackModelPolicyError,
            entry_preserves_requested_model,
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: 1.0}),
        )

    def test_null_fails_closed(self):
        """``preserve_requested_model:`` with no value parses to ``None``."""
        entry = _yaml_entry("")
        assert entry[PRESERVE_REQUESTED_MODEL_KEY] is None
        exc = _raises(
            FallbackModelPolicyError, entry_preserves_requested_model, entry
        )
        assert "empty" in str(exc).lower() or "null" in str(exc).lower()

    def test_collections_fail_closed(self):
        for raw in ([], [True], {}, {"nested": True}, ("true",), {"true"}):
            _raises(
                FallbackModelPolicyError,
                entry_preserves_requested_model,
                _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: raw}),
            )

    def test_non_mapping_entry_is_opt_out(self):
        assert entry_preserves_requested_model(None) is False
        assert entry_preserves_requested_model("claude-sonnet-5") is False
        assert entry_preserves_requested_model([{"a": 1}]) is False


# ── Errors must be actionable and must not leak secrets ───────────────────


class TestErrorsAreActionableAndSecretFree:
    def test_message_names_the_key_and_the_accepted_values(self):
        exc = _raises(
            FallbackModelPolicyError,
            entry_preserves_requested_model,
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: "maybe"}),
        )
        text = str(exc)
        assert PRESERVE_REQUESTED_MODEL_KEY in text
        assert "true" in text.lower()
        assert "false" in text.lower()

    def test_message_never_echoes_the_entry_credentials(self):
        entry = _team_entry(
            key_env="TEAM_KEY_ENV", **{PRESERVE_REQUESTED_MODEL_KEY: "maybe"}
        )
        exc = _raises(
            FallbackModelPolicyError, entry_preserves_requested_model, entry
        )
        text = str(exc)
        assert SECRETISH not in text
        assert "TEAM_KEY_ENV" not in text
        # The endpoint can embed credentials (user:pass@host) — never echoed.
        assert TEAM_URL not in text

    def test_message_never_echoes_an_arbitrary_string_value(self):
        """A pasted token in the wrong key must not be reflected into logs."""
        exc = _raises(
            FallbackModelPolicyError,
            entry_preserves_requested_model,
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: SECRETISH}),
        )
        assert SECRETISH not in str(exc)
        assert "string" in str(exc).lower()

    def test_known_boolean_word_is_quoted_back_for_actionability(self):
        """Closed-set words are safe to echo and make the fix obvious."""
        exc = _raises(
            FallbackModelPolicyError,
            entry_preserves_requested_model,
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: "yes"}),
        )
        assert "yes" in str(exc)

    def test_shared_hint_shows_a_correct_entry(self):
        assert f"{PRESERVE_REQUESTED_MODEL_KEY}: true" in PRESERVE_REQUESTED_MODEL_HINT
        assert SECRETISH not in PRESERVE_REQUESTED_MODEL_HINT


# ── resolve_fallback_model inherits the strict parse ──────────────────────


class TestResolveInheritsStrictParsing:
    def test_string_opt_in_never_resolves_a_preserved_model(self):
        _raises(
            FallbackModelPolicyError,
            resolve_fallback_model,
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: "yes"}),
            requested_model=PRIMARY_MODEL,
        )

    def test_string_opt_out_is_not_silently_honored_either(self):
        """``"no"`` must fail closed too — the entry's meaning is unknown."""
        _raises(
            FallbackModelPolicyError,
            resolve_fallback_model,
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: "no"}),
            requested_model=PRIMARY_MODEL,
        )

    def test_real_boolean_still_preserves(self):
        decision = resolve_fallback_model(
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True}),
            requested_model=PRIMARY_MODEL,
        )
        assert decision.model == PRIMARY_MODEL
        assert decision.preserved is True

    def test_chain_loading_still_tolerates_a_bad_value(self):
        """Loading must not explode — activation is where it fails closed.

        ``hermes fallback list`` and ``hermes doctor`` have to be able to show
        the operator the broken entry.
        """
        chain = get_fallback_chain(
            {"fallback_providers": [_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: "yes"})]}
        )
        assert len(chain) == 1
        assert chain[0][PRESERVE_REQUESTED_MODEL_KEY] == "yes"

    def test_a_broken_entry_is_not_deduped_into_its_valid_twin(self):
        """Dedup must not delete the evidence.

        A malformed opt-in is a third route policy, not "opted out": folding it
        into an identical plain entry would drop the broken hop during loading,
        so the operator would see the plain entry activate with the downgraded
        model and find nothing in the log about it.
        """
        config = {
            "fallback_providers": [
                _team_entry(),
                _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: "yes"}),
            ]
        }
        chain = get_fallback_chain(config)
        assert len(chain) == 2
        assert PRESERVE_REQUESTED_MODEL_KEY not in chain[0]
        assert chain[1][PRESERVE_REQUESTED_MODEL_KEY] == "yes"


# ── Model families: never preserve across families ────────────────────────


class TestModelFamilyKey:
    def test_family_ignores_vendor_prefix_and_dot_notation(self):
        assert model_family_key("claude-opus-5") == "claude"
        assert model_family_key("anthropic/claude-opus-5") == "claude"
        assert model_family_key("claude-opus-5.1") == "claude"
        assert model_family_key("  CLAUDE-Sonnet-5  ") == "claude"

    def test_family_is_generic_across_vendors(self):
        assert model_family_key("openai/gpt-5.4") == "gpt"
        assert model_family_key("gemini-3-pro") == "gemini"
        assert model_family_key("glm-4.7") == "glm"

    def test_unusable_ids_have_no_family(self):
        for value in ("", "   ", None, "1234", "-", "///", 5):
            assert model_family_key(value) == "", value


class TestPreservedModelEligibility:
    def test_same_family_mirror_preserves(self):
        decision = resolve_fallback_model(
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True}),
            requested_model=PRIMARY_MODEL,
        )
        assert decision.model == PRIMARY_MODEL

    def test_cross_family_entry_is_refused(self):
        """An entry declaring a GPT model must never answer a Claude request."""
        exc = _raises(
            FallbackModelPolicyError,
            resolve_fallback_model,
            _team_entry(
                provider="openrouter",
                model="openai/gpt-5.4",
                **{PRESERVE_REQUESTED_MODEL_KEY: True},
            ),
            requested_model=PRIMARY_MODEL,
        )
        text = str(exc)
        assert "cross-family" in text.lower()
        assert PRIMARY_MODEL in text
        assert "gpt-5.4" in text

    def test_cross_family_is_symmetric(self):
        _raises(
            FallbackModelPolicyError,
            resolve_fallback_model,
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True}),
            requested_model="openai/gpt-5.4",
        )

    def test_vendor_prefixed_declaration_is_same_family(self):
        decision = resolve_fallback_model(
            _team_entry(
                provider="openrouter",
                model="anthropic/claude-sonnet-5",
                **{PRESERVE_REQUESTED_MODEL_KEY: True},
            ),
            requested_model=PRIMARY_MODEL,
        )
        assert decision.model == PRIMARY_MODEL

    def test_entry_without_a_declared_model_declares_no_family(self):
        """Auxiliary chains legitimately omit ``model`` — nothing to cross."""
        decision = resolve_fallback_model(
            {"provider": TEAM_PROVIDER, "base_url": TEAM_URL,
             PRESERVE_REQUESTED_MODEL_KEY: True},
            requested_model=PRIMARY_MODEL,
        )
        assert decision.model == PRIMARY_MODEL
        assert decision.configured_model == ""

    def test_unsupported_requested_model_id_is_refused(self):
        for junk in ("1234", "-", "///"):
            exc = _raises(
                FallbackModelPolicyError,
                resolve_fallback_model,
                _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True}),
                requested_model=junk,
            )
            assert "unsupported" in str(exc).lower(), junk

    def test_models_allowlist_rejects_a_model_the_endpoint_cannot_serve(self):
        exc = _raises(
            FallbackModelPolicyError,
            resolve_fallback_model,
            _team_entry(
                models=["claude-sonnet-5", "claude-haiku-5"],
                **{PRESERVE_REQUESTED_MODEL_KEY: True},
            ),
            requested_model=PRIMARY_MODEL,
        )
        assert "unsupported" in str(exc).lower()
        assert PRIMARY_MODEL in str(exc)

    def test_models_allowlist_accepts_a_formatting_variant(self):
        decision = resolve_fallback_model(
            _team_entry(
                models=["anthropic/claude-opus-5", "claude-sonnet-5"],
                **{PRESERVE_REQUESTED_MODEL_KEY: True},
            ),
            requested_model=PRIMARY_MODEL,
        )
        assert decision.model == PRIMARY_MODEL

    def test_eligibility_never_constrains_opted_out_entries(self):
        """Legacy behavior is untouched: configured model wins, no checks."""
        decision = resolve_fallback_model(
            _team_entry(
                provider="openrouter",
                model="openai/gpt-5.4",
                models=["openai/gpt-5.4"],
            ),
            requested_model=PRIMARY_MODEL,
        )
        assert decision.model == "openai/gpt-5.4"
        assert decision.preserved is False

    def test_eligibility_errors_are_secret_free(self):
        exc = _raises(
            FallbackModelPolicyError,
            resolve_fallback_model,
            _team_entry(
                provider="openrouter",
                model="openai/gpt-5.4",
                **{PRESERVE_REQUESTED_MODEL_KEY: True},
            ),
            requested_model=PRIMARY_MODEL,
        )
        assert SECRETISH not in str(exc)
        assert TEAM_URL not in str(exc)


# ── Structured config diagnostics (path/message/hint) ─────────────────────


class TestPreserveRequestedModelIssues:
    """The structured form ``validate_config_structure`` adapts.

    Callers (doctor output, future tooling) need the machine-readable YAML path
    of each problem, not just a sentence.
    """

    def test_each_issue_carries_the_yaml_path(self):
        config = {
            "model": {"provider": "claude-sdk-local", "default": PRIMARY_MODEL},
            PRESERVE_REQUESTED_MODEL_KEY: True,
            "fallback_providers": [
                _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: "yes"}),
            ],
            "auxiliary": {
                "compression": {
                    "fallback_chain": [
                        _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: 2}),
                    ]
                }
            },
        }
        paths = {issue.path for issue in preserve_requested_model_issues(config)}
        assert paths == {
            "the config root",
            "fallback_providers[0]",
            "auxiliary.compression.fallback_chain[0]",
        }

    def test_every_issue_has_a_hint_and_no_secrets(self):
        config = {
            "model": {"provider": "claude-sdk-local", "default": PRIMARY_MODEL},
            "fallback_providers": [_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: []})],
        }
        issues = preserve_requested_model_issues(config)
        assert issues
        for issue in issues:
            assert issue.hint.strip()
            assert SECRETISH not in f"{issue.message}{issue.hint}"

    def test_a_clean_config_has_no_issues(self):
        assert preserve_requested_model_issues({
            "model": {"provider": "claude-sdk-local", "default": PRIMARY_MODEL},
            "fallback_providers": [_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})],
        }) == []

    def test_non_mapping_config_is_tolerated(self):
        for config in (None, [], "config"):
            assert preserve_requested_model_issues(config) == []
