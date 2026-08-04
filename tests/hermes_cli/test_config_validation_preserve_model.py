"""``validate_config_structure`` must fail closed on ``preserve_requested_model``.

Two Nemo HIGH findings meet here:

1. **Bad values were invisible.**  ``preserve_requested_model: "yes"`` (quoted)
   or ``: 2`` parsed as a non-boolean and was silently ignored by config
   validation, so the operator believed the mirror endpoint kept their model
   while the runtime skipped the flag (or, before the strict parse, honored a
   coerced string).  Either way the deployed routing did not match the file.

2. **Illegal placement was invisible.**  The key only means anything *inside a
   fallback chain entry*.  Written at root level (a one-notch indentation
   mistake) it is not just ignored — top-level scalars are bridged into
   ``os.environ``, so the flag becomes an env var and the fallback entry it was
   meant for keeps silently downgrading the model.  Same for ``model:``,
   ``providers.<name>`` and ``custom_providers[i]``.

Every issue must be an actionable *error* (not a warning), must name the exact
YAML path, and must never echo credentials — these messages are printed at
startup, land in logs and get pasted into bug reports.
"""

from __future__ import annotations

import yaml

from hermes_cli.config import validate_config_structure
from hermes_cli.fallback_config import PRESERVE_REQUESTED_MODEL_KEY

SECRETISH = "sk-live-8f3a9d2c-not-a-real-key"
TEAM_URL = "http://127.0.0.1:4319/v1"


def _errors(config):
    return [i for i in validate_config_structure(config) if i.severity == "error"]


def _preserve_errors(config):
    return [
        i for i in _errors(config)
        if PRESERVE_REQUESTED_MODEL_KEY in i.message
    ]


def _team_entry(**overrides):
    entry = {
        "provider": "claude-sdk-team-local",
        "model": "claude-sonnet-5",
        "base_url": TEAM_URL,
        "api_key": SECRETISH,
    }
    entry.update(overrides)
    return entry


def _mirrored_config(**entry_overrides):
    """The live 4318 → 4319 shape (primary bridge + mirrored bridge)."""
    return {
        "model": {
            "provider": "claude-sdk-local",
            "default": "claude-opus-5",
            "base_url": "http://127.0.0.1:4318/v1",
        },
        "fallback_providers": [_team_entry(**entry_overrides)],
    }


# ── Legal placement, legal value: silent ──────────────────────────────────


class TestValidOptInIsSilent:
    def test_boolean_true_on_a_chain_entry_is_accepted(self):
        assert _preserve_errors(
            _mirrored_config(**{PRESERVE_REQUESTED_MODEL_KEY: True})
        ) == []

    def test_boolean_false_on_a_chain_entry_is_accepted(self):
        assert _preserve_errors(
            _mirrored_config(**{PRESERVE_REQUESTED_MODEL_KEY: False})
        ) == []

    def test_legacy_fallback_model_dict_is_accepted(self):
        config = {
            "model": {"provider": "claude-sdk-local", "default": "claude-opus-5"},
            "fallback_model": _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True}),
        }
        assert _preserve_errors(config) == []

    def test_legacy_fallback_model_list_is_accepted(self):
        config = {
            "model": {"provider": "claude-sdk-local", "default": "claude-opus-5"},
            "fallback_model": [_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})],
        }
        assert _preserve_errors(config) == []

    def test_auxiliary_task_chain_entry_is_accepted(self):
        config = {
            "model": {"provider": "claude-sdk-local", "default": "claude-opus-5"},
            "auxiliary": {
                "compression": {
                    "provider": "auto",
                    "fallback_chain": [
                        _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})
                    ],
                }
            },
        }
        assert _preserve_errors(config) == []

    def test_config_without_the_key_is_unchanged(self):
        """No regression for the overwhelmingly common config."""
        assert validate_config_structure({
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4"},
            "fallback_providers": [
                {"provider": "anthropic", "model": "claude-sonnet-4-6"}
            ],
        }) == []


# ── Illegal values inside a legal location ────────────────────────────────


class TestIllegalValues:
    def test_quoted_string_is_an_error_naming_the_path(self):
        issues = _preserve_errors(
            _mirrored_config(**{PRESERVE_REQUESTED_MODEL_KEY: "yes"})
        )
        assert len(issues) == 1
        assert "fallback_providers[0]" in issues[0].message
        assert "boolean" in issues[0].message.lower()
        assert f"{PRESERVE_REQUESTED_MODEL_KEY}: true" in issues[0].hint

    def test_integer_two_is_an_error(self):
        issues = _preserve_errors(
            _mirrored_config(**{PRESERVE_REQUESTED_MODEL_KEY: 2})
        )
        assert len(issues) == 1
        assert "fallback_providers[0]" in issues[0].message

    def test_null_is_an_error(self):
        config = yaml.safe_load(
            "model:\n"
            "  provider: claude-sdk-local\n"
            "  default: claude-opus-5\n"
            "fallback_providers:\n"
            "  - provider: claude-sdk-team-local\n"
            "    model: claude-sonnet-5\n"
            f"    {PRESERVE_REQUESTED_MODEL_KEY}:\n"
        )
        assert config["fallback_providers"][0][PRESERVE_REQUESTED_MODEL_KEY] is None
        assert len(_preserve_errors(config)) == 1

    def test_collection_is_an_error(self):
        for raw in ([], [True], {"nested": True}):
            issues = _preserve_errors(
                _mirrored_config(**{PRESERVE_REQUESTED_MODEL_KEY: raw})
            )
            assert len(issues) == 1, raw

    def test_legacy_list_form_reports_the_index(self):
        config = {
            "model": {"provider": "claude-sdk-local", "default": "claude-opus-5"},
            "fallback_model": [
                _team_entry(),
                _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: "on"}),
            ],
        }
        issues = _preserve_errors(config)
        assert len(issues) == 1
        assert "fallback_model[1]" in issues[0].message

    def test_auxiliary_chain_reports_the_task_and_index(self):
        config = {
            "model": {"provider": "claude-sdk-local", "default": "claude-opus-5"},
            "auxiliary": {
                "compression": {
                    "fallback_chain": [
                        _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: "yes"})
                    ]
                }
            },
        }
        issues = _preserve_errors(config)
        assert len(issues) == 1
        assert "auxiliary.compression.fallback_chain[0]" in issues[0].message


# ── Illegal placement ─────────────────────────────────────────────────────


class TestIllegalPlacement:
    def test_root_level_is_an_error(self):
        config = _mirrored_config()
        config[PRESERVE_REQUESTED_MODEL_KEY] = True
        issues = _preserve_errors(config)
        assert len(issues) == 1
        assert "root" in issues[0].message.lower() or "top level" in issues[0].message.lower()
        assert "fallback" in issues[0].hint.lower()

    def test_root_level_error_fires_even_when_the_value_is_valid(self):
        """A perfectly-typed boolean in the wrong place is still a silent no-op."""
        assert _preserve_errors({PRESERVE_REQUESTED_MODEL_KEY: True})

    def test_inside_model_section_is_an_error(self):
        config = _mirrored_config()
        config["model"][PRESERVE_REQUESTED_MODEL_KEY] = True
        issues = _preserve_errors(config)
        assert len(issues) == 1
        assert "model" in issues[0].message

    def test_inside_a_custom_providers_entry_is_an_error(self):
        config = {
            "model": {"provider": "custom", "default": "claude-opus-5"},
            "custom_providers": [
                {
                    "name": "claude-sdk-team-local",
                    "base_url": TEAM_URL,
                    PRESERVE_REQUESTED_MODEL_KEY: True,
                }
            ],
        }
        issues = _preserve_errors(config)
        assert len(issues) == 1
        assert "custom_providers[0]" in issues[0].message

    def test_inside_a_providers_map_entry_is_an_error(self):
        config = {
            "model": {"provider": "claude-sdk-team-local", "default": "claude-opus-5"},
            "providers": {
                "claude-sdk-team-local": {
                    "base_url": TEAM_URL,
                    PRESERVE_REQUESTED_MODEL_KEY: True,
                }
            },
        }
        issues = _preserve_errors(config)
        assert len(issues) == 1
        assert "providers.claude-sdk-team-local" in issues[0].message

    def test_nested_under_an_entrys_extra_body_is_an_error(self):
        """Right entry, wrong depth — still never read as the routing flag."""
        config = _mirrored_config(extra_body={PRESERVE_REQUESTED_MODEL_KEY: True})
        assert len(_preserve_errors(config)) == 1

    def test_every_illegal_placement_is_reported_once(self):
        config = _mirrored_config(**{PRESERVE_REQUESTED_MODEL_KEY: True})
        config[PRESERVE_REQUESTED_MODEL_KEY] = True
        config["model"][PRESERVE_REQUESTED_MODEL_KEY] = True
        issues = _preserve_errors(config)
        assert len(issues) == 2


# ── Illegal use of a well-placed, well-typed flag ─────────────────────────


class TestIllegalUse:
    def test_opt_in_without_a_provider_is_an_error(self):
        config = {
            "model": {"provider": "claude-sdk-local", "default": "claude-opus-5"},
            "fallback_providers": [
                {"model": "claude-sonnet-5", PRESERVE_REQUESTED_MODEL_KEY: True}
            ],
        }
        issues = _preserve_errors(config)
        assert len(issues) == 1
        assert "provider" in issues[0].message

    def test_opt_in_without_a_model_is_an_error_on_the_main_chain(self):
        """The main chain drops model-less entries, so the opt-in never runs."""
        config = {
            "model": {"provider": "claude-sdk-local", "default": "claude-opus-5"},
            "fallback_providers": [
                {
                    "provider": "claude-sdk-team-local",
                    "base_url": TEAM_URL,
                    PRESERVE_REQUESTED_MODEL_KEY: True,
                }
            ],
        }
        issues = _preserve_errors(config)
        assert len(issues) == 1
        assert "model" in issues[0].message

    def test_model_less_auxiliary_entry_is_allowed(self):
        """Auxiliary chains document ``model`` as optional — no error there."""
        config = {
            "model": {"provider": "claude-sdk-local", "default": "claude-opus-5"},
            "auxiliary": {
                "compression": {
                    "fallback_chain": [
                        {
                            "provider": "claude-sdk-team-local",
                            "base_url": TEAM_URL,
                            PRESERVE_REQUESTED_MODEL_KEY: True,
                        }
                    ]
                }
            },
        }
        assert _preserve_errors(config) == []

    def test_cross_family_entry_is_an_error(self):
        """It can only ever fail closed at runtime — say so at config time."""
        config = _mirrored_config(
            provider="openrouter",
            model="openai/gpt-5.4",
            **{PRESERVE_REQUESTED_MODEL_KEY: True},
        )
        issues = _preserve_errors(config)
        assert len(issues) == 1
        assert "cross-family" in issues[0].message.lower()
        assert "claude-opus-5" in issues[0].message
        assert "gpt-5.4" in issues[0].message

    def test_cross_family_is_not_flagged_when_opted_out(self):
        config = _mirrored_config(provider="openrouter", model="openai/gpt-5.4")
        assert _preserve_errors(config) == []

    def test_cross_family_uses_the_auxiliary_task_model_as_the_anchor(self):
        """An aux mirror is measured against the aux model, not the main one."""
        config = {
            "model": {"provider": "claude-sdk-local", "default": "claude-opus-5"},
            "auxiliary": {
                "compression": {
                    "provider": "openrouter",
                    "model": "openai/gpt-5.4",
                    "fallback_chain": [
                        {
                            "provider": "openrouter-mirror",
                            "model": "openai/gpt-5.4",
                            PRESERVE_REQUESTED_MODEL_KEY: True,
                        }
                    ],
                }
            },
        }
        assert _preserve_errors(config) == []

    def test_unknown_primary_model_does_not_guess(self):
        """No primary model configured → no family claim → no error."""
        config = {
            "fallback_providers": [
                _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})
            ]
        }
        assert _preserve_errors(config) == []


# ── Diagnostics must never leak credentials ───────────────────────────────


class TestDiagnosticsAreSecretFree:
    def test_no_issue_text_contains_credentials(self):
        config = _mirrored_config(
            key_env="TEAM_KEY_ENV", **{PRESERVE_REQUESTED_MODEL_KEY: "yes"}
        )
        config[PRESERVE_REQUESTED_MODEL_KEY] = "yes"
        issues = validate_config_structure(config)
        assert issues
        for issue in issues:
            blob = f"{issue.message}\n{issue.hint}"
            assert SECRETISH not in blob
            assert TEAM_URL not in blob
