"""Activation-path tests for ``preserve_requested_model`` fallback entries.

Reproduces the live V2.9 routing incident end to end through
``AIAgent._try_activate_fallback``: primary ``claude-opus-5`` on
``claude-sdk-local`` (4318) failing over to ``claude-sdk-team-local`` (4319).
Without the opt-in the chain entry's own model is requested (the historical
Opus→Sonnet downgrade).  With the opt-in the requested model must survive the
provider/base_url swap, and any substitution attempt must fail closed instead
of silently running a different model.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hermes_cli.fallback_config import PRESERVE_REQUESTED_MODEL_KEY
from run_agent import AIAgent

PRIMARY_MODEL = "claude-opus-5"
PRIMARY_PROVIDER = "claude-sdk-local"
PRIMARY_URL = "http://127.0.0.1:4318/v1"
TEAM_PROVIDER = "claude-sdk-team-local"
TEAM_URL = "http://127.0.0.1:4319/v1"


def _make_agent(fallback_model=None):
    """Primary route pinned to the 4318 bridge running Opus 5."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url=PRIMARY_URL,
            model=PRIMARY_MODEL,
            provider=PRIMARY_PROVIDER,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url=TEAM_URL, api_key="team-key"):
    client = MagicMock()
    client.base_url = base_url
    client.api_key = api_key
    return client


def _team_entry(**overrides):
    entry = {
        "provider": TEAM_PROVIDER,
        "model": "claude-sonnet-5",
        "base_url": TEAM_URL,
    }
    entry.update(overrides)
    return entry


class _Resolver:
    """Records every ``resolve_provider_client`` call and its model argument."""

    def __init__(self, resolved_model=None, client=None):
        self.calls = []
        self._resolved_model = resolved_model
        self._client = client

    def __call__(self, provider, **kwargs):
        self.calls.append((provider, kwargs.get("model")))
        client = self._client if self._client is not None else _mock_client()
        resolved = (
            self._resolved_model
            if self._resolved_model is not None
            else kwargs.get("model")
        )
        return client, resolved

    @property
    def models(self):
        return [model for _provider, model in self.calls]


def _identity_normalizer():
    return patch(
        "hermes_cli.model_normalize.normalize_model_for_provider",
        side_effect=lambda model, provider: model,
    )


# ── (2) No opt-in: the configured fallback model still wins ───────────────


class TestWithoutOptIn:
    def test_configured_fallback_model_is_requested(self):
        """Pins the pre-existing behavior the incident exposed.

        The 4319 entry declares ``claude-sonnet-5``; without the opt-in that
        is exactly what must be requested (backward compatibility), which is
        how the live downgrade happened.
        """
        agent = _make_agent(fallback_model=[_team_entry()])
        resolver = _Resolver()
        with (
            _identity_normalizer(),
            patch("agent.auxiliary_client.resolve_provider_client", resolver),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.model == "claude-sonnet-5"
        assert agent.provider == TEAM_PROVIDER
        assert resolver.models == ["claude-sonnet-5"]

    def test_explicit_false_behaves_like_no_key(self):
        agent = _make_agent(
            fallback_model=[_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: False})]
        )
        resolver = _Resolver()
        with (
            _identity_normalizer(),
            patch("agent.auxiliary_client.resolve_provider_client", resolver),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.model == "claude-sonnet-5"


# ── (1) Opt-in preserves the requested model across the provider switch ───


class TestOptInPreservesRequestedModel:
    def test_opus_survives_the_switch_to_the_team_bridge(self):
        agent = _make_agent(
            fallback_model=[_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})]
        )
        resolver = _Resolver()
        with (
            _identity_normalizer(),
            patch("agent.auxiliary_client.resolve_provider_client", resolver),
        ):
            assert agent._try_activate_fallback() is True

        # Provider/base_url switched, model did not.
        assert agent.provider == TEAM_PROVIDER
        assert agent.requested_provider == TEAM_PROVIDER
        assert agent.base_url == TEAM_URL
        assert agent.model == PRIMARY_MODEL
        # The model handed to the provider router is what the 4319 child runs.
        assert resolver.models == [PRIMARY_MODEL]

    def test_status_and_notice_report_the_preserved_model(self):
        agent = _make_agent(
            fallback_model=[_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})]
        )
        with (
            _identity_normalizer(),
            patch("agent.auxiliary_client.resolve_provider_client", _Resolver()),
        ):
            assert agent._try_activate_fallback() is True

        notice = agent._pending_fallback_notice or ""
        assert PRIMARY_MODEL in notice
        assert "claude-sonnet-5" not in notice

    def test_unavailable_memo_is_keyed_on_the_preserved_model(self):
        """Session cooldown/skip memo must describe what was actually tried."""
        agent = _make_agent(
            fallback_model=[_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})]
        )
        with (
            _identity_normalizer(),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(None, None),
            ),
        ):
            assert agent._try_activate_fallback() is False

        keys = agent._unavailable_fallback_keys
        assert (TEAM_PROVIDER, PRIMARY_MODEL, TEAM_URL) in keys

    def test_policy_is_per_agent_not_per_chain(self):
        """A delegate child on its own model preserves ITS model, generically.

        The same inherited chain entry must resolve against whichever agent
        walks it — nothing in core is pinned to one model name.
        """
        child_model = "claude-haiku-5"
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            child = AIAgent(
                api_key="test-key",
                base_url=PRIMARY_URL,
                model=child_model,
                provider=PRIMARY_PROVIDER,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                fallback_model=[_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})],
            )
        child.client = MagicMock()
        resolver = _Resolver()
        with (
            _identity_normalizer(),
            patch("agent.auxiliary_client.resolve_provider_client", resolver),
        ):
            assert child._try_activate_fallback() is True

        assert child.model == child_model
        assert resolver.models == [child_model]

    def test_chain_entry_is_never_mutated_by_activation(self):
        """Chains are shared objects; activation must leave them untouched."""
        entry = _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})
        agent = _make_agent(fallback_model=[entry])
        with (
            _identity_normalizer(),
            patch("agent.auxiliary_client.resolve_provider_client", _Resolver()),
        ):
            assert agent._try_activate_fallback() is True

        assert entry["model"] == "claude-sonnet-5"
        assert agent._fallback_chain[0]["model"] == "claude-sonnet-5"

    def test_anchor_is_the_primary_model_not_the_active_fallback(self):
        """A second hop preserves the ORIGINAL request, not hop #1's model."""
        agent = _make_agent(
            fallback_model=[
                {"provider": "openrouter", "model": "openai/gpt-5.4"},
                _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True}),
            ]
        )
        resolver = _Resolver()
        with (
            _identity_normalizer(),
            patch("agent.auxiliary_client.resolve_provider_client", resolver),
        ):
            assert agent._try_activate_fallback() is True
            assert agent.model == "openai/gpt-5.4"
            assert agent._try_activate_fallback() is True

        assert agent.model == PRIMARY_MODEL
        assert resolver.models == ["openai/gpt-5.4", PRIMARY_MODEL]


# ── (3) Same-backend loop prevention is unchanged ─────────────────────────


class TestLoopPrevention:
    def test_preserving_entry_on_the_current_backend_is_skipped(self):
        """Preserving the model must not create a fall-back-to-self loop."""
        agent = _make_agent(
            fallback_model=[
                {
                    "provider": PRIMARY_PROVIDER,
                    "model": "claude-sonnet-5",
                    "base_url": PRIMARY_URL,
                    PRESERVE_REQUESTED_MODEL_KEY: True,
                }
            ]
        )
        resolver = _Resolver()
        with (
            _identity_normalizer(),
            patch("agent.auxiliary_client.resolve_provider_client", resolver),
        ):
            assert agent._try_activate_fallback() is False

        assert resolver.calls == []
        assert agent.model == PRIMARY_MODEL
        assert agent.provider == PRIMARY_PROVIDER
        assert agent._fallback_activated is False

    def test_preserving_entry_skips_self_then_takes_the_next_hop(self):
        agent = _make_agent(
            fallback_model=[
                {
                    "provider": PRIMARY_PROVIDER,
                    "model": "claude-sonnet-5",
                    "base_url": PRIMARY_URL,
                    PRESERVE_REQUESTED_MODEL_KEY: True,
                },
                _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True}),
            ]
        )
        resolver = _Resolver()
        with (
            _identity_normalizer(),
            patch("agent.auxiliary_client.resolve_provider_client", resolver),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.provider == TEAM_PROVIDER
        assert agent.model == PRIMARY_MODEL
        assert resolver.models == [PRIMARY_MODEL]


# ── (4) Invalid config and substitution attempts fail closed ──────────────


class TestFailsClosed:
    def test_router_resolving_another_model_is_refused(self):
        """A provider that answers with a different model must be skipped."""
        agent = _make_agent(
            fallback_model=[_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})]
        )
        resolver = _Resolver(resolved_model="claude-sonnet-5")
        with (
            _identity_normalizer(),
            patch("agent.auxiliary_client.resolve_provider_client", resolver),
        ):
            assert agent._try_activate_fallback() is False

        # Nothing was swapped: still on the primary route.
        assert agent.model == PRIMARY_MODEL
        assert agent.provider == PRIMARY_PROVIDER
        assert agent.base_url == PRIMARY_URL
        assert agent._fallback_activated is False

    def test_normalizer_rewriting_the_model_is_refused(self):
        agent = _make_agent(
            fallback_model=[_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})]
        )
        with (
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda model, provider: "claude-sonnet-5",
            ),
            patch("agent.auxiliary_client.resolve_provider_client", _Resolver()),
        ):
            assert agent._try_activate_fallback() is False

        assert agent.model == PRIMARY_MODEL
        assert agent._fallback_activated is False

    def test_vendor_prefix_normalization_is_still_accepted(self):
        """Reformatting the same model is not a substitution."""
        agent = _make_agent(
            fallback_model=[
                _team_entry(
                    provider="openrouter",
                    base_url="https://openrouter.ai/api/v1",
                    **{PRESERVE_REQUESTED_MODEL_KEY: True},
                )
            ]
        )
        with (
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda model, provider: f"anthropic/{model}",
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                _Resolver(client=_mock_client(base_url="https://openrouter.ai/api/v1")),
            ),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.model == "anthropic/claude-opus-5"
        assert agent.provider == "openrouter"

    def test_invalid_opt_in_value_skips_the_entry_and_walks_on(self):
        agent = _make_agent(
            fallback_model=[
                _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: "sometimes"}),
                {"provider": "openrouter", "model": "openai/gpt-5.4"},
            ]
        )
        resolver = _Resolver()
        with (
            _identity_normalizer(),
            patch("agent.auxiliary_client.resolve_provider_client", resolver),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.provider == "openrouter"
        assert agent.model == "openai/gpt-5.4"
        assert resolver.models == ["openai/gpt-5.4"]

    def test_invalid_opt_in_value_alone_exhausts_the_chain(self):
        agent = _make_agent(
            fallback_model=[_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: []})]
        )
        resolver = _Resolver()
        with (
            _identity_normalizer(),
            patch("agent.auxiliary_client.resolve_provider_client", resolver),
        ):
            assert agent._try_activate_fallback() is False

        assert resolver.calls == []
        assert agent.model == PRIMARY_MODEL
        assert agent._fallback_activated is False
