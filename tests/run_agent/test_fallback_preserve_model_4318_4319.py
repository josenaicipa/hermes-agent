"""End-to-end 4318 → 4319 mirror switch, from YAML text to activation.

The focused tests elsewhere pin each helper.  This file wires the whole path
together for the exact live shape of the V2.9 routing incident, because every
regression so far slipped through a *seam*, not through a helper:

  YAML text → get_fallback_chain → AIAgent(fallback_model=…)
            → _try_activate_fallback → provider router

Covered seams:

* main agent: the mirrored 4319 entry keeps ``claude-opus-5`` when the opt-in is
  a real YAML boolean, and is skipped (never downgraded) when it is a quoted
  string, a number, an empty value or a collection;
* delegate: a subagent inherits the chain and preserves **its own** model, and
  inherits the fail-closed behavior too;
* config validation and activation agree — a config the validator rejects never
  silently changes the model at runtime.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hermes_cli.config import validate_config_structure
from hermes_cli.fallback_config import PRESERVE_REQUESTED_MODEL_KEY, get_fallback_chain
from run_agent import AIAgent

PRIMARY_MODEL = "claude-opus-5"
PRIMARY_PROVIDER = "claude-sdk-local"
PRIMARY_URL = "http://127.0.0.1:4318/v1"
TEAM_PROVIDER = "claude-sdk-team-local"
TEAM_URL = "http://127.0.0.1:4319/v1"

#: The operator's config, verbatim.  ``{opt_in}`` is the only variable so every
#: case below differs exclusively in how the flag is written.
CONFIG_TEMPLATE = """
model:
  provider: claude-sdk-local
  default: claude-opus-5
  base_url: http://127.0.0.1:4318/v1

fallback_providers:
  - provider: claude-sdk-team-local
    model: claude-sonnet-5
    base_url: http://127.0.0.1:4319/v1
    preserve_requested_model: {opt_in}
"""


def _config(opt_in: str) -> dict:
    return yaml.safe_load(CONFIG_TEMPLATE.format(opt_in=opt_in))


class _Resolver:
    """Stand-in for the provider router; records the model it is asked for."""

    def __init__(self, resolved_model=None, base_url=TEAM_URL):
        self.calls = []
        self._resolved_model = resolved_model
        self._base_url = base_url

    def __call__(self, provider, **kwargs):
        self.calls.append((provider, kwargs.get("model")))
        client = MagicMock()
        client.base_url = self._base_url
        client.api_key = "team-key"
        resolved = (
            self._resolved_model
            if self._resolved_model is not None
            else kwargs.get("model")
        )
        return client, resolved

    @property
    def models(self):
        return [model for _provider, model in self.calls]


def _make_agent(chain, *, model=PRIMARY_MODEL, provider=PRIMARY_PROVIDER, base_url=PRIMARY_URL):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url=base_url,
            model=model,
            provider=provider,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=chain,
        )
    agent.client = MagicMock()
    return agent


def _activate(agent, resolver):
    with (
        patch(
            "hermes_cli.model_normalize.normalize_model_for_provider",
            side_effect=lambda model, provider: model,
        ),
        patch("agent.auxiliary_client.resolve_provider_client", resolver),
    ):
        return agent._try_activate_fallback()


# ── Main agent, real YAML boolean ─────────────────────────────────────────


class TestMirroredSwitchKeepsTheModel:
    def test_unquoted_true_preserves_opus_on_4319(self):
        chain = get_fallback_chain(_config("true"))
        assert chain[0][PRESERVE_REQUESTED_MODEL_KEY] is True

        agent = _make_agent(chain)
        resolver = _Resolver()
        assert _activate(agent, resolver) is True

        assert agent.provider == TEAM_PROVIDER
        assert agent.base_url == TEAM_URL
        assert agent.model == PRIMARY_MODEL
        assert resolver.models == [PRIMARY_MODEL]

    def test_unquoted_yes_is_a_yaml_boolean_too(self):
        """YAML 1.1 resolves bare ``yes`` to a boolean — Hermes coerces nothing."""
        config = _config("yes")
        assert config["fallback_providers"][0][PRESERVE_REQUESTED_MODEL_KEY] is True

        agent = _make_agent(get_fallback_chain(config))
        resolver = _Resolver()
        assert _activate(agent, resolver) is True
        assert agent.model == PRIMARY_MODEL

    def test_false_keeps_the_legacy_downgrade_behavior(self):
        agent = _make_agent(get_fallback_chain(_config("false")))
        resolver = _Resolver()
        assert _activate(agent, resolver) is True

        assert agent.provider == TEAM_PROVIDER
        assert agent.model == "claude-sonnet-5"
        assert resolver.models == ["claude-sonnet-5"]

    def test_the_validator_accepts_the_boolean_form(self):
        assert [
            i for i in validate_config_structure(_config("true"))
            if PRESERVE_REQUESTED_MODEL_KEY in i.message
        ] == []


# ── Main agent, malformed opt-in: skip, never downgrade ───────────────────


@pytest.mark.parametrize(
    "opt_in",
    ["'true'", '"yes"', "'1'", "2", "1", "", "[]", "{}"],
)
class TestMalformedOptInFailsClosed:
    def test_entry_is_skipped_and_the_primary_route_is_kept(self, opt_in):
        agent = _make_agent(get_fallback_chain(_config(opt_in)))
        resolver = _Resolver()
        assert _activate(agent, resolver) is False

        assert resolver.calls == []
        assert agent.model == PRIMARY_MODEL
        assert agent.provider == PRIMARY_PROVIDER
        assert agent.base_url == PRIMARY_URL
        assert agent._fallback_activated is False

    def test_the_validator_reports_it(self, opt_in):
        errors = [
            i for i in validate_config_structure(_config(opt_in))
            if i.severity == "error" and PRESERVE_REQUESTED_MODEL_KEY in i.message
        ]
        assert len(errors) == 1
        assert "boolean" in errors[0].message.lower()


class TestSubstitutionIsRefusedEndToEnd:
    def test_a_router_answering_sonnet_is_refused(self):
        agent = _make_agent(get_fallback_chain(_config("true")))
        assert _activate(agent, _Resolver(resolved_model="claude-sonnet-5")) is False
        assert agent.model == PRIMARY_MODEL
        assert agent._fallback_activated is False

    def test_a_normalizer_rewriting_the_model_is_refused(self):
        agent = _make_agent(get_fallback_chain(_config("true")))
        with (
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda model, provider: "claude-sonnet-5",
            ),
            patch("agent.auxiliary_client.resolve_provider_client", _Resolver()),
        ):
            assert agent._try_activate_fallback() is False
        assert agent.model == PRIMARY_MODEL

    def test_a_cross_family_mirror_entry_is_refused(self):
        """Never preserve across families, even with a perfect boolean."""
        config = _config("true")
        config["fallback_providers"][0]["provider"] = "openrouter"
        config["fallback_providers"][0]["model"] = "openai/gpt-5.4"
        config["fallback_providers"][0].pop("base_url")

        agent = _make_agent(get_fallback_chain(config))
        resolver = _Resolver(base_url="https://openrouter.ai/api/v1")
        assert _activate(agent, resolver) is False

        assert resolver.calls == []
        assert agent.model == PRIMARY_MODEL
        assert [
            i for i in validate_config_structure(config)
            if i.severity == "error" and "cross-family" in i.message.lower()
        ]


# ── Delegate path ─────────────────────────────────────────────────────────


def _mock_parent(chain):
    parent = MagicMock()
    parent.base_url = PRIMARY_URL
    parent.api_key = "***"
    parent.provider = PRIMARY_PROVIDER
    parent.api_mode = "chat_completions"
    parent.model = PRIMARY_MODEL
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._fallback_chain = chain
    return parent


def _inherited_chain(chain):
    """The chain a delegate child is actually constructed with."""
    from tools.delegate_tool import _build_child_agent

    with patch("run_agent.AIAgent") as MockAgent:
        MockAgent.return_value = MagicMock()
        _build_child_agent(
            task_index=0,
            goal="mirror switch",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=10,
            parent_agent=_mock_parent(chain),
            task_count=1,
        )
    return MockAgent.call_args.kwargs["fallback_model"]


class TestDelegateChildInheritsThePolicy:
    def test_child_preserves_its_own_model_not_the_parents(self):
        inherited = _inherited_chain(get_fallback_chain(_config("true")))
        assert inherited[0][PRESERVE_REQUESTED_MODEL_KEY] is True

        child_model = "claude-haiku-5"
        child = _make_agent(inherited, model=child_model)
        resolver = _Resolver()
        assert _activate(child, resolver) is True

        assert child.model == child_model
        assert child.provider == TEAM_PROVIDER
        assert resolver.models == [child_model]

    def test_child_does_not_mutate_the_parents_chain(self):
        parent_chain = get_fallback_chain(_config("true"))
        inherited = _inherited_chain(parent_chain)
        child = _make_agent(inherited, model="claude-haiku-5")
        assert _activate(child, _Resolver()) is True

        assert parent_chain[0]["model"] == "claude-sonnet-5"
        assert parent_chain[0][PRESERVE_REQUESTED_MODEL_KEY] is True

    def test_child_inherits_the_fail_closed_behavior(self):
        inherited = _inherited_chain(get_fallback_chain(_config("'true'")))
        child = _make_agent(inherited, model="claude-haiku-5")
        resolver = _Resolver()
        assert _activate(child, resolver) is False

        assert resolver.calls == []
        assert child.model == "claude-haiku-5"
        assert child.provider == PRIMARY_PROVIDER
