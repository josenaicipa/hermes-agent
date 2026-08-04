"""Auxiliary fallback must honor ``preserve_requested_model`` (Nemo HIGH).

The live incident had two halves.  The main agent half is covered by
``tests/run_agent/test_fallback_preserve_requested_model.py``.  This file covers
the **auxiliary** half, which was still silently downgrading:

The primary route is ``claude-opus-5`` on the local Claude bridge
(``claude-sdk-local``, 127.0.0.1:4318).  Auxiliary tasks on ``provider: auto``
run on that same route, so when 4318 reports exhausted credits they walk the
same chain the main agent walks — including the mirrored 4319 bridge entry that
declares ``model: claude-sonnet-5`` for readability and
``preserve_requested_model: true`` to say *"same model, other endpoint"*.

``_try_main_fallback_chain`` / ``_try_configured_fallback_chain`` ignored the
flag and requested the entry's configured model, so title generation,
compression and friends silently answered from Sonnet while the operator's
config said the model must be preserved.  Fixed here, with three hard bounds:

* substitution by the provider router is refused (entry skipped, chain walks on)
* cross-family / unsupported models are never preserved
* entries that do not opt in behave exactly as before
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_cli.fallback_config import PRESERVE_REQUESTED_MODEL_KEY

MAIN_MODEL = "claude-opus-5"
MAIN_PROVIDER = "claude-sdk-local"
TEAM_PROVIDER = "claude-sdk-team-local"
TEAM_URL = "http://127.0.0.1:4319/v1"
AUX_MODEL = "claude-haiku-5"


def _team_entry(**overrides):
    entry = {
        "provider": TEAM_PROVIDER,
        "model": "claude-sonnet-5",
        "base_url": TEAM_URL,
    }
    entry.update(overrides)
    return entry


class _Resolver:
    """Records the entry handed to ``_resolve_fallback_entry``."""

    def __init__(self, resolved_model=None, client=None):
        self.entries = []
        self.last_object = None
        self._resolved_model = resolved_model
        self._client = client

    def __call__(self, entry):
        self.entries.append(dict(entry))
        self.last_object = entry
        client = self._client if self._client is not None else MagicMock()
        resolved = (
            self._resolved_model
            if self._resolved_model is not None
            else str(entry.get("model") or "") or None
        )
        return client, resolved

    @property
    def models(self):
        return [str(e.get("model") or "") for e in self.entries]


@pytest.fixture()
def main_route(monkeypatch):
    """Pin the aux runtime to the 4318 bridge running Opus 5."""
    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_provider", lambda: MAIN_PROVIDER
    )
    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_model", lambda: MAIN_MODEL
    )
    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_model_for_aux", lambda: MAIN_MODEL
    )
    monkeypatch.setattr(
        "agent.auxiliary_client._is_provider_unhealthy", lambda label: False
    )
    # Never probe a live endpoint for a context window in these tests.
    monkeypatch.setattr(
        "agent.auxiliary_client.get_model_context_length",
        lambda model, **kwargs: 1_000_000,
    )


def _main_chain(monkeypatch, chain):
    monkeypatch.setattr(
        "hermes_cli.fallback_config.get_fallback_chain", lambda cfg: chain
    )


def _task_chain(monkeypatch, chain, task="compression"):
    monkeypatch.setattr(
        "agent.auxiliary_client._get_auxiliary_task_config",
        lambda t: {"fallback_chain": chain} if t == task else {},
    )


# ── Main fallback chain (top-level fallback_providers / fallback_model) ────


class TestMainChainPreservesTheRequestedModel:
    def test_opus_survives_the_mirrored_4318_to_4319_switch(self, monkeypatch, main_route):
        from agent.auxiliary_client import _try_main_fallback_chain

        chain = [_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})]
        _main_chain(monkeypatch, chain)
        resolver = _Resolver()

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            client, model, label = _try_main_fallback_chain(
                "title_generation", MAIN_PROVIDER
            )

        assert client is not None
        assert model == MAIN_MODEL, (
            "The auxiliary route mirrored onto 4319 must keep Opus 5 — "
            "requesting the entry's claude-sonnet-5 is the silent downgrade."
        )
        assert resolver.models == [MAIN_MODEL]
        assert resolver.entries[0]["base_url"] == TEAM_URL
        assert label == TEAM_PROVIDER

    def test_the_shared_chain_entry_is_never_mutated(self, monkeypatch, main_route):
        from agent.auxiliary_client import _try_main_fallback_chain

        chain = [_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})]
        _main_chain(monkeypatch, chain)

        with patch("agent.auxiliary_client._resolve_fallback_entry", _Resolver()):
            _try_main_fallback_chain("title_generation", MAIN_PROVIDER)

        assert chain[0]["model"] == "claude-sonnet-5"

    def test_without_the_opt_in_the_configured_model_still_wins(self, monkeypatch, main_route):
        """Backward compatibility: this is the historical behavior."""
        from agent.auxiliary_client import _try_main_fallback_chain

        chain = [_team_entry()]
        _main_chain(monkeypatch, chain)
        resolver = _Resolver()

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            _client, model, _label = _try_main_fallback_chain(
                "title_generation", MAIN_PROVIDER
            )

        assert model == "claude-sonnet-5"
        assert resolver.models == ["claude-sonnet-5"]

    def test_explicit_false_behaves_like_no_key(self, monkeypatch, main_route):
        from agent.auxiliary_client import _try_main_fallback_chain

        _main_chain(monkeypatch, [_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: False})])
        resolver = _Resolver()

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            _client, model, _label = _try_main_fallback_chain(
                "title_generation", MAIN_PROVIDER
            )

        assert model == "claude-sonnet-5"

    def test_opted_out_entries_are_resolved_from_the_original_object(
        self, monkeypatch, main_route
    ):
        """No copy, no key loss, no behavior change when the flag is off."""
        from agent.auxiliary_client import _try_main_fallback_chain

        chain = [_team_entry()]
        _main_chain(monkeypatch, chain)
        resolver = _Resolver()

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            _try_main_fallback_chain("title_generation", MAIN_PROVIDER)

        assert resolver.last_object is chain[0]


class TestMainChainFailsClosed:
    def test_invalid_opt_in_value_skips_the_entry_and_walks_on(self, monkeypatch, main_route):
        from agent.auxiliary_client import _try_main_fallback_chain

        _main_chain(monkeypatch, [
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: "yes"}),
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-5"},
        ])
        resolver = _Resolver()

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            _client, model, label = _try_main_fallback_chain(
                "title_generation", MAIN_PROVIDER
            )

        assert resolver.models == ["anthropic/claude-sonnet-5"]
        assert model == "anthropic/claude-sonnet-5"
        assert label == "openrouter"

    def test_invalid_opt_in_value_alone_exhausts_the_chain(self, monkeypatch, main_route):
        from agent.auxiliary_client import _try_main_fallback_chain

        _main_chain(monkeypatch, [_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: 2})])
        resolver = _Resolver()

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            client, model, label = _try_main_fallback_chain(
                "title_generation", MAIN_PROVIDER
            )

        assert (client, model, label) == (None, None, "")
        assert resolver.entries == []

    def test_router_answering_with_another_model_is_refused(self, monkeypatch, main_route):
        from agent.auxiliary_client import _try_main_fallback_chain

        _main_chain(monkeypatch, [_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})])
        resolver = _Resolver(resolved_model="claude-sonnet-5")

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            client, model, label = _try_main_fallback_chain(
                "title_generation", MAIN_PROVIDER
            )

        assert (client, model, label) == (None, None, ""), (
            "A router that substitutes the model must be skipped, not used."
        )

    def test_vendor_prefix_reformatting_is_not_a_substitution(self, monkeypatch, main_route):
        from agent.auxiliary_client import _try_main_fallback_chain

        _main_chain(monkeypatch, [
            _team_entry(
                provider="openrouter",
                model="anthropic/claude-sonnet-5",
                **{PRESERVE_REQUESTED_MODEL_KEY: True},
            )
        ])
        resolver = _Resolver(resolved_model="anthropic/claude-opus-5")

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            client, model, _label = _try_main_fallback_chain(
                "title_generation", MAIN_PROVIDER
            )

        assert client is not None
        assert model == "anthropic/claude-opus-5"

    def test_cross_family_entry_is_never_preserved(self, monkeypatch, main_route):
        """A GPT mirror must not be handed a Claude model, opt-in or not."""
        from agent.auxiliary_client import _try_main_fallback_chain

        _main_chain(monkeypatch, [
            _team_entry(
                provider="openrouter",
                model="openai/gpt-5.4",
                **{PRESERVE_REQUESTED_MODEL_KEY: True},
            )
        ])
        resolver = _Resolver()

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            client, model, label = _try_main_fallback_chain(
                "title_generation", MAIN_PROVIDER
            )

        assert (client, model, label) == (None, None, "")
        assert resolver.entries == []

    def test_opt_in_without_a_known_aux_model_is_skipped(self, monkeypatch, main_route):
        from agent.auxiliary_client import _try_main_fallback_chain

        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_model_for_aux", lambda: ""
        )
        _main_chain(monkeypatch, [_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})])
        resolver = _Resolver()

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            client, _model, _label = _try_main_fallback_chain(
                "title_generation", MAIN_PROVIDER
            )

        assert client is None
        assert resolver.entries == []

    def test_preserved_candidate_is_still_context_screened(self, monkeypatch, main_route):
        """Compression's 64K floor applies to the preserved model too."""
        from agent.auxiliary_client import _try_main_fallback_chain

        _main_chain(monkeypatch, [
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True}),
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-5"},
        ])
        monkeypatch.setattr(
            "agent.auxiliary_client.get_model_context_length",
            lambda model, **kwargs: 8192 if model == MAIN_MODEL else 1_000_000,
        )
        resolver = _Resolver()

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            _client, model, _label = _try_main_fallback_chain(
                "compression", MAIN_PROVIDER
            )

        assert resolver.models == [MAIN_MODEL, "anthropic/claude-sonnet-5"]
        assert model == "anthropic/claude-sonnet-5"


class TestMainChainAnchor:
    def test_the_failed_aux_model_is_the_anchor_when_known(self, monkeypatch, main_route):
        """An aux call that failed on its own model preserves THAT model."""
        from agent.auxiliary_client import _try_main_fallback_chain

        _main_chain(monkeypatch, [_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})])
        resolver = _Resolver()

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            _client, model, _label = _try_main_fallback_chain(
                "title_generation", MAIN_PROVIDER, failed_model="claude-haiku-5"
            )

        assert model == "claude-haiku-5"
        assert resolver.models == ["claude-haiku-5"]


# ── Per-task chain (auxiliary.<task>.fallback_chain) ──────────────────────


class TestTaskChainPreservesTheRequestedModel:
    def test_preserves_the_failed_aux_model(self, monkeypatch, main_route):
        from agent.auxiliary_client import _try_configured_fallback_chain

        chain = [_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})]
        _task_chain(monkeypatch, chain)
        resolver = _Resolver()

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            client, model, label = _try_configured_fallback_chain(
                "compression", MAIN_PROVIDER, failed_model=MAIN_MODEL
            )

        assert client is not None
        assert model == MAIN_MODEL
        assert resolver.models == [MAIN_MODEL]
        assert "claude-sdk-team-local" in label
        assert chain[0]["model"] == "claude-sonnet-5"

    def test_falls_back_to_the_main_aux_model_as_the_anchor(self, monkeypatch, main_route):
        from agent.auxiliary_client import _try_configured_fallback_chain

        _task_chain(monkeypatch, [_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})])
        resolver = _Resolver()

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            _client, model, _label = _try_configured_fallback_chain(
                "compression", MAIN_PROVIDER
            )

        assert model == MAIN_MODEL

    def test_model_less_mirror_entry_preserves(self, monkeypatch, main_route):
        """Aux entries may omit ``model``; the opt-in then pins the anchor."""
        from agent.auxiliary_client import _try_configured_fallback_chain

        _task_chain(monkeypatch, [
            {
                "provider": TEAM_PROVIDER,
                "base_url": TEAM_URL,
                PRESERVE_REQUESTED_MODEL_KEY: True,
            }
        ])
        resolver = _Resolver()

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            _client, model, _label = _try_configured_fallback_chain(
                "compression", MAIN_PROVIDER, failed_model=MAIN_MODEL
            )

        assert model == MAIN_MODEL
        assert resolver.models == [MAIN_MODEL]

    def test_without_the_opt_in_nothing_changes(self, monkeypatch, main_route):
        from agent.auxiliary_client import _try_configured_fallback_chain

        chain = [_team_entry()]
        _task_chain(monkeypatch, chain)
        resolver = _Resolver()

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            _client, model, _label = _try_configured_fallback_chain(
                "compression", MAIN_PROVIDER, failed_model=MAIN_MODEL
            )

        assert model == "claude-sonnet-5"
        assert resolver.last_object is chain[0]

    def test_invalid_opt_in_value_skips_the_entry(self, monkeypatch, main_route):
        from agent.auxiliary_client import _try_configured_fallback_chain

        _task_chain(monkeypatch, [
            _team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: "yes"}),
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-5"},
        ])
        resolver = _Resolver()

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            _client, model, _label = _try_configured_fallback_chain(
                "compression", MAIN_PROVIDER, failed_model=MAIN_MODEL
            )

        assert resolver.models == ["anthropic/claude-sonnet-5"]
        assert model == "anthropic/claude-sonnet-5"

    def test_router_substitution_is_refused(self, monkeypatch, main_route):
        from agent.auxiliary_client import _try_configured_fallback_chain

        _task_chain(monkeypatch, [_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})])
        resolver = _Resolver(resolved_model="claude-sonnet-5")

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            client, model, label = _try_configured_fallback_chain(
                "compression", MAIN_PROVIDER, failed_model=MAIN_MODEL
            )

        assert (client, model, label) == (None, None, "")

    def test_cross_family_entry_is_never_preserved(self, monkeypatch, main_route):
        from agent.auxiliary_client import _try_configured_fallback_chain

        _task_chain(monkeypatch, [
            _team_entry(
                provider="openrouter",
                model="openai/gpt-5.4",
                **{PRESERVE_REQUESTED_MODEL_KEY: True},
            )
        ])
        resolver = _Resolver()

        with patch("agent.auxiliary_client._resolve_fallback_entry", resolver):
            client, model, label = _try_configured_fallback_chain(
                "compression", MAIN_PROVIDER, failed_model=MAIN_MODEL
            )

        assert (client, model, label) == (None, None, "")
        assert resolver.entries == []


class _PaymentError(Exception):
    status_code = 402


class TestProviderWideFailureKeepsTheAuxiliaryAnchor:
    """401/402 skip a credential surface but must not erase model identity."""

    @staticmethod
    def _payment_error():
        return _PaymentError("Payment Required")

    def test_sync_402_preserves_aux_model_not_main_model(self, monkeypatch, main_route):
        from agent.auxiliary_client import call_llm

        _task_chain(
            monkeypatch,
            [_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})],
            task="title_generation",
        )
        primary = MagicMock()
        primary.chat.completions.create.side_effect = self._payment_error()
        fallback = MagicMock()
        fallback.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="mirrored"))]
        )
        resolver = _Resolver(client=fallback)

        with patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(primary, AUX_MODEL),
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client._resolve_fallback_entry", resolver,
        ), patch("agent.auxiliary_client._try_main_fallback_chain") as main_chain:
            response = call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert response.choices[0].message.content == "mirrored"
        assert resolver.models == [AUX_MODEL]
        main_chain.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_402_preserves_aux_model_not_main_model(
        self, monkeypatch, main_route
    ):
        from agent.auxiliary_client import async_call_llm

        _task_chain(
            monkeypatch,
            [_team_entry(**{PRESERVE_REQUESTED_MODEL_KEY: True})],
            task="compression",
        )
        primary = MagicMock()
        primary.chat.completions.create = AsyncMock(side_effect=self._payment_error())
        sync_fallback = MagicMock()
        async_fallback = MagicMock()
        async_fallback.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="mirrored async"))]
            )
        )
        resolver = _Resolver(client=sync_fallback)

        with patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(primary, AUX_MODEL),
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client._resolve_fallback_entry", resolver,
        ), patch(
            "agent.auxiliary_client._to_async_client",
            return_value=(async_fallback, AUX_MODEL),
        ), patch("agent.auxiliary_client._try_main_fallback_chain") as main_chain:
            response = await async_call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert response.choices[0].message.content == "mirrored async"
        assert resolver.models == [AUX_MODEL]
        main_chain.assert_not_called()
