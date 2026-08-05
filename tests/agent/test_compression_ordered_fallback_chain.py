"""Ordered compression fallback: Gemini CLI -> Codex -> Kimi Code CLI.

Before this change the configured ``auxiliary.compression.fallback_chain``
only ever contributed ONE candidate per failure: the first entry that could
be *built*. If that candidate's call then failed, the error propagated and
entries 1..n were never attempted — so a three-entry chain behaved like a
one-entry chain for every ordinary retryable adapter failure (not just
HTTP 402).

These tests pin the ordering contract:

* primary success  -> no fallback at all;
* primary fails    -> entry 0 (Gemini CLI);
* entry 0 fails    -> entry 1 (Codex/Terra), and Kimi is NOT touched;
* 0 and 1 fail     -> entry 2 (Kimi Code CLI);
* everything fails -> the original error surfaces (fail closed);
* a non-retryable failure (auth/permission/config) does NOT widen routing.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import agent.auxiliary_client as ac

GEMINI = {"provider": "google-gemini-cli", "model": "gemini-3.6-flash-low"}
TERRA = {"provider": "openai-codex", "model": "gpt-5.6-terra"}
KIMI = {"provider": "kimi-code-cli", "model": "kimi-code/k3"}
CHAIN = [GEMINI, TERRA, KIMI]


def _response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None),
            finish_reason="stop")],
        usage=None,
        model="m",
    )


class _Client:
    def __init__(self, name):
        self.name = name
        self.base_url = f"{name}://local"


class _ChainHarness:
    """Drives ``_run_configured_chain_sync`` over a fake three-entry chain.

    ``outcomes`` maps a provider id to either an exception to raise or a
    response to return. Records the exact order of attempts.
    """

    def __init__(self, outcomes, chain=None, unbuildable=()):
        self.outcomes = outcomes
        self.chain = chain if chain is not None else CHAIN
        self.unbuildable = set(unbuildable)
        self.attempts = []

    def resolve_entry(self, entry):
        provider = entry.get("provider")
        if provider in self.unbuildable:
            return None, None
        return _Client(provider), entry.get("model")

    def call_candidate(self, fb_client, fb_model, fb_label, **kwargs):
        self.attempts.append((fb_client.name, fb_model, fb_label))
        outcome = self.outcomes.get(fb_client.name)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def run(self, **overrides):
        kwargs = dict(
            task="compression",
            # "auto" keeps every chain entry eligible; the primary-provider
            # topology is exercised separately in TestPrimaryProviderTopology.
            failed_provider="auto",
            reason="connection error",
            failed_model=None,
            requested_model=None,
            messages=[{"role": "user", "content": "x"}],
            temperature=None,
            max_tokens=None,
            tools=None,
            effective_timeout=300.0,
            effective_extra_body={},
            reasoning_config=None,
        )
        kwargs.update(overrides)
        with patch.object(ac, "_get_auxiliary_task_config",
                          lambda task: {"fallback_chain": self.chain}), \
             patch.object(ac, "_resolve_fallback_entry", self.resolve_entry), \
             patch.object(ac, "_call_fallback_candidate_sync",
                          self.call_candidate), \
             patch.object(ac, "_task_minimum_context_length", lambda task: None), \
             patch.object(ac, "_read_main_provider", lambda: ""), \
             patch.object(ac, "_read_main_model_for_aux", lambda: ""):
            return ac._run_configured_chain_sync(**kwargs)

    @property
    def providers_attempted(self):
        return [a[0] for a in self.attempts]


class _CLIConnectionError(RuntimeError):
    """Stand-in adapter transport error.

    The class name contains ``Connection``, which is exactly how the real
    CLI adapters (``AgyCLIConnectionError``, ``KimiCodeCLIConnectionError``)
    make themselves visible to Hermes' canonical retryable classifier
    without a provider-specific branch.
    """


def _retryable(msg="upstream connection reset"):
    return _CLIConnectionError(msg)


# ── entry resumption primitive ───────────────────────────────────────────


class TestChainStartIndex:
    def test_default_starts_at_entry_zero(self):
        seen = []

        def _resolve(entry):
            seen.append(entry["provider"])
            return _Client(entry["provider"]), entry["model"]

        with patch.object(ac, "_get_auxiliary_task_config",
                          lambda task: {"fallback_chain": CHAIN}), \
             patch.object(ac, "_resolve_fallback_entry", _resolve), \
             patch.object(ac, "_task_minimum_context_length", lambda task: None), \
             patch.object(ac, "_read_main_provider", lambda: ""):
            client, model, label = ac._try_configured_fallback_chain(
                "compression", "primary")
        assert seen == ["google-gemini-cli"]
        assert label == "fallback_chain[0](google-gemini-cli)"
        assert model == "gemini-3.6-flash-low"

    def test_start_index_resumes_mid_chain(self):
        seen = []

        def _resolve(entry):
            seen.append(entry["provider"])
            return _Client(entry["provider"]), entry["model"]

        with patch.object(ac, "_get_auxiliary_task_config",
                          lambda task: {"fallback_chain": CHAIN}), \
             patch.object(ac, "_resolve_fallback_entry", _resolve), \
             patch.object(ac, "_task_minimum_context_length", lambda task: None), \
             patch.object(ac, "_read_main_provider", lambda: ""):
            client, model, label = ac._try_configured_fallback_chain(
                "compression", "primary", start_index=2)
        assert seen == ["kimi-code-cli"]
        assert label == "fallback_chain[2](kimi-code-cli)"
        assert model == "kimi-code/k3"

    def test_start_index_past_the_end_yields_nothing(self):
        with patch.object(ac, "_get_auxiliary_task_config",
                          lambda task: {"fallback_chain": CHAIN}), \
             patch.object(ac, "_task_minimum_context_length", lambda task: None), \
             patch.object(ac, "_read_main_provider", lambda: ""):
            assert ac._try_configured_fallback_chain(
                "compression", "primary", start_index=3) == (None, None, "")

    def test_label_index_parser(self):
        assert ac._chain_entry_index("fallback_chain[2](kimi-code-cli)") == 2
        assert ac._chain_entry_index("main:openai-codex") is None


# ── ordered walk ─────────────────────────────────────────────────────────


class TestOrderedFallback:
    def test_gemini_success_never_touches_terra_or_kimi(self):
        h = _ChainHarness({"google-gemini-cli": _response("gemini summary")})
        resp = h.run()
        assert resp.choices[0].message.content == "gemini summary"
        assert h.providers_attempted == ["google-gemini-cli"]

    def test_gemini_failure_falls_to_terra_and_stops(self):
        h = _ChainHarness({
            "google-gemini-cli": _retryable("agy CLI connection reset"),
            "openai-codex": _response("terra summary"),
            "kimi-code-cli": _response("kimi summary"),
        })
        resp = h.run()
        assert resp.choices[0].message.content == "terra summary"
        # The critical assertion: Kimi is NOT attempted when Terra succeeds.
        assert h.providers_attempted == ["google-gemini-cli", "openai-codex"]
        assert "kimi-code-cli" not in h.providers_attempted

    def test_gemini_and_terra_failure_reaches_kimi(self):
        h = _ChainHarness({
            "google-gemini-cli": _retryable("agy CLI connection reset"),
            "openai-codex": _retryable("codex stream ended prematurely"),
            "kimi-code-cli": _response("kimi summary"),
        })
        resp = h.run()
        assert resp.choices[0].message.content == "kimi summary"
        assert h.providers_attempted == [
            "google-gemini-cli", "openai-codex", "kimi-code-cli",
        ]

    def test_all_three_failing_exhausts_the_chain(self):
        h = _ChainHarness({
            "google-gemini-cli": _retryable("agy connection reset"),
            "openai-codex": _retryable("codex connection reset"),
            "kimi-code-cli": _retryable("kimi-code-cli connection reset"),
        })
        assert h.run() is None  # caller re-raises the original error
        assert h.providers_attempted == [
            "google-gemini-cli", "openai-codex", "kimi-code-cli",
        ]

    def test_ordering_is_declaration_order_not_alphabetical(self):
        h = _ChainHarness(
            {
                "kimi-code-cli": _retryable("x"),
                "openai-codex": _retryable("y"),
                "google-gemini-cli": _response("last"),
            },
            chain=[KIMI, TERRA, GEMINI],
        )
        resp = h.run()
        assert resp.choices[0].message.content == "last"
        assert h.providers_attempted == [
            "kimi-code-cli", "openai-codex", "google-gemini-cli",
        ]

    def test_unbuildable_entry_is_skipped_without_consuming_the_chain(self):
        """kimi-code-cli unconfigured -> chain still reaches later entries."""
        h = _ChainHarness(
            {"openai-codex": _response("terra summary")},
            chain=[KIMI, TERRA],
            unbuildable={"kimi-code-cli"},
        )
        resp = h.run()
        assert resp.choices[0].message.content == "terra summary"
        assert h.providers_attempted == ["openai-codex"]


class TestPrimaryProviderTopology:
    """The recommended shape: Gemini is the task's PRIMARY provider and the
    chain declares only the two fallbacks (Terra, then Kimi)."""

    FALLBACKS = [TERRA, KIMI]

    def test_terra_serves_when_the_gemini_primary_fails(self):
        h = _ChainHarness(
            {
                "openai-codex": _response("terra summary"),
                "kimi-code-cli": _response("kimi summary"),
            },
            chain=self.FALLBACKS,
        )
        resp = h.run(failed_provider="google-gemini-cli")
        assert resp.choices[0].message.content == "terra summary"
        assert h.providers_attempted == ["openai-codex"]
        assert "kimi-code-cli" not in h.providers_attempted

    def test_kimi_serves_when_gemini_and_terra_fail(self):
        h = _ChainHarness(
            {
                "openai-codex": _retryable("codex connection reset"),
                "kimi-code-cli": _response("kimi summary"),
            },
            chain=self.FALLBACKS,
        )
        resp = h.run(failed_provider="google-gemini-cli")
        assert resp.choices[0].message.content == "kimi summary"
        assert h.providers_attempted == ["openai-codex", "kimi-code-cli"]

    def test_all_routes_failing_exhausts_explicitly(self):
        h = _ChainHarness(
            {
                "openai-codex": _retryable("codex connection reset"),
                "kimi-code-cli": _retryable("kimi-code-cli connection reset"),
            },
            chain=self.FALLBACKS,
        )
        assert h.run(failed_provider="google-gemini-cli") is None
        assert h.providers_attempted == ["openai-codex", "kimi-code-cli"]

    def test_duplicate_primary_entry_stays_credential_skipped(self):
        """Existing scope semantics are unchanged: a chain entry naming the
        provider that just failed provider-wide is still skipped."""
        h = _ChainHarness(
            {
                "google-gemini-cli": _response("must not be reached"),
                "openai-codex": _response("terra summary"),
            },
            chain=CHAIN,
        )
        resp = h.run(failed_provider="google-gemini-cli")
        assert resp.choices[0].message.content == "terra summary"
        assert h.providers_attempted == ["openai-codex"]


class TestFailClosedBoundaries:
    def test_non_retryable_error_raises_and_does_not_widen_routing(self):
        """An auth/permission/config error must not walk other credentials."""
        boom = PermissionError("permission denied writing session")
        h = _ChainHarness({
            "google-gemini-cli": boom,
            "openai-codex": _response("must not be reached"),
        })
        with pytest.raises(PermissionError):
            h.run()
        assert h.providers_attempted == ["google-gemini-cli"]

    def test_value_error_from_adapter_raises(self):
        h = _ChainHarness({
            "google-gemini-cli": ValueError("adapter does not support images"),
            "openai-codex": _response("must not be reached"),
        })
        with pytest.raises(ValueError):
            h.run()
        assert h.providers_attempted == ["google-gemini-cli"]

    def test_stale_credential_quarantine_advances_the_chain(self):
        # _call_fallback_candidate_sync returns None for a quarantined
        # candidate; that must advance rather than end the walk.
        h = _ChainHarness({
            "google-gemini-cli": None,
            "openai-codex": _response("terra summary"),
        })
        resp = h.run()
        assert resp.choices[0].message.content == "terra summary"
        assert h.providers_attempted == ["google-gemini-cli", "openai-codex"]

    def test_empty_chain_returns_none(self):
        h = _ChainHarness({}, chain=[])
        assert h.run() is None
        assert h.providers_attempted == []


class TestWalkIsBounded:
    """The walk must terminate even when the resolver does not advance.

    Existing tests stub ``_try_configured_fallback_chain`` with a *fixed*
    return value that ignores ``start_index``; the walk must not spin on it.
    """

    def _run(self, chain_return, candidate_side_effect):
        calls = {"n": 0}

        def _candidate(*_a, **_kw):
            calls["n"] += 1
            if isinstance(candidate_side_effect, BaseException):
                raise candidate_side_effect
            return candidate_side_effect

        with patch.object(ac, "_get_auxiliary_task_config",
                          lambda task: {"fallback_chain": CHAIN}), \
             patch.object(ac, "_try_configured_fallback_chain",
                          return_value=chain_return), \
             patch.object(ac, "_call_fallback_candidate_sync", _candidate):
            result = ac._run_configured_chain_sync(
                task="compression", failed_provider="auto",
                reason="connection error", failed_model=None,
                requested_model=None, messages=[], temperature=None,
                max_tokens=None, tools=None, effective_timeout=300.0,
                effective_extra_body={}, reasoning_config=None)
        return result, calls["n"]

    def test_non_advancing_stub_terminates(self):
        result, n = self._run(
            (_Client("openai-codex"), "gpt-5.6-terra",
             "fallback_chain[0](openai-codex)"),
            _retryable("connection reset"),
        )
        assert result is None
        assert n == 1, "the same entry must not be retried in a loop"

    def test_non_advancing_stub_still_returns_a_success(self):
        result, n = self._run(
            (_Client("openai-codex"), "gpt-5.6-terra",
             "fallback_chain[0](openai-codex)"),
            _response("terra summary"),
        )
        assert result.choices[0].message.content == "terra summary"
        assert n == 1

    def test_unlabelled_candidate_terminates(self):
        result, n = self._run(
            (_Client("openai-codex"), "gpt-5.6-terra", "main-agent(openrouter)"),
            _retryable("connection reset"),
        )
        assert result is None
        assert n == 1


class TestRetryablePredicate:
    def test_connection_and_timeout_named_adapter_errors_are_retryable(self):
        class _CLIConnectionError(RuntimeError):
            pass

        class _CLITimeout(RuntimeError):
            pass

        assert ac._is_retryable_chain_error(_CLIConnectionError("reset"))
        assert ac._is_retryable_chain_error(_CLITimeout("timed out"))

    def test_kimi_and_agy_adapter_errors_are_retryable(self):
        from agent.agy_cli_client import AgyCLIConnectionError, AgyCLITimeout
        from agent.kimi_code_cli_client import (
            KimiCodeCLIConnectionError,
            KimiCodeCLITimeout,
        )

        for exc in (
            AgyCLIConnectionError("agy CLI failed with non-zero exit status 1"),
            AgyCLITimeout("agy CLI timed out after 300.0s"),
            KimiCodeCLIConnectionError("no final assistant response"),
            KimiCodeCLITimeout("kimi-code-cli timed out after 300.0s"),
        ):
            assert ac._is_retryable_chain_error(exc), exc

    def test_configuration_error_is_not_retryable(self):
        from agent.kimi_code_cli_client import KimiCodeCLIConfigurationError

        assert not ac._is_retryable_chain_error(
            KimiCodeCLIConfigurationError("tool_policy missing")
        )

    def test_plain_errors_are_not_retryable(self):
        assert not ac._is_retryable_chain_error(ValueError("bad request shape"))
        assert not ac._is_retryable_chain_error(PermissionError("denied"))
