"""Tests for Arcee Trinity Large Thinking per-model overrides.

Arcee Trinity Large Thinking is a reasoning model that wants:
- Fixed temperature=0.5 (vs the global default)
- Compression threshold=0.75 (delay compression to preserve reasoning context)

The helpers must match the bare model name, including when it arrives via
OpenRouter as ``arcee-ai/trinity-large-thinking``, but must NOT hit sibling
Arcee models like trinity-large-preview or trinity-mini.
"""

from __future__ import annotations

import pytest

from agent.auxiliary_client import (
    _compression_threshold_for_model,
    _fixed_temperature_for_model,
    _is_arcee_trinity_thinking,
    _is_codex_gpt55,
)


@pytest.mark.parametrize(
    "model",
    [
        "trinity-large-thinking",
        "arcee-ai/trinity-large-thinking",
        "Arcee-AI/Trinity-Large-Thinking",  # case-insensitive
        "  trinity-large-thinking  ",  # whitespace tolerant
    ],
)
def test_is_arcee_trinity_thinking_matches(model: str) -> None:
    assert _is_arcee_trinity_thinking(model) is True


@pytest.mark.parametrize(
    "model",
    [
        None,
        "",
        "trinity-large-preview",
        "arcee-ai/trinity-large-preview:free",
        "trinity-mini",
        "arcee-ai/trinity-mini",
        "trinity-large",  # prefix-only must not match
        "claude-sonnet-4.6",
        "gpt-5.4",
    ],
)
def test_is_arcee_trinity_thinking_rejects_non_matches(model) -> None:
    assert _is_arcee_trinity_thinking(model) is False


def test_fixed_temperature_for_trinity_thinking() -> None:
    assert _fixed_temperature_for_model("trinity-large-thinking") == 0.5
    assert _fixed_temperature_for_model("arcee-ai/trinity-large-thinking") == 0.5


def test_fixed_temperature_sibling_arcee_models_unaffected() -> None:
    # Preview and mini do not pin temperature — caller chooses its default.
    assert _fixed_temperature_for_model("trinity-large-preview") is None
    assert _fixed_temperature_for_model("trinity-mini") is None


def test_compression_threshold_for_trinity_thinking() -> None:
    assert _compression_threshold_for_model("trinity-large-thinking") == 0.75
    assert _compression_threshold_for_model("arcee-ai/trinity-large-thinking") == 0.75


def test_compression_threshold_default_none_for_other_models() -> None:
    # None means "leave the user's config value unchanged".
    assert _compression_threshold_for_model(None) is None
    assert _compression_threshold_for_model("") is None
    assert _compression_threshold_for_model("trinity-large-preview") is None
    assert _compression_threshold_for_model("claude-sonnet-4.6") is None
    assert _compression_threshold_for_model("kimi-k2") is None


# ---------------------------------------------------------------------------
# Codex gpt-5.5 compaction-threshold autoraise
#
# ChatGPT's Codex OAuth backend caps gpt-5.5 at a 272K window (verified live:
# ~330K-token request rejected with context_length_exceeded, ~250K accepted).
# The default 50% compaction trigger would fire at ~136K — half the usable
# window — so this route floors the trigger at 60% (lowered from the original
# 85%, which made summarizer payloads too large to digest). Only the Codex
# OAuth route is affected; the same slug on OpenAI direct / OpenRouter /
# Copilot exposes a larger window and keeps the user's global threshold. The
# floor never lowers a user-set threshold that already meets or exceeds it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.5-2026-04-23",  # dated snapshot
        "gpt-5.5-codex-mini",  # Codex variant of the 5.5 family (also 272K-capped)
        "openai/gpt-5.5",  # aggregator-prefixed (still on the codex route)
        "GPT-5.5",  # case-insensitive
        "  gpt-5.5  ",  # whitespace tolerant
    ],
)
def test_is_codex_gpt55_matches_on_codex_provider(model: str) -> None:
    assert _is_codex_gpt55(model, "openai-codex") is True


@pytest.mark.parametrize(
    "provider",
    ["openrouter", "openai", "copilot", "openai-api", "", None],
)
def test_is_codex_gpt55_rejects_non_codex_providers(provider) -> None:
    # gpt-5.5 on any non-Codex route keeps the larger window — no override.
    assert _is_codex_gpt55("gpt-5.5", provider) is False


@pytest.mark.parametrize(
    "model",
    ["gpt-5.4", "gpt-5", "gpt-5.55", "gpt-5.50", "", None],
)
def test_is_codex_gpt55_rejects_non_55_models(model) -> None:
    # gpt-5.55 / gpt-5.50 are different families and must NOT match — the
    # "gpt-5.5-" / "gpt-5.5." prefix guards require a separator after "5.5".
    assert _is_codex_gpt55(model, "openai-codex") is False


def test_compression_threshold_for_codex_gpt55() -> None:
    # 0.60, lowered from the original 0.85: compacting at ~231K made the
    # summarizer input large enough to time out the auxiliary model, so the
    # autoraise now stops at ~163K (60% of the 272K Codex window).
    assert _compression_threshold_for_model("gpt-5.5", "openai-codex") == 0.60
    assert _compression_threshold_for_model("gpt-5.5-pro", "openai-codex") == 0.60
    assert _compression_threshold_for_model("openai/gpt-5.5", "openai-codex") == 0.60


def test_compression_threshold_codex_gpt55_other_routes_unaffected() -> None:
    # Same slug, different route → no override (keep the user's config value).
    assert _compression_threshold_for_model("gpt-5.5", "openrouter") is None
    assert _compression_threshold_for_model("gpt-5.5", "openai") is None
    assert _compression_threshold_for_model("gpt-5.5", "copilot") is None
    assert _compression_threshold_for_model("openai/gpt-5.5") is None  # no provider


def test_compression_threshold_codex_gpt55_opt_out() -> None:
    # allow_codex_gpt55_autoraise=False reverts to the global default (None).
    assert (
        _compression_threshold_for_model(
            "gpt-5.5", "openai-codex", allow_codex_gpt55_autoraise=False
        )
        is None
    )


def test_compression_threshold_opt_out_does_not_disable_trinity() -> None:
    # The opt-out flag is scoped to the Codex gpt-5.5 autoraise; the Arcee
    # Trinity override must still apply when the flag is False.
    assert (
        _compression_threshold_for_model(
            "trinity-large-thinking", "openrouter", allow_codex_gpt55_autoraise=False
        )
        == 0.75
    )


# ---------------------------------------------------------------------------
# Effective-threshold resolution (agent_init): the Codex gpt-5.5 autoraise is
# a floor — max(user threshold, 0.60) — never a hard override that could
# silently lower a user-set compression.threshold.
# ---------------------------------------------------------------------------


def test_resolve_threshold_codex_gpt55_raises_default_with_notice() -> None:
    from agent.agent_init import _resolve_compression_threshold

    threshold, autoraised = _resolve_compression_threshold(
        0.50, "gpt-5.5", "openai-codex", codex_gpt55_autoraise=True
    )
    assert threshold == pytest.approx(0.60)
    assert autoraised == {"from": 0.50, "to": 0.60}


def test_resolve_threshold_codex_gpt55_preserves_higher_user_threshold() -> None:
    # A user-set compression.threshold of 0.75 must survive on Codex gpt-5.5
    # with the autoraise enabled — the 0.60 override is a floor, not a cap.
    from agent.agent_init import _resolve_compression_threshold

    threshold, autoraised = _resolve_compression_threshold(
        0.75, "gpt-5.5", "openai-codex", codex_gpt55_autoraise=True
    )
    assert threshold == pytest.approx(0.75)
    # No raise happened, so no "raised" notice either.
    assert autoraised is None


def test_resolve_threshold_codex_gpt55_equal_user_threshold_no_notice() -> None:
    from agent.agent_init import _resolve_compression_threshold

    threshold, autoraised = _resolve_compression_threshold(
        0.60, "gpt-5.5", "openai-codex", codex_gpt55_autoraise=True
    )
    assert threshold == pytest.approx(0.60)
    assert autoraised is None


def test_resolve_threshold_codex_gpt55_opt_out_keeps_user_threshold() -> None:
    from agent.agent_init import _resolve_compression_threshold

    threshold, autoraised = _resolve_compression_threshold(
        0.50, "gpt-5.5", "openai-codex", codex_gpt55_autoraise=False
    )
    assert threshold == pytest.approx(0.50)
    assert autoraised is None


def test_resolve_threshold_trinity_still_replaces_silently() -> None:
    # Arcee Trinity keeps its long-standing behavior: replace the global
    # threshold outright (even downward) with no notice.
    from agent.agent_init import _resolve_compression_threshold

    threshold, autoraised = _resolve_compression_threshold(
        0.90, "trinity-large-thinking", "openrouter", codex_gpt55_autoraise=True
    )
    assert threshold == pytest.approx(0.75)
    assert autoraised is None


def test_resolve_threshold_no_override_returns_user_threshold() -> None:
    from agent.agent_init import _resolve_compression_threshold

    threshold, autoraised = _resolve_compression_threshold(
        0.50, "claude-sonnet-4.6", "anthropic", codex_gpt55_autoraise=True
    )
    assert threshold == pytest.approx(0.50)
    assert autoraised is None
