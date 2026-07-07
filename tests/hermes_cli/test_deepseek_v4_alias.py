"""Regression test: the bare ``deepseek`` alias must resolve to a V4 model.

Context: ``resolve_alias()`` looks up the short alias in ``MODEL_ALIASES``,
then filters the provider's catalog to model IDs that start with the
alias's ``family`` string, and picks the highest-version match.

``MODEL_ALIASES["deepseek"]`` used ``family="deepseek-chat"`` — which only
ever matches the literal model ``"deepseek-chat"`` itself, because
DeepSeek's V4 models are named ``deepseek-v4-pro`` / ``deepseek-v4-flash``,
not ``deepseek-chat-v4``. Unlike the "sonnet"/"opus" aliases (whose family
prefix spans generations, e.g. "claude-sonnet" matches both
claude-sonnet-4 and claude-sonnet-5), the "deepseek" alias was pinned to V3
forever and could never benefit from a newer DeepSeek release — the exact
failure mode the actualidad-ai 2026-07-05 report's "migrate DeepSeek before
2026-07-24" recommendation flags: DeepSeek retires the legacy
``deepseek-chat``/``deepseek-reasoner`` IDs on that date.

Fix: point ``family`` at ``"deepseek-v4-flash"`` (matching the variant
Hermes' own config.yaml already prefers for auxiliary tasks), so the bare
alias resolves to a currently-supported model instead of one on a
retirement countdown.
"""

from __future__ import annotations


class TestDeepSeekBareAliasResolvesToV4:
    def test_family_prefix_no_longer_pins_v3(self):
        from hermes_cli.model_switch import MODEL_ALIASES

        identity = MODEL_ALIASES["deepseek"]
        assert identity.vendor == "deepseek"
        assert identity.family != "deepseek-chat", (
            "family must not be the literal V3 model id — it can never "
            "prefix-match deepseek-v4-* models, so the alias would stay "
            "pinned to V3 even after DeepSeek retires it on 2026-07-24"
        )

    def test_resolves_to_a_v4_model_against_the_catalog(self, monkeypatch):
        import hermes_cli.model_switch as ms

        # Fake catalog mirroring what the live deepseek provider currently
        # returns (see ~/.hermes/provider_models_cache.json): V4 + legacy.
        monkeypatch.setattr(
            ms,
            "list_provider_models",
            lambda provider: [
                "deepseek-v4-pro",
                "deepseek-v4-flash",
                "deepseek-chat",
                "deepseek-reasoner",
            ],
        )
        monkeypatch.setattr(ms, "DIRECT_ALIASES", {})

        result = ms.resolve_alias("deepseek", "deepseek")

        assert result is not None, "bare 'deepseek' alias must still resolve"
        provider, model, alias = result
        assert provider == "deepseek"
        assert alias == "deepseek"
        assert model.startswith("deepseek-v4"), (
            f"expected a V4 model, got {model!r} — the alias is still "
            "pinned to a legacy id"
        )

    def test_static_provider_models_fallback_also_has_v4(self):
        """`resolve_alias` merges a static `_PROVIDER_MODELS` list into the
        live catalog as a safety net for when models.dev hasn't synced yet
        (see resolve_alias's `static = _PROVIDER_MODELS.get(...)` merge).
        Confirm that static list was updated too — otherwise a live-fetch
        outage would silently re-pin the alias to V3 despite the family fix
        above, since resolve_alias merges both sources before matching."""
        from hermes_cli.models import _PROVIDER_MODELS

        static_models = _PROVIDER_MODELS.get("deepseek", [])
        assert any(m.startswith("deepseek-v4") for m in static_models), (
            "static _PROVIDER_MODELS fallback for deepseek has no V4 entry — "
            "resolve_alias would fall back to a legacy-only catalog if the "
            "live models.dev fetch is unavailable"
        )

    def test_falls_through_to_none_when_no_v4_available_anywhere(self, monkeypatch):
        """Genuine worst case: both the live catalog AND the static fallback
        lack any V4 entry (e.g. someone regresses the static list from the
        test above). The alias must degrade to "no match" rather than crash
        or silently resolve to something wrong — same contract as any other
        unmatched alias; the caller's `_resolve_alias_fallback` then tries
        other authenticated providers."""
        import hermes_cli.model_switch as ms

        monkeypatch.setattr(
            ms,
            "list_provider_models",
            lambda provider: ["deepseek-chat", "deepseek-reasoner"],
        )
        monkeypatch.setattr(ms, "DIRECT_ALIASES", {})
        monkeypatch.setattr("hermes_cli.models._PROVIDER_MODELS", {"deepseek": []})

        result = ms.resolve_alias("deepseek", "deepseek")
        assert result is None
