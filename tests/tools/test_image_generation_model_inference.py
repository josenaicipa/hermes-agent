"""Model-to-provider inference for image generation.

Covers the split-brain config that used to swallow requests silently:
``image_gen.model`` names a plugin model (e.g. ``gpt-image-2-high``) while
``image_gen.provider`` is unset. The old dispatcher fell through to the
in-tree FAL path, where ``_resolve_fal_model()`` does not recognise the ID
and quietly substitutes the FAL default — so the user got a different
backend and a different model than they picked, or (with no FAL_KEY) no
``image_generate`` tool at all.
"""

from __future__ import annotations

import json

import pytest

from agent import image_gen_registry
from agent.image_gen_provider import ImageGenProvider


@pytest.fixture(autouse=True)
def _reset_registry():
    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


class _FakeOpenAIProvider(ImageGenProvider):
    """Stands in for ``plugins/image_gen/openai`` — no key, so unavailable."""

    def __init__(self, available: bool = False):
        self._available = available

    @property
    def name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        return self._available

    def list_models(self):
        return [
            {"id": "gpt-image-2-low", "display": "GPT Image 2 (Low)"},
            {"id": "gpt-image-2-medium", "display": "GPT Image 2 (Medium)"},
            {"id": "gpt-image-2-high", "display": "GPT Image 2 (High)"},
        ]

    def default_model(self):
        return "gpt-image-2-medium"

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        return {
            "success": True,
            "image": "/tmp/openai-test.png",
            "model": "gpt-image-2-high",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "provider": "openai",
        }


@pytest.fixture
def _no_fal(monkeypatch):
    """Simulate a host with no FAL credentials at all."""
    from tools import image_generation_tool

    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setattr(image_generation_tool, "check_fal_api_key", lambda: False)


@pytest.fixture
def _config(monkeypatch):
    """Return a setter for the ``image_gen`` config section."""
    from tools import image_generation_tool

    def _set(provider=None, model=None):
        monkeypatch.setattr(
            image_generation_tool, "_read_configured_image_provider", lambda: provider
        )
        monkeypatch.setattr(
            image_generation_tool, "_read_configured_image_model", lambda: model
        )

    return _set


@pytest.fixture
def _stub_discovery(monkeypatch):
    from hermes_cli import plugins as plugins_module

    monkeypatch.setattr(
        plugins_module, "_ensure_plugins_discovered", lambda force=False: None
    )


class TestProviderOwningModel:
    def test_resolves_plugin_that_owns_the_model(self, _stub_discovery):
        from tools import image_generation_tool

        image_gen_registry.register_provider(_FakeOpenAIProvider())
        assert (
            image_generation_tool._provider_owning_model("gpt-image-2-high") == "openai"
        )

    def test_fal_models_stay_on_the_in_tree_path(self, _stub_discovery):
        from tools import image_generation_tool

        image_gen_registry.register_provider(_FakeOpenAIProvider())
        fal_model = next(iter(image_generation_tool.FAL_MODELS))
        assert image_generation_tool._provider_owning_model(fal_model) is None

    def test_unknown_model_resolves_to_nothing(self, _stub_discovery):
        from tools import image_generation_tool

        image_gen_registry.register_provider(_FakeOpenAIProvider())
        assert image_generation_tool._provider_owning_model("no-such-model") is None

    def test_empty_model_short_circuits(self, _stub_discovery):
        from tools import image_generation_tool

        assert image_generation_tool._provider_owning_model("") is None


class TestResolveImageProviderName:
    def test_explicit_provider_wins_over_model(self, _config, _stub_discovery):
        from tools import image_generation_tool

        image_gen_registry.register_provider(_FakeOpenAIProvider())
        _config(provider="krea", model="gpt-image-2-high")
        assert image_generation_tool._resolve_image_provider_name() == "krea"

    def test_falls_back_to_model_owner(self, _config, _stub_discovery):
        from tools import image_generation_tool

        image_gen_registry.register_provider(_FakeOpenAIProvider())
        _config(provider=None, model="gpt-image-2-high")
        assert image_generation_tool._resolve_image_provider_name() == "openai"

    def test_nothing_configured_resolves_to_none(self, _config, _stub_discovery):
        from tools import image_generation_tool

        _config(provider=None, model=None)
        assert image_generation_tool._resolve_image_provider_name() is None


class TestDispatchUsesInference:
    def test_dispatch_routes_by_model_when_provider_unset(
        self, _config, _stub_discovery, monkeypatch
    ):
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module

        image_gen_registry.register_provider(_FakeOpenAIProvider())
        _config(provider=None, model="gpt-image-2-high")
        monkeypatch.setattr(
            registry_module,
            "get_provider",
            lambda name: _FakeOpenAIProvider() if name == "openai" else None,
        )

        payload = json.loads(
            image_generation_tool._dispatch_to_plugin_provider("draw a cat", "square")
        )
        assert payload["success"] is True
        assert payload["provider"] == "openai"
        assert payload["model"] == "gpt-image-2-high"

    def test_fal_model_still_falls_through_to_in_tree_path(
        self, _config, _stub_discovery
    ):
        from tools import image_generation_tool

        image_gen_registry.register_provider(_FakeOpenAIProvider())
        fal_model = next(iter(image_generation_tool.FAL_MODELS))
        _config(provider=None, model=fal_model)

        assert (
            image_generation_tool._dispatch_to_plugin_provider("draw a cat", "square")
            is None
        )

    def test_inferred_provider_error_names_the_model(
        self, _config, _stub_discovery, monkeypatch
    ):
        """A model owned by an uninstalled plugin explains itself in model terms."""
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module

        image_gen_registry.register_provider(_FakeOpenAIProvider())
        _config(provider=None, model="gpt-image-2-high")
        monkeypatch.setattr(registry_module, "get_provider", lambda name: None)

        payload = json.loads(
            image_generation_tool._dispatch_to_plugin_provider("draw a cat", "square")
        )
        assert payload["success"] is False
        assert payload["error_type"] == "provider_not_registered"
        assert "image_gen.model='gpt-image-2-high'" in payload["error"]


class TestToolStaysExposed:
    def test_configured_but_unavailable_backend_keeps_tool_exposed(
        self, _no_fal, _config, _stub_discovery
    ):
        """The regression the user hit: tool vanished instead of erroring.

        With ``image_gen.model: gpt-image-2-high`` set but no
        ``OPENAI_API_KEY``, the check used to return False, so
        ``image_generate`` was dropped from the agent's schema list with
        nothing but a debug log — the agent could only report that no
        image tool existed. It must stay exposed so the call returns the
        provider's actionable "set OPENAI_API_KEY" error instead.
        """
        from tools import image_generation_tool

        image_gen_registry.register_provider(_FakeOpenAIProvider(available=False))
        _config(provider=None, model="gpt-image-2-high")

        assert image_generation_tool.check_image_generation_requirements() is True

    def test_explicit_provider_keeps_tool_exposed(
        self, _no_fal, _config, _stub_discovery
    ):
        from tools import image_generation_tool

        image_gen_registry.register_provider(_FakeOpenAIProvider(available=False))
        _config(provider="openai", model=None)

        assert image_generation_tool.check_image_generation_requirements() is True

    def test_nothing_configured_and_no_backend_stays_hidden(
        self, _no_fal, _config, _stub_discovery
    ):
        from tools import image_generation_tool

        image_gen_registry.register_provider(_FakeOpenAIProvider(available=False))
        _config(provider=None, model=None)

        assert image_generation_tool.check_image_generation_requirements() is False

    def test_available_plugin_exposes_tool_without_config(
        self, _no_fal, _config, _stub_discovery
    ):
        from tools import image_generation_tool

        image_gen_registry.register_provider(_FakeOpenAIProvider(available=True))
        _config(provider=None, model=None)

        assert image_generation_tool.check_image_generation_requirements() is True
