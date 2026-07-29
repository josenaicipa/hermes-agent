"""Selective / names-only skills index for the stable system-prompt tier.

``skills.index_mode`` (config.yaml, no env var):

* ``full`` (default) — backward-compatible: category + name + description.
* ``names_only`` — names and categories stay visible; long descriptions are
  omitted from the system-prompt index. ``skill_view`` still loads full
  content on explicit demand.

Cache-safe: the mode is part of the in-process skills prompt cache key so a
full entry never leaks into a names-only session (and vice versa). Stable for
the life of a conversation when config is fixed at session start.
"""

from unittest.mock import patch

import pytest

from agent.prompt_builder import (
    build_skills_system_prompt,
    clear_skills_system_prompt_cache,
)


@pytest.fixture(autouse=True)
def _clear_skills_cache():
    clear_skills_system_prompt_cache(clear_snapshot=True)
    yield
    clear_skills_system_prompt_cache(clear_snapshot=True)


def _write_skill(tmp_path, category: str, name: str, description: str) -> None:
    d = tmp_path / "skills" / category / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n"
        f"# {name}\n\nFull body for {name} with lots of procedure detail.\n",
        encoding="utf-8",
    )


class TestSkillsIndexMode:
    def test_full_default_includes_descriptions(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_skill(
            tmp_path,
            "mlops",
            "heavy-training",
            "Very long description of a heavy training skill with many steps",
        )

        # Explicit full + implicit default must both include the description.
        full = build_skills_system_prompt(index_mode="full")
        assert "heavy-training" in full
        assert "Very long description of a heavy training skill" in full
        assert "partially relevant" in full  # aggressive default guidance

        clear_skills_system_prompt_cache(clear_snapshot=True)
        defaulted = build_skills_system_prompt()
        assert "Very long description of a heavy training skill" in defaulted

    def test_names_only_omits_descriptions_keeps_names(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_skill(
            tmp_path,
            "mlops",
            "heavy-training",
            "Very long description of a heavy training skill with many steps",
        )
        _write_skill(
            tmp_path,
            "github",
            "pr-review",
            "Review pull requests carefully",
        )

        result = build_skills_system_prompt(index_mode="names_only")
        assert "heavy-training" in result
        assert "pr-review" in result
        assert "mlops" in result
        assert "github" in result
        # Long descriptions must not enter the stable system prompt.
        assert "Very long description of a heavy training skill" not in result
        assert "Review pull requests carefully" not in result
        # Selective loading guidance (no partial-match pressure).
        assert "partially relevant" not in result
        assert "skill_view" in result
        assert "hermes-agent" in result
        assert "AGENT_CONTEXT" in result or "Context Fabric" in result

    def test_index_mode_is_isolated_in_cache_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_skill(
            tmp_path,
            "mlops",
            "heavy-training",
            "Very long description of a heavy training skill with many steps",
        )

        names_only = build_skills_system_prompt(index_mode="names_only")
        assert "Very long description" not in names_only

        # A subsequent full build must not be served from the names-only entry.
        full = build_skills_system_prompt(index_mode="full")
        assert "Very long description of a heavy training skill" in full

        # And flipping back must still omit descriptions.
        again = build_skills_system_prompt(index_mode="names_only")
        assert "Very long description" not in again
        assert "heavy-training" in again

    def test_explicit_skill_view_still_loads_full_content(self, monkeypatch, tmp_path):
        """Index mode only affects the system-prompt index, not skill_view."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_skill(
            tmp_path,
            "mlops",
            "heavy-training",
            "Very long description of a heavy training skill with many steps",
        )

        index = build_skills_system_prompt(index_mode="names_only")
        assert "heavy-training" in index
        assert "Full body for heavy-training" not in index

        from tools.skills_tool import skill_view

        loaded = skill_view("heavy-training")
        # skill_view returns a JSON string with the full SKILL.md body.
        assert "heavy-training" in loaded
        assert "Full body for heavy-training" in loaded
        assert "Very long description of a heavy training skill" in loaded
