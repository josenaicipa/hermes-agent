"""Invariants for the independent-agent-network skill."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "autonomous-ai-agents"
    / "independent-agent-network"
    / "SKILL.md"
)


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def test_description_is_one_short_sentence(skill_text):
    match = re.search(r"^description:\s+(.*)$", skill_text, re.MULTILINE)
    assert match, "missing description frontmatter"
    description = match.group(1).strip().strip('"').strip("'")
    assert description.endswith(".")
    assert len(description) <= 60
    assert " " in description


def test_skill_points_at_native_surfaces(skill_text):
    assert "`terminal`" in skill_text
    assert "hermes network" in skill_text
    assert "--linear" in skill_text
    assert "never the secret" in skill_text
