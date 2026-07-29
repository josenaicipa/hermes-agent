"""Tests for the blueprints layer (skill frontmatter <-> cron automation bridge).

A blueprint is a skill with a metadata.hermes.blueprint block. These verify parsing,
the create-job bridge, and the export round-trip without touching the real
cron store.
"""

from unittest.mock import patch

import pytest

from tools.blueprints import (
    BlueprintError,
    BlueprintSpec,
    blueprint_to_job_spec,
    create_blueprint_job,
    export_blueprint,
    parse_blueprint,
    blueprint_spec_for_installed,
)


BLUEPRINT_SKILL = """---
name: morning-brief
description: Summarize unread email and calendar every morning.
version: 1.0.0
metadata:
  hermes:
    tags: [blueprint, email]
    blueprint:
      schedule: "0 8 * * *"
      deliver: telegram
      prompt: "Summarize my unread email and today's calendar."
      category: justified_cadence
      material_result_criterion: archivo_entregado
---

# Morning Brief

Every morning, gather unread email and the day's calendar and send a digest.
"""

PLAIN_SKILL = """---
name: not-a-blueprint
description: Just a regular skill.
metadata:
  hermes:
    tags: [misc]
---

# Not a blueprint
"""

MALFORMED_BLUEPRINT = """---
name: broken
description: Blueprint with no schedule.
metadata:
  hermes:
    blueprint:
      deliver: origin
      category: justified_cadence
      material_result_criterion: archivo_entregado
---

# Broken
"""

MISSING_ADMISSION_BLUEPRINT = """---
name: missing-admission
description: LLM blueprint without admission declarations.
metadata:
  hermes:
    blueprint:
      schedule: "0 9 * * *"
      prompt: "Do the thing."
---

# Missing admission
"""


class TestParseBlueprint:
    def test_parses_full_blueprint(self):
        spec = parse_blueprint(BLUEPRINT_SKILL)
        assert spec is not None
        assert spec.skill_name == "morning-brief"
        assert spec.schedule == "0 8 * * *"
        assert spec.deliver == "telegram"
        assert spec.prompt is not None and spec.prompt.startswith("Summarize")
        assert spec.category == "justified_cadence"
        assert spec.material_result_criterion == "archivo_entregado"

    def test_plain_skill_is_not_a_blueprint(self):
        assert parse_blueprint(PLAIN_SKILL) is None

    def test_no_frontmatter_is_not_a_blueprint(self):
        assert parse_blueprint("just some text, no frontmatter") is None

    def test_missing_schedule_raises(self):
        with pytest.raises(BlueprintError):
            parse_blueprint(MALFORMED_BLUEPRINT)

    def test_blueprint_not_mapping_raises(self):
        bad = "---\nname: x\nmetadata:\n  hermes:\n    blueprint: not-a-dict\n---\n\nbody"
        with pytest.raises(BlueprintError):
            parse_blueprint(bad)

    def test_deliver_defaults_to_origin(self):
        skill = (
            "---\nname: r\ndescription: d\nmetadata:\n  hermes:\n"
            '    blueprint:\n      schedule: "every 1h"\n'
            "      category: event\n"
            "      material_result_criterion: alerta_accionable\n"
            "---\n\nbody"
        )
        spec = parse_blueprint(skill)
        assert spec is not None
        assert spec.deliver == "origin"
        assert spec.category == "event"
        assert spec.material_result_criterion == "alerta_accionable"

    def test_llm_missing_admission_raises_and_names_fields(self):
        with pytest.raises(BlueprintError) as excinfo:
            parse_blueprint(MISSING_ADMISSION_BLUEPRINT)
        err = str(excinfo.value)
        assert "category" in err
        assert "material_result_criterion" in err

    def test_llm_invalid_criterion_names_allowlist(self):
        skill = (
            "---\nname: r\ndescription: d\nmetadata:\n  hermes:\n"
            '    blueprint:\n      schedule: "every 1h"\n'
            "      category: event\n"
            "      material_result_criterion: free-text-not-allowed\n"
            "---\n\nbody"
        )
        with pytest.raises(BlueprintError, match="material_result_criterion") as excinfo:
            parse_blueprint(skill)
        err = str(excinfo.value)
        assert "alerta_accionable" in err
        assert "archivo_entregado" in err

    def test_llm_invalid_category_rejected(self):
        skill = (
            "---\nname: r\ndescription: d\nmetadata:\n  hermes:\n"
            '    blueprint:\n      schedule: "every 1h"\n'
            "      category: vibes\n"
            "      material_result_criterion: alerta_accionable\n"
            "---\n\nbody"
        )
        with pytest.raises(BlueprintError, match="category"):
            parse_blueprint(skill)

    def test_no_agent_blueprint_exempt_from_admission(self):
        skill = (
            "---\nname: watchdog\ndescription: d\nmetadata:\n  hermes:\n"
            '    blueprint:\n      schedule: "every 5m"\n'
            "      no_agent: true\n"
            "---\n\nbody"
        )
        spec = parse_blueprint(skill)
        assert spec is not None
        assert spec.no_agent is True
        assert spec.category is None
        assert spec.material_result_criterion is None


class TestBlueprintSpecForInstalled:
    def test_finds_and_parses_installed_blueprint(self, tmp_path):
        skills_dir = tmp_path / "skills"
        rec_dir = skills_dir / "productivity" / "morning-brief"
        rec_dir.mkdir(parents=True)
        (rec_dir / "SKILL.md").write_text(BLUEPRINT_SKILL, encoding="utf-8")

        with patch("tools.skills_hub.SKILLS_DIR", skills_dir):
            spec = blueprint_spec_for_installed("morning-brief")
        assert spec is not None
        assert spec.schedule == "0 8 * * *"
        assert spec.category == "justified_cadence"
        assert spec.material_result_criterion == "archivo_entregado"

    def test_missing_skill_returns_none(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        with patch("tools.skills_hub.SKILLS_DIR", skills_dir):
            assert blueprint_spec_for_installed("nope") is None

    def test_plain_skill_returns_none(self, tmp_path):
        skills_dir = tmp_path / "skills"
        d = skills_dir / "misc" / "not-a-blueprint"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(PLAIN_SKILL, encoding="utf-8")
        with patch("tools.skills_hub.SKILLS_DIR", skills_dir):
            assert blueprint_spec_for_installed("not-a-blueprint") is None


class TestCreateBlueprintJob:
    def test_bridges_to_create_job(self):
        spec = parse_blueprint(BLUEPRINT_SKILL)
        assert spec is not None
        captured = {}

        def fake_create_job(**kwargs):
            captured.update(kwargs)
            return {"id": "abc123", **kwargs}

        with patch("cron.jobs.create_job", fake_create_job):
            job = create_blueprint_job(spec, origin={"platform": "telegram"})

        assert captured["schedule"] == "0 8 * * *"
        assert captured["skills"] == ["morning-brief"]
        assert captured["deliver"] == "telegram"
        assert captured["prompt"].startswith("Summarize")
        assert captured["category"] == "justified_cadence"
        assert captured["material_result_criterion"] == "archivo_entregado"
        assert job["id"] == "abc123"

    def test_direct_spec_missing_admission_rejected(self):
        spec = BlueprintSpec(skill_name="x", schedule="every 2h", prompt="p")
        with pytest.raises(BlueprintError) as excinfo:
            blueprint_to_job_spec(spec)
        err = str(excinfo.value)
        assert "category" in err
        assert "material_result_criterion" in err

    def test_blueprint_to_job_spec_propagates_admission_unchanged(self):
        spec = BlueprintSpec(
            skill_name="x",
            schedule="every 2h",
            prompt="p",
            category="event",
            material_result_criterion="metrica_registrada",
        )
        js = blueprint_to_job_spec(spec)
        assert js["category"] == "event"
        assert js["material_result_criterion"] == "metrica_registrada"


class TestExportBlueprint:
    def test_round_trips_job_to_skill_md(self):
        job = {
            "name": "My Morning Brief",
            "schedule_display": "0 8 * * *",
            "skills": ["morning-brief"],
            "deliver": "telegram",
            "prompt": "Summarize my unread email.",
            "category": "justified_cadence",
            "material_result_criterion": "archivo_entregado",
        }
        md = export_blueprint(job, "# Morning Brief\n\nDoes the morning digest.")
        # The exported SKILL.md must itself parse back as a blueprint.
        spec = parse_blueprint(md)
        assert spec is not None
        assert spec.schedule == "0 8 * * *"
        assert spec.deliver == "telegram"
        assert spec.category == "justified_cadence"
        assert spec.material_result_criterion == "archivo_entregado"
        # Name is sanitized to a valid skill identifier.
        assert spec.skill_name == "my-morning-brief"

    def test_export_has_blueprint_tag(self):
        job = {
            "name": "x",
            "schedule_display": "every 2h",
            "skills": ["x"],
            "category": "event",
            "material_result_criterion": "alerta_accionable",
        }
        md = export_blueprint(job, "body")
        assert "blueprint" in md
        assert "automation" in md

    def test_export_interval_job_without_display(self):
        # Regression: parse_schedule stores interval periods as "minutes" —
        # exporting a job with only the parsed schedule dict must round-trip
        # the real interval, not fall back to the daily default.
        job = {
            "name": "poller",
            "schedule": {"kind": "interval", "minutes": 30},
            "skills": ["poller"],
            "category": "event",
            "material_result_criterion": "metrica_registrada",
        }
        md = export_blueprint(job, "body")
        spec = parse_blueprint(md)
        assert spec is not None
        assert spec.schedule == "every 30m"

        job["schedule"] = {"kind": "interval", "minutes": 120}
        spec = parse_blueprint(export_blueprint(job, "body"))
        assert spec is not None
        assert spec.schedule == "every 2h"

    def test_export_llm_missing_admission_fails_clearly(self):
        job = {
            "name": "incomplete",
            "schedule_display": "0 9 * * *",
            "prompt": "do stuff",
        }
        with pytest.raises(BlueprintError) as excinfo:
            export_blueprint(job, "body")
        err = str(excinfo.value)
        assert "category" in err
        assert "material_result_criterion" in err

    def test_export_llm_invalid_criterion_fails_clearly(self):
        job = {
            "name": "bad-crit",
            "schedule_display": "0 9 * * *",
            "prompt": "do stuff",
            "category": "event",
            "material_result_criterion": "not-a-standard-criterion",
        }
        with pytest.raises(BlueprintError, match="material_result_criterion"):
            export_blueprint(job, "body")

    def test_export_no_agent_exempt(self):
        job = {
            "name": "watchdog",
            "schedule_display": "every 5m",
            "no_agent": True,
            "script": "watch.sh",
        }
        md = export_blueprint(job, "body")
        assert "no_agent" in md
        spec = parse_blueprint(md)
        assert spec is not None
        assert spec.no_agent is True
