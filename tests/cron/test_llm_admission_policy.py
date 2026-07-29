"""LLM-cron admission policy — category + material-result criterion.

Every new enabled LLM cron must declare:
  1) category ∈ {event, justified_cadence, necessary_as_is}
  2) non-empty material_result_criterion

Deterministic no_agent jobs are exempt. Existing enabled jobs without the
fields remain readable and enabled (grandfathered). Resume/reactivate of an
LLM job missing either field fails closed.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME and exercise the real admission gate.

    This module is excluded from the suite-wide test-only create_job
    admission defaults (see ``_cron_llm_admission_test_defaults`` in
    ``tests/conftest.py``), so create/resume hit production fail-closed
    behavior with no env-controlled bypass.

    ``cron.jobs`` resolves the store from the current HERMES_HOME /
    ``use_cron_store`` context dynamically — do not reload the module here
    (reload would desync the suite-wide create_job wrapper in other files).

    Hold an explicit store override for the entire test so production cron
    calls and raw fixture writes share the same isolated jobs file.  The
    ContextVar is reset automatically during fixture teardown.
    """
    import cron.jobs

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with cron.jobs.use_cron_store(home):
        yield home


# ---------------------------------------------------------------------------
# create_job data layer
# ---------------------------------------------------------------------------


def test_create_llm_job_without_admission_rejected(hermes_env):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="material_result_criterion|category"):
        create_job(prompt="sync CRM to calendar", schedule="every 1h", deliver="local")


def test_create_llm_job_missing_only_category_rejected(hermes_env):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="category"):
        create_job(
            prompt="sync CRM to calendar",
            schedule="every 1h",
            deliver="local",
            material_result_criterion="CRM event X exists on Calendar",
        )


def test_create_llm_job_missing_only_criterion_rejected(hermes_env):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="material_result_criterion"):
        create_job(
            prompt="sync CRM to calendar",
            schedule="every 1h",
            deliver="local",
            category="event",
        )


def test_create_llm_job_invalid_category_rejected(hermes_env):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="category"):
        create_job(
            prompt="sync CRM to calendar",
            schedule="every 1h",
            deliver="local",
            category="vibes",
            material_result_criterion="something verifiable happened",
        )


def test_create_llm_job_with_admission_succeeds_and_persists(hermes_env):
    from cron.jobs import create_job, get_job

    job = create_job(
        prompt="sync CRM to calendar",
        schedule="every 1h",
        deliver="local",
        category="justified_cadence",
        material_result_criterion="At least one CRM change mirrored to Calendar with matching IDs",
    )
    assert job["enabled"] is True
    assert job["category"] == "justified_cadence"
    assert "mirrored to Calendar" in job["material_result_criterion"]

    reloaded = get_job(job["id"])
    assert reloaded is not None
    assert reloaded["enabled"] is True
    assert reloaded["category"] == "justified_cadence"
    assert reloaded["material_result_criterion"] == job["material_result_criterion"]


@pytest.mark.parametrize("category", ["event", "justified_cadence", "necessary_as_is"])
def test_create_llm_job_accepts_all_allowed_categories(hermes_env, category):
    from cron.jobs import create_job

    job = create_job(
        prompt="do the thing",
        schedule="every 6h",
        deliver="local",
        category=category,
        material_result_criterion="verifiable outcome X",
    )
    assert job["category"] == category
    assert job["enabled"] is True


def test_create_no_agent_job_exempt_from_admission(hermes_env):
    from cron.jobs import create_job

    script = hermes_env / "scripts" / "watchdog.sh"
    script.write_text("#!/bin/bash\necho ok\n")

    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="watchdog.sh",
        no_agent=True,
        deliver="local",
    )
    assert job["no_agent"] is True
    assert job["enabled"] is True
    # Admission fields are optional/absent for script-only jobs.
    assert not job.get("category")
    assert not job.get("material_result_criterion")


# ---------------------------------------------------------------------------
# Optional material-criterion allowlist (blueprint/suggestion surface)
# ---------------------------------------------------------------------------


def test_check_admission_allowlist_accepts_listed_criterion():
    from cron.jobs import (
        LLM_BLUEPRINT_MATERIAL_CRITERIA,
        check_llm_admission_for_enable,
    )

    check_llm_admission_for_enable(
        category="event",
        material_result_criterion="alerta_accionable",
        allowed_material_criteria=LLM_BLUEPRINT_MATERIAL_CRITERIA,
    )


def test_check_admission_allowlist_rejects_unlisted_criterion():
    from cron.jobs import (
        LLM_BLUEPRINT_MATERIAL_CRITERIA,
        LlmCronAdmissionError,
        check_llm_admission_for_enable,
    )

    with pytest.raises(LlmCronAdmissionError, match="material_result_criterion") as excinfo:
        check_llm_admission_for_enable(
            category="event",
            material_result_criterion="free-text outcome not in allowlist",
            allowed_material_criteria=LLM_BLUEPRINT_MATERIAL_CRITERIA,
        )
    err = str(excinfo.value)
    assert "alerta_accionable" in err
    assert "archivo_entregado" in err
    assert "integracion_produccion" in err
    assert "metrica_registrada" in err


def test_check_admission_allowlist_lists_both_missing_fields():
    from cron.jobs import (
        LLM_BLUEPRINT_MATERIAL_CRITERIA,
        LlmCronAdmissionError,
        check_llm_admission_for_enable,
    )

    with pytest.raises(LlmCronAdmissionError) as excinfo:
        check_llm_admission_for_enable(
            allowed_material_criteria=LLM_BLUEPRINT_MATERIAL_CRITERIA,
        )
    err = str(excinfo.value)
    assert "category" in err
    assert "material_result_criterion" in err


def test_check_admission_without_allowlist_accepts_any_nonempty_criterion():
    from cron.jobs import check_llm_admission_for_enable

    # Existing non-blueprint callers: free-text criteria remain valid.
    check_llm_admission_for_enable(
        category="justified_cadence",
        material_result_criterion="CRM deal id appears on Calendar with matching ids",
    )


def test_check_admission_allowlist_no_agent_exempt():
    from cron.jobs import (
        LLM_BLUEPRINT_MATERIAL_CRITERIA,
        check_llm_admission_for_enable,
    )

    check_llm_admission_for_enable(
        no_agent=True,
        allowed_material_criteria=LLM_BLUEPRINT_MATERIAL_CRITERIA,
    )


# ---------------------------------------------------------------------------
# Grandfathering — existing enabled jobs without fields stay enabled
# ---------------------------------------------------------------------------


def _write_raw_jobs(jobs_file, jobs):
    """Seed pre-policy / on-disk state without going through save_jobs().

    Grandfathered incomplete enabled LLM records predate the save-time gate;
    tests simulate that by writing jobs.json directly.
    """
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    jobs_file.write_text(
        json.dumps({"jobs": jobs, "updated_at": "2026-01-01T00:00:00+00:00"}, indent=2),
        encoding="utf-8",
    )


def _legacy_enabled_incomplete_llm(**overrides):
    job = {
        "id": "legacy44crm",
        "name": "CRM↔Calendar #44",
        "prompt": "Sync CRM contacts to Google Calendar",
        "skills": [],
        "skill": None,
        "model": None,
        "provider": None,
        "base_url": None,
        "script": None,
        "no_agent": False,
        "schedule": {"kind": "interval", "minutes": 60, "display": "every 1h"},
        "schedule_display": "every 1h",
        "repeat": {"times": None, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "next_run_at": "2030-01-01T00:00:00+00:00",
        "last_run_at": None,
        "last_status": None,
        "last_error": None,
        "deliver": "local",
        "origin": None,
        # Intentionally omit category + material_result_criterion
    }
    job.update(overrides)
    return job


def test_legacy_enabled_llm_job_loads_and_stays_enabled(hermes_env):
    """Jobs written before the policy remain readable and enabled."""
    from cron.jobs import ensure_dirs, get_job, list_jobs, load_jobs

    ensure_dirs()
    jobs_file = hermes_env / "cron" / "jobs.json"
    _write_raw_jobs(jobs_file, [_legacy_enabled_incomplete_llm()])

    loaded = load_jobs()
    assert len(loaded) == 1
    assert loaded[0]["enabled"] is True
    assert "category" not in loaded[0] or not loaded[0].get("category")

    fetched = get_job("legacy44crm")
    assert fetched is not None
    assert fetched["enabled"] is True
    assert fetched["name"] == "CRM↔Calendar #44"

    active = list_jobs(include_disabled=False)
    assert any(j["id"] == "legacy44crm" for j in active)
    assert jobs_file.exists()


# ---------------------------------------------------------------------------
# resume / reactivate fails closed without admission; succeeds after set
# ---------------------------------------------------------------------------


def test_resume_llm_job_missing_admission_fails_closed(hermes_env):
    from cron.jobs import pause_job, resume_job, save_jobs, ensure_dirs

    ensure_dirs()
    legacy = {
        "id": "paused44",
        "name": "CRM↔Calendar #44 paused",
        "prompt": "Sync CRM to Calendar",
        "skills": [],
        "skill": None,
        "no_agent": False,
        "schedule": {"kind": "interval", "minutes": 60, "display": "every 1h"},
        "schedule_display": "every 1h",
        "repeat": {"times": None, "completed": 0},
        "enabled": False,
        "state": "paused",
        "paused_at": "2026-06-01T00:00:00+00:00",
        "paused_reason": "operator paused",
        "created_at": "2026-01-01T00:00:00+00:00",
        "next_run_at": None,
        "deliver": "local",
        "origin": None,
    }
    save_jobs([legacy])

    with pytest.raises(ValueError, match="material_result_criterion|category|admission"):
        resume_job("paused44")


def test_resume_llm_job_succeeds_after_admission_set(hermes_env):
    from cron.jobs import get_job, resume_job, save_jobs, ensure_dirs, update_job

    ensure_dirs()
    legacy = {
        "id": "paused44b",
        "name": "CRM↔Calendar #44",
        "prompt": "Sync CRM to Calendar",
        "skills": [],
        "skill": None,
        "no_agent": False,
        "schedule": {"kind": "interval", "minutes": 60, "display": "every 1h"},
        "schedule_display": "every 1h",
        "repeat": {"times": None, "completed": 0},
        "enabled": False,
        "state": "paused",
        "paused_at": "2026-06-01T00:00:00+00:00",
        "paused_reason": "operator paused",
        "created_at": "2026-01-01T00:00:00+00:00",
        "next_run_at": None,
        "deliver": "local",
        "origin": None,
    }
    save_jobs([legacy])

    update_job(
        "paused44b",
        {
            "category": "necessary_as_is",
            "material_result_criterion": "CRM contact changes appear on Calendar within the hour",
        },
    )
    resumed = resume_job("paused44b")
    assert resumed is not None
    assert resumed["enabled"] is True
    assert resumed["state"] == "scheduled"
    assert resumed["category"] == "necessary_as_is"
    assert "Calendar" in resumed["material_result_criterion"]

    assert get_job("paused44b")["enabled"] is True


def test_resume_no_agent_job_exempt(hermes_env):
    from cron.jobs import create_job, pause_job, resume_job

    script = hermes_env / "scripts" / "w.sh"
    script.write_text("echo hi\n")
    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="w.sh",
        no_agent=True,
        deliver="local",
    )
    pause_job(job["id"], reason="temp")
    resumed = resume_job(job["id"])
    assert resumed["enabled"] is True
    assert resumed["no_agent"] is True


# ---------------------------------------------------------------------------
# cronjob tool surface
# ---------------------------------------------------------------------------


def test_cronjob_tool_create_llm_without_admission_fails(hermes_env):
    from tools.cronjob_tools import cronjob

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 1h",
            prompt="sync CRM to calendar",
            deliver="local",
        )
    )
    assert result.get("success") is False
    err = result.get("error", "")
    assert "category" in err or "material_result_criterion" in err


def test_cronjob_tool_create_llm_with_admission_succeeds(hermes_env):
    from tools.cronjob_tools import cronjob

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 1h",
            prompt="sync CRM to calendar",
            deliver="local",
            category="event",
            material_result_criterion="New CRM deal produces a Calendar event with matching deal id",
        )
    )
    assert result.get("success") is True
    job = result["job"]
    assert job.get("category") == "event"
    assert "deal id" in (job.get("material_result_criterion") or "")


def test_cronjob_tool_create_no_agent_still_exempt(hermes_env):
    from tools.cronjob_tools import cronjob

    script = hermes_env / "scripts" / "alert.sh"
    script.write_text("#!/bin/bash\necho alert\n")

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="alert.sh",
            no_agent=True,
            deliver="local",
        )
    )
    assert result.get("success") is True
    assert result["job"]["no_agent"] is True


def test_cronjob_tool_resume_missing_admission_fails(hermes_env):
    from cron.jobs import ensure_dirs, save_jobs
    from tools.cronjob_tools import cronjob

    ensure_dirs()
    save_jobs(
        [
            {
                "id": "toolpause44",
                "name": "CRM↔Calendar #44",
                "prompt": "Sync",
                "skills": [],
                "skill": None,
                "no_agent": False,
                "schedule": {"kind": "interval", "minutes": 60, "display": "every 1h"},
                "schedule_display": "every 1h",
                "repeat": {"times": None, "completed": 0},
                "enabled": False,
                "state": "paused",
                "paused_at": "2026-06-01T00:00:00+00:00",
                "paused_reason": "paused",
                "created_at": "2026-01-01T00:00:00+00:00",
                "next_run_at": None,
                "deliver": "local",
                "origin": None,
            }
        ]
    )

    result = json.loads(cronjob(action="resume", job_id="toolpause44"))
    assert result.get("success") is False
    err = result.get("error", "")
    assert "category" in err or "material_result_criterion" in err or "admission" in err


def test_cronjob_schema_documents_admission_fields():
    from tools.cronjob_tools import CRONJOB_SCHEMA

    props = CRONJOB_SCHEMA["parameters"]["properties"]
    assert "category" in props
    assert "material_result_criterion" in props
    cat = props["category"]
    # Allowed values must be enumerable for the model.
    enum = cat.get("enum") or []
    assert set(enum) == {"event", "justified_cadence", "necessary_as_is"}
    crit_desc = props["material_result_criterion"]["description"].lower()
    assert "verif" in crit_desc or "material" in crit_desc


# ---------------------------------------------------------------------------
# Lifecycle invariant — no silent strip / no save-path bypass
# ---------------------------------------------------------------------------


def _compliant_enabled_llm(**overrides):
    job = {
        "id": "compliant01",
        "name": "Compliant LLM cron",
        "prompt": "do verifiable work",
        "skills": [],
        "skill": None,
        "no_agent": False,
        "schedule": {"kind": "interval", "minutes": 60, "display": "every 1h"},
        "schedule_display": "every 1h",
        "repeat": {"times": None, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "next_run_at": "2030-01-01T00:00:00+00:00",
        "last_run_at": None,
        "last_status": None,
        "last_error": None,
        "deliver": "local",
        "origin": None,
        "category": "event",
        "material_result_criterion": "Outcome X is present with matching ids",
    }
    job.update(overrides)
    return job


def test_update_job_clearing_both_admission_fields_rejected_and_unchanged(hermes_env):
    """Enabled compliant LLM cron cannot have both declarations cleared."""
    from cron.jobs import create_job, get_job, update_job

    job = create_job(
        prompt="sync CRM to calendar",
        schedule="every 1h",
        deliver="local",
        category="event",
        material_result_criterion="CRM deal id appears on Calendar",
    )
    job_id = job["id"]
    before = get_job(job_id)

    with pytest.raises(ValueError, match="category|material_result_criterion|admission"):
        update_job(
            job_id,
            {"category": None, "material_result_criterion": None},
        )

    after = get_job(job_id)
    assert after is not None
    assert after["category"] == before["category"] == "event"
    assert after["material_result_criterion"] == before["material_result_criterion"]
    assert after["enabled"] is True


@pytest.mark.parametrize(
    "clear_field,keep_field,keep_value",
    [
        ("category", "material_result_criterion", "CRM deal id appears on Calendar"),
        ("material_result_criterion", "category", "event"),
    ],
)
def test_update_job_clearing_one_admission_field_rejected_and_unchanged(
    hermes_env, clear_field, keep_field, keep_value
):
    """Clearing either mandatory field alone on an enabled LLM cron fails closed."""
    from cron.jobs import create_job, get_job, update_job

    job = create_job(
        prompt="sync CRM to calendar",
        schedule="every 1h",
        deliver="local",
        category="event",
        material_result_criterion="CRM deal id appears on Calendar",
    )
    job_id = job["id"]
    before = get_job(job_id)

    with pytest.raises(ValueError, match="category|material_result_criterion|admission"):
        update_job(job_id, {clear_field: None, keep_field: keep_value})

    after = get_job(job_id)
    assert after is not None
    assert after["category"] == before["category"]
    assert after["material_result_criterion"] == before["material_result_criterion"]
    assert after["enabled"] is True


def test_update_job_clearing_admission_via_empty_string_rejected(hermes_env):
    from cron.jobs import create_job, get_job, update_job

    job = create_job(
        prompt="sync CRM to calendar",
        schedule="every 1h",
        deliver="local",
        category="justified_cadence",
        material_result_criterion="At least one change mirrored",
    )
    before = get_job(job["id"])

    with pytest.raises(ValueError, match="category|material_result_criterion|admission"):
        update_job(job["id"], {"category": "", "material_result_criterion": ""})

    after = get_job(job["id"])
    assert after["category"] == before["category"]
    assert after["material_result_criterion"] == before["material_result_criterion"]


def test_save_jobs_rejects_new_enabled_incomplete_llm_record(hermes_env):
    """Public save/import path cannot introduce a new incomplete enabled LLM cron."""
    from cron.jobs import ensure_dirs, load_jobs, save_jobs

    ensure_dirs()
    jobs_file = hermes_env / "cron" / "jobs.json"
    incomplete = _legacy_enabled_incomplete_llm(id="brandnew99", name="Illicit import")

    with pytest.raises(ValueError, match="category|material_result_criterion|admission"):
        save_jobs([incomplete])

    # Nothing persisted (or empty store preserved).
    if jobs_file.exists():
        assert load_jobs() == []


def test_save_jobs_rejects_stripping_fields_from_enabled_compliant_record(hermes_env):
    """Direct save cannot strip declarations from an enabled compliant LLM cron."""
    from cron.jobs import ensure_dirs, load_jobs, save_jobs

    ensure_dirs()
    compliant = _compliant_enabled_llm()
    save_jobs([compliant])

    stripped = dict(compliant)
    stripped.pop("category", None)
    stripped.pop("material_result_criterion", None)

    with pytest.raises(ValueError, match="category|material_result_criterion|admission"):
        save_jobs([stripped])

    reloaded = load_jobs()
    assert len(reloaded) == 1
    assert reloaded[0]["category"] == "event"
    assert reloaded[0]["material_result_criterion"] == compliant["material_result_criterion"]
    assert reloaded[0]["enabled"] is True


def test_save_jobs_allows_bookkeeping_for_legacy_enabled_incomplete(hermes_env):
    """Grandfathered LLM jobs remain writable by scheduler bookkeeping."""
    from cron.jobs import ensure_dirs, load_jobs, save_jobs

    ensure_dirs()
    legacy = _legacy_enabled_incomplete_llm()
    _write_raw_jobs(hermes_env / "cron" / "jobs.json", [legacy])

    jobs = load_jobs()
    assert len(jobs) == 1
    jobs[0]["last_run_at"] = "2030-01-02T00:00:00+00:00"
    jobs[0]["last_status"] = "ok"
    save_jobs(jobs)

    reloaded = load_jobs()
    assert reloaded[0]["id"] == "legacy44crm"
    assert reloaded[0]["enabled"] is True
    assert reloaded[0]["last_status"] == "ok"
    assert not reloaded[0].get("category")
    assert not reloaded[0].get("material_result_criterion")


def test_update_job_allows_unrelated_edit_of_legacy_enabled_incomplete(hermes_env):
    """Legacy jobs can be edited and classified without pausing first."""
    from cron.jobs import ensure_dirs, get_job, update_job

    ensure_dirs()
    legacy = _legacy_enabled_incomplete_llm()
    _write_raw_jobs(hermes_env / "cron" / "jobs.json", [legacy])

    updated = update_job("legacy44crm", {"name": "renamed but still incomplete"})
    assert updated is not None
    assert updated["name"] == "renamed but still incomplete"

    persisted = get_job("legacy44crm")
    assert persisted is not None
    assert persisted["name"] == "renamed but still incomplete"
    assert persisted["enabled"] is True
    assert not persisted.get("category")
    assert not persisted.get("material_result_criterion")


def test_multiple_legacy_jobs_do_not_lock_store_remediation(hermes_env):
    """One legacy record cannot block classification or bookkeeping of another."""
    from cron.jobs import ensure_dirs, get_job, load_jobs, save_jobs, update_job

    ensure_dirs()
    first = _legacy_enabled_incomplete_llm(id="legacy-a", name="Legacy A")
    second = _legacy_enabled_incomplete_llm(id="legacy-b", name="Legacy B")
    _write_raw_jobs(hermes_env / "cron" / "jobs.json", [first, second])

    jobs = load_jobs()
    jobs[0]["last_status"] = "ok"
    save_jobs(jobs)
    classified = update_job(
        "legacy-a",
        {
            "category": "event",
            "material_result_criterion": "Execution A records a verified result",
        },
    )

    assert classified is not None
    assert classified["category"] == "event"
    remaining = get_job("legacy-b")
    assert remaining is not None
    assert remaining["enabled"] is True
    assert not remaining.get("category")


def test_save_jobs_rejects_enabling_legacy_incomplete_via_direct_write(hermes_env):
    """A disabled legacy incomplete record still cannot be enabled without both fields."""
    from cron.jobs import ensure_dirs, load_jobs, save_jobs

    ensure_dirs()
    paused = _legacy_enabled_incomplete_llm(
        id="pausedlegacy",
        enabled=False,
        state="paused",
        next_run_at=None,
        paused_at="2026-06-01T00:00:00+00:00",
        paused_reason="operator",
    )
    save_jobs([paused])  # disabled incomplete is allowed

    enabled = dict(paused)
    enabled["enabled"] = True
    enabled["state"] = "scheduled"
    enabled["next_run_at"] = "2030-01-01T00:00:00+00:00"
    enabled["paused_at"] = None
    enabled["paused_reason"] = None

    with pytest.raises(ValueError, match="category|material_result_criterion|admission"):
        save_jobs([enabled])

    reloaded = load_jobs()
    assert reloaded[0]["enabled"] is False
    assert reloaded[0]["state"] == "paused"


def test_save_jobs_no_agent_incomplete_still_allowed(hermes_env):
    """no_agent jobs remain exempt on the direct save path."""
    from cron.jobs import load_jobs, save_jobs

    script = hermes_env / "scripts" / "na.sh"
    script.write_text("echo ok\n")

    job = {
        "id": "noagent01",
        "name": "script only",
        "prompt": None,
        "script": "na.sh",
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 5, "display": "every 5m"},
        "schedule_display": "every 5m",
        "repeat": {"times": None, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "created_at": "2026-01-01T00:00:00+00:00",
        "next_run_at": "2030-01-01T00:00:00+00:00",
        "deliver": "local",
    }
    save_jobs([job])
    assert load_jobs()[0]["no_agent"] is True
    assert load_jobs()[0]["enabled"] is True
