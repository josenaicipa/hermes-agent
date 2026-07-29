from __future__ import annotations

import asyncio
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@pytest.fixture
def cfg():
    return {
        "enabled": True,
        "compaction_tokens": 70_000,
        "blocker_alert_hours": 6,
        "redesign": {
            "primary_url": "http://127.0.0.1:4318/v1/chat/completions",
            "backup_url": "http://127.0.0.1:4319/v1/chat/completions",
            "project": "cron-efficiency",
        },
    }


def test_success_log_is_summarized_and_full_artifact_is_0600(tmp_path):
    from cron.context_efficiency import format_script_context

    raw = "\n".join(f"success line {i}" for i in range(300))
    rendered = format_script_context(
        {"id": "abc123", "name": "job"}, True, raw,
        hermes_home=tmp_path, run_id="run-1",
    )
    artifact = tmp_path / "cron" / "context-efficiency" / "logs" / "abc123" / "run-1.log"
    assert artifact.read_text() == raw
    assert os.stat(artifact).st_mode & 0o777 == 0o600
    assert "success line 299" not in rendered
    assert str(artifact) in rendered
    assert "300 lines" in rendered


def test_failure_log_preserves_150_stderr_lines_and_unfiltered_tail(tmp_path):
    from cron.context_efficiency import format_script_context

    stderr = "\n".join(f"ERR-{i}" for i in range(180))
    stdout = "\n".join(f"OUT-{i}" for i in range(80))
    raw = f"Script exited with code 1\nstderr:\n{stderr}\nstdout:\n{stdout}"
    rendered = format_script_context(
        {"id": "abc123"}, False, raw,
        hermes_home=tmp_path, run_id="run-2",
    )
    assert "ERR-0" in rendered and "ERR-149" in rendered
    assert "ERR-150" not in rendered
    assert "OUT-79" in rendered
    assert "unfiltered tail" in rendered.lower()


def test_apply_cron_compaction_ceiling_is_cron_only_and_fail_closed():
    from cron.context_efficiency import EfficiencyConfig, apply_cron_compaction_ceiling

    class Builtin:
        compression_enabled = True
        threshold_tokens = 120_000
        summary_target_ratio = 0.25
        context_length = 200_000

    class Agent:
        platform = "cron"
        compression_enabled = True
        context_compressor = Builtin()

    applied = apply_cron_compaction_ceiling(Agent(), EfficiencyConfig(enabled=True))
    assert applied == 70_000
    assert Agent.context_compressor.threshold_tokens == 70_000

    Agent.platform = "cli"
    Agent.context_compressor.threshold_tokens = 120_000
    assert apply_cron_compaction_ceiling(Agent(), EfficiencyConfig(enabled=True)) is None
    assert Agent.context_compressor.threshold_tokens == 120_000

    Agent.platform = "cron"
    Agent.context_compressor = Builtin()
    Agent.context_compressor.compression_enabled = False
    with pytest.raises(RuntimeError, match="disabled"):
        apply_cron_compaction_ceiling(Agent(), EfficiencyConfig(enabled=True))

    Agent.context_compressor = object()
    with pytest.raises(RuntimeError, match="compatible threshold_tokens"):
        apply_cron_compaction_ceiling(Agent(), EfficiencyConfig(enabled=True))


def test_auxiliary_override_isolated_and_restored(monkeypatch):
    from agent import auxiliary_client as ac

    monkeypatch.setattr(ac, "_get_auxiliary_task_config", lambda task: {})

    async def resolve(provider, model):
        with ac.auxiliary_task_override("compression", provider=provider, model=model):
            await asyncio.sleep(0)
            return ac._resolve_task_provider_model("compression")[:2]

    async def both():
        return await asyncio.gather(
            resolve("google-gemini-cli", "Gemini 3.5 Flash (Medium)"),
            resolve("custom", "other-model"),
        )

    got = asyncio.run(both())
    assert got == [
        ("google-gemini-cli", "Gemini 3.5 Flash (Medium)"),
        ("custom", "other-model"),
    ]
    assert ac._resolve_task_provider_model("compression")[:2] == ("auto", None)


def test_two_failures_allow_fresh_redesign_once_then_append_once(tmp_path, cfg):
    from cron.context_efficiency import ContextEfficiencyGate, EfficiencyConfig

    calls = []

    def backend(package):
        calls.append(package)
        return "FRESH REDESIGN"

    gate = ContextEfficiencyGate(tmp_path, EfficiencyConfig.from_mapping(cfg), redesign_backend=backend)
    job = {"id": "job1", "name": "Job 1", "prompt": "STATIC GOAL"}
    first = gate.before_run(job, dynamic_input="same-input")
    assert first.allow
    gate.record_outcome(job, first.fingerprint, "failure", error="error one", attempted="try one")
    second = gate.before_run(job, dynamic_input="same-input")
    assert second.allow
    gate.record_outcome(job, second.fingerprint, "failure", error="error two", attempted="try two")

    third = gate.before_run(job, dynamic_input="same-input")
    assert not third.allow and third.redesign_requested
    assert len(calls) == 1
    package = calls[0]
    assert package["objective"] == "STATIC GOAL"
    assert package["dynamic_input"] == "same-input"
    assert [x["error"] for x in package["failures"]] == ["error one", "error two"]
    assert "transcript" not in json.dumps(package).lower()

    next_tick = gate.before_run(job, dynamic_input="same-input")
    assert next_tick.allow and next_tick.dynamic_append == "FRESH REDESIGN"
    later = gate.before_run(job, dynamic_input="same-input")
    assert not later.allow
    assert later.dynamic_append is None
    assert len(calls) == 1


def test_redesign_network_call_does_not_hold_global_ledger_lock(tmp_path, cfg):
    from cron.context_efficiency import ContextEfficiencyGate, EfficiencyConfig

    entered = threading.Event()
    release = threading.Event()
    finished_other = threading.Event()

    def slow_backend(_package):
        entered.set()
        assert release.wait(2)
        return "redesign"

    config = EfficiencyConfig.from_mapping(cfg)
    gate = ContextEfficiencyGate(tmp_path, config, redesign_backend=slow_backend)
    job1 = {"id": "job1", "prompt": "goal"}
    for error in ("e1", "e2"):
        d = gate.before_run(job1, dynamic_input="same")
        gate.record_outcome(job1, d.fingerprint, "failure", error=error)

    worker = threading.Thread(target=lambda: gate.before_run(job1, dynamic_input="same"))
    worker.start()
    assert entered.wait(1)

    def run_other():
        ContextEfficiencyGate(tmp_path, config).before_run(
            {"id": "job2", "prompt": "other"}, dynamic_input="new"
        )
        finished_other.set()

    other = threading.Thread(target=run_other)
    other.start()
    assert finished_other.wait(0.5), "unrelated cron blocked behind Opus network call"
    release.set()
    worker.join(2)
    other.join(2)


def test_material_or_new_input_resets_failure_streak(tmp_path, cfg):
    from cron.context_efficiency import ContextEfficiencyGate, EfficiencyConfig

    gate = ContextEfficiencyGate(tmp_path, EfficiencyConfig.from_mapping(cfg), redesign_backend=lambda _: "x")
    job = {"id": "job1", "prompt": "goal"}
    d1 = gate.before_run(job, dynamic_input="one")
    gate.record_outcome(job, d1.fingerprint, "failure", error="e1")
    d2 = gate.before_run(job, dynamic_input="one")
    gate.record_outcome(job, d2.fingerprint, "success", material=True)
    assert gate.before_run(job, dynamic_input="one").allow
    assert gate.before_run(job, dynamic_input="two").allow


def test_blocker_alert_after_six_hours_once_and_resets(tmp_path, cfg):
    from cron.context_efficiency import ContextEfficiencyGate, EfficiencyConfig

    gate = ContextEfficiencyGate(tmp_path, EfficiencyConfig.from_mapping(cfg))
    job = {"id": "job1", "name": "Job 1", "prompt": "goal"}
    t0 = datetime(2026, 7, 23, tzinfo=timezone.utc)
    gate.record_blocker(job, "quota exhausted", "block-a", now=t0)
    assert gate.blocker_alert(job, "block-a", now=t0 + timedelta(hours=6)) is None
    alert = gate.blocker_alert(job, "block-a", now=t0 + timedelta(hours=6, seconds=1))
    assert "Job 1" in alert and "quota exhausted" in alert
    assert gate.blocker_alert(job, "block-a", now=t0 + timedelta(hours=7)) is None
    gate.record_blocker(job, "different", "block-b", now=t0 + timedelta(hours=7))
    assert gate.blocker_alert(job, "block-b", now=t0 + timedelta(hours=13, seconds=2))


def test_corrupt_ledger_fails_closed(tmp_path, cfg):
    from cron.context_efficiency import ContextEfficiencyGate, EfficiencyConfig, EfficiencyStateError

    state = tmp_path / "cron" / "context-efficiency" / "ledger.json"
    state.parent.mkdir(parents=True)
    state.write_text("{broken")
    with pytest.raises(EfficiencyStateError):
        ContextEfficiencyGate(tmp_path, EfficiencyConfig.from_mapping(cfg)).before_run(
            {"id": "j", "prompt": "g"}, dynamic_input="x"
        )


def test_opus_backend_primary_then_backup_once(cfg):
    from cron.context_efficiency import EfficiencyConfig, OpusRedesignBackend

    seen = []

    def post(url, body, timeout):
        seen.append((url, body))
        if url.endswith("4318/v1/chat/completions"):
            raise OSError("primary down")
        return {"choices": [{"message": {"content": "redesign"}}]}

    backend = OpusRedesignBackend(EfficiencyConfig.from_mapping(cfg), http_post=post)
    package = {"objective": "g", "dynamic_input": "x", "failures": []}
    assert backend(package) == "redesign"
    assert [x[0] for x in seen] == [
        "http://127.0.0.1:4318/v1/chat/completions",
        "http://127.0.0.1:4319/v1/chat/completions",
    ]
    body = seen[0][1]
    assert body["profile"] == "review_bundle"
    assert body["model"] == "claude-opus-4-8"
    assert body["effort"] == "max"
    assert len(body["candidate_sha"]) == 64


def test_weekly_records_do_not_invent_missing_usage(tmp_path, cfg):
    from cron.context_efficiency import ContextEfficiencyGate, EfficiencyConfig

    gate = ContextEfficiencyGate(tmp_path, EfficiencyConfig.from_mapping(cfg))
    job = {"id": "j", "prompt": "g"}
    d = gate.before_run(job, dynamic_input="x")
    gate.record_outcome(job, d.fingerprint, "valid_noop", material=False, usage={"input_tokens": 3})
    row = gate.weekly_records("j")[-1]
    assert row["usage"] == {"input_tokens": 3}
    assert "output_tokens" not in row["usage"]
