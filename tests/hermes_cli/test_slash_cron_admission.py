"""Regression tests for LLM cron admission fields on the interactive /cron UI."""

import json

import cli

from hermes_cli.cli_commands_mixin import CLICommandsMixin


def test_slash_cron_add_forwards_required_llm_admission_fields(monkeypatch, capsys):
    captured = {}

    def fake_cronjob(**kwargs):
        captured.update(kwargs)
        return json.dumps(
            {
                "success": True,
                "job_id": "job-1",
                "schedule": "every 2h",
                "next_run_at": "later",
                "skills": [],
            }
        )

    monkeypatch.setattr("tools.cronjob_tools.cronjob", fake_cronjob)

    CLICommandsMixin()._handle_cron_command(
        '/cron add "every 2h" "Check server status" '
        '--category event '
        '--material-result-criterion "verified status report delivered"'
    )

    assert captured["category"] == "event"
    assert captured["material_result_criterion"] == "verified status report delivered"
    assert "Created job: job-1" in capsys.readouterr().out


def test_slash_cron_edit_forwards_required_llm_admission_fields(monkeypatch):
    captured = {}

    def fake_cronjob(**kwargs):
        captured.update(kwargs)
        return json.dumps(
            {
                "success": True,
                "job": {"job_id": "job-1", "schedule": "every 4h", "skills": []},
            }
        )

    monkeypatch.setattr(cli, "get_job", lambda _job_id: {"job_id": "job-1", "skills": []})
    monkeypatch.setattr("tools.cronjob_tools.cronjob", fake_cronjob)

    CLICommandsMixin()._handle_cron_command(
        '/cron edit job-1 '
        '--category justified_cadence '
        '--material-result-criterion "verified weekly report exists"'
    )

    assert captured["category"] == "justified_cadence"
    assert captured["material_result_criterion"] == "verified weekly report exists"
