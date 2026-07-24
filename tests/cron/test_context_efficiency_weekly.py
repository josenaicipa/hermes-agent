from scripts.cron_efficiency_weekly import compare_reports, render_report


def _report(rows):
    return {"jobs": rows}


def test_drop_greater_than_five_points_requests_per_job_rollback():
    baseline = _report([
        {"job_id": "a", "name": "A", "contract_success_rate": 0.90, "tokens_per_material_result": 100},
        {"job_id": "b", "name": "B", "contract_success_rate": 0.80, "tokens_per_material_result": None},
    ])
    weekly = _report([
        {"job_id": "a", "name": "A", "completed_runs": 10, "contract_success_rate": 0.84, "tokens_per_material_result": 80},
        {"job_id": "b", "name": "B", "completed_runs": 4, "contract_success_rate": 0.75, "tokens_per_material_result": None},
    ])
    rows = compare_reports(baseline, weekly)
    assert rows[0]["rollback"] is True
    assert rows[0]["drop_points"] == 6.0
    assert rows[1]["rollback"] is False
    assert rows[1]["drop_points"] == 5.0


def test_missing_or_zero_runs_are_not_evaluable():
    baseline = _report([
        {"job_id": "a", "name": "A", "contract_success_rate": None, "tokens_per_material_result": None},
        {"job_id": "b", "name": "B", "contract_success_rate": 0.90, "tokens_per_material_result": 1},
    ])
    weekly = _report([
        {"job_id": "a", "completed_runs": 10, "contract_success_rate": 0.10},
        {"job_id": "b", "completed_runs": 0, "contract_success_rate": 0.0},
    ])
    rows = compare_reports(baseline, weekly)
    assert all(row["evaluable"] is False and row["rollback"] is False for row in rows)
    text = render_report(rows, rolled_back=[])
    assert text.count("N/D") >= 2


def test_render_report_marks_rollback_and_tokens_without_inventing():
    rows = [{
        "job_id": "a", "name": "A", "evaluable": True,
        "baseline_success_rate": 0.91, "weekly_success_rate": 0.80,
        "drop_points": 11.0, "rollback": True,
        "baseline_tokens_per_material": None, "weekly_tokens_per_material": 42.0,
        "completed_runs": 5,
    }]
    text = render_report(rows, rolled_back=["a"])
    assert "ROLLBACK APLICADO" in text
    assert "baseline N/D" in text
    assert "42" in text
