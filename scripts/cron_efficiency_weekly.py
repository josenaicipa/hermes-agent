#!/usr/bin/env python3
"""Weekly cron quality comparison and per-job fail-safe rollback."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def compare_reports(baseline: Mapping[str, Any], weekly: Mapping[str, Any]) -> list[dict[str, Any]]:
    weekly_by_id = {
        str(row.get("job_id")): row
        for row in weekly.get("jobs", [])
        if isinstance(row, Mapping) and row.get("job_id")
    }
    rows: list[dict[str, Any]] = []
    for base in baseline.get("jobs", []):
        if not isinstance(base, Mapping) or not base.get("job_id"):
            continue
        job_id = str(base["job_id"])
        current = weekly_by_id.get(job_id, {})
        base_rate = base.get("contract_success_rate")
        week_rate = current.get("contract_success_rate")
        completed = int(current.get("completed_runs") or 0)
        evaluable = (
            isinstance(base_rate, (int, float))
            and isinstance(week_rate, (int, float))
            and completed > 0
        )
        drop = round((float(base_rate) - float(week_rate)) * 100, 4) if evaluable else None
        rows.append({
            "job_id": job_id,
            "name": str(base.get("name") or current.get("name") or job_id),
            "evaluable": evaluable,
            "baseline_success_rate": base_rate if isinstance(base_rate, (int, float)) else None,
            "weekly_success_rate": week_rate if isinstance(week_rate, (int, float)) else None,
            "drop_points": drop,
            "rollback": bool(evaluable and drop is not None and drop > 5.0),
            "baseline_tokens_per_material": base.get("tokens_per_material_result"),
            "weekly_tokens_per_material": current.get("tokens_per_material_result"),
            "completed_runs": completed,
        })
    return rows


def _pct(value: Any) -> str:
    return "N/D" if not isinstance(value, (int, float)) else f"{float(value) * 100:.1f}%"


def _token_metric(value: Any) -> str:
    return "N/D" if not isinstance(value, (int, float)) else f"{float(value):,.0f}"


def render_report(rows: list[Mapping[str, Any]], *, rolled_back: list[str]) -> str:
    lines = [
        "# Comparación semanal de eficiencia de crons",
        "",
        "Regla: rollback por cron únicamente cuando la tasa de éxito cae más de 5 puntos porcentuales.",
        "Métrica final: tokens por resultado material sin degradación de éxito.",
        "",
    ]
    rolled = set(rolled_back)
    for row in rows:
        if not row.get("evaluable"):
            status = "N/D — sin muestra semanal comparable"
        elif row.get("job_id") in rolled:
            status = "ROLLBACK APLICADO"
        elif row.get("rollback"):
            status = "ROLLBACK REQUERIDO"
        else:
            status = "OK"
        drop = row.get("drop_points")
        drop_text = "N/D" if not isinstance(drop, (int, float)) else f"{drop:.1f} pp"
        lines.append(
            f"- **{row.get('name')}** (`{row.get('job_id')}`): {status}; "
            f"éxito baseline {_pct(row.get('baseline_success_rate'))} → "
            f"semana {_pct(row.get('weekly_success_rate'))} ({drop_text}); "
            f"tokens/material baseline {_token_metric(row.get('baseline_tokens_per_material'))} → "
            f"semana {_token_metric(row.get('weekly_tokens_per_material'))}; "
            f"runs={row.get('completed_runs', 0)}."
        )
    return "\n".join(lines) + "\n"


def _run_auditor(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    json_out = output_dir / "weekly.json"
    md_out = output_dir / "weekly.md"
    command = [
        sys.executable,
        str(args.auditor),
        "--jobs", str(args.jobs),
        "--state-db", str(args.state_db),
        "--rules", str(args.rules),
        "--as-of", as_of,
        "--days", "7",
        "--prompt-threshold", "2000",
        "--json-out", str(json_out),
        "--md-out", str(md_out),
    ]
    subprocess.run(command, check=True, timeout=300)
    return json.loads(json_out.read_text(encoding="utf-8"))


def _rollback(job_ids: list[str]) -> list[str]:
    from cron.jobs import update_job

    changed: list[str] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for job_id in job_ids:
        updated = update_job(job_id, {
            "context_efficiency": {
                "enabled": False,
                "rollback_reason": "weekly success rate dropped more than 5 percentage points",
                "rolled_back_at": timestamp,
            }
        })
        if updated is None:
            raise RuntimeError(f"rollback target not found: {job_id}")
        changed.append(job_id)
    return changed


def parse_args() -> argparse.Namespace:
    home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=home / "cron/context-efficiency/baseline.json")
    parser.add_argument("--auditor", type=Path, default=home / "scripts/cron_efficiency_baseline.py")
    parser.add_argument("--rules", type=Path, default=home / "cron/context-efficiency/baseline-rules.json")
    parser.add_argument("--jobs", type=Path, default=home / "cron/jobs.json")
    parser.add_argument("--state-db", type=Path, default=home / "state.db")
    parser.add_argument("--output-dir", type=Path, default=home / "cron/context-efficiency/weekly")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        weekly = _run_auditor(args, args.output_dir)
        rows = compare_reports(baseline, weekly)
        targets = [str(row["job_id"]) for row in rows if row["rollback"]]
        rolled_back = [] if args.dry_run else _rollback(targets)
        report = render_report(rows, rolled_back=rolled_back)
        report_path = args.output_dir / "comparison.md"
        report_path.write_text(report, encoding="utf-8")
        os.chmod(report_path, 0o600)
        print(report, end="")
        return 0
    except Exception as exc:
        print(f"ALERTA: comparación semanal de crons falló cerrada: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
