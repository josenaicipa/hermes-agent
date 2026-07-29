"""Task A: minimal agentic-efficiency SQLite ledger for cron runs.

Strict TDD coverage for the three-column ``agentic_efficiency`` table and the
scheduler integration that appends exactly once per agentic model-path attempt.
"""

from __future__ import annotations

import concurrent.futures
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _point_ledger(monkeypatch, tmp_path: Path):
    """Point the agentic ledger at an isolated DB under tmp_path."""
    import cron.agentic_efficiency as ae

    db = tmp_path / "cron" / "agentic_efficiency.db"
    monkeypatch.setattr(ae, "AGENTIC_EFFICIENCY_FILE", db)
    return ae


def _pragma_columns(db_path: Path) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("PRAGMA table_info(agentic_efficiency)").fetchall()
    # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
    return [r[1] for r in rows]


def _all_rows(db_path: Path) -> list[tuple]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT tipo, resultado_material, tokens FROM agentic_efficiency"
        ).fetchall()


def _run_job_with_agent(
    job: dict,
    *,
    tmp_path: Path,
    agent: MagicMock,
    result: dict | None = None,
    raise_exc: BaseException | None = None,
):
    """Drive ``run_job`` with a fake AIAgent (no network)."""
    from cron.scheduler import run_job

    fake_db = MagicMock()
    if raise_exc is not None:
        agent.run_conversation.side_effect = raise_exc
    else:
        agent.run_conversation.return_value = result or {
            "final_response": "ok",
            "completed": True,
            "failed": False,
            "messages": [],
        }

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "test-key",
                 "base_url": "https://example.invalid/v1",
                 "provider": "openrouter",
                 "api_mode": "chat_completions",
             },
         ), \
         patch("run_agent.AIAgent", return_value=agent):
        return run_job(job)


# ---------------------------------------------------------------------------
# Schema + append API
# ---------------------------------------------------------------------------


class TestAgenticEfficiencySchema:
    def test_table_has_exactly_three_named_columns(self, monkeypatch, tmp_path):
        ae = _point_ledger(monkeypatch, tmp_path)
        ae.append_agentic_efficiency("nueva", "no", 0)
        assert _pragma_columns(ae.AGENTIC_EFFICIENCY_FILE) == [
            "tipo",
            "resultado_material",
            "tokens",
        ]

    def test_append_persists_row(self, monkeypatch, tmp_path):
        ae = _point_ledger(monkeypatch, tmp_path)
        ae.append_agentic_efficiency("retry", "sí", 42)
        assert _all_rows(ae.AGENTIC_EFFICIENCY_FILE) == [("retry", "sí", 42)]

    def test_invalid_tipo_rejected(self, monkeypatch, tmp_path):
        ae = _point_ledger(monkeypatch, tmp_path)
        with pytest.raises(ValueError, match="tipo"):
            ae.append_agentic_efficiency("bogus", "no", 0)
        assert not ae.AGENTIC_EFFICIENCY_FILE.exists() or _all_rows(
            ae.AGENTIC_EFFICIENCY_FILE
        ) == []

    def test_invalid_resultado_rejected(self, monkeypatch, tmp_path):
        ae = _point_ledger(monkeypatch, tmp_path)
        with pytest.raises(ValueError, match="resultado_material"):
            ae.append_agentic_efficiency("nueva", "maybe", 0)

    def test_negative_tokens_rejected(self, monkeypatch, tmp_path):
        ae = _point_ledger(monkeypatch, tmp_path)
        with pytest.raises(ValueError, match="tokens"):
            ae.append_agentic_efficiency("nueva", "no", -1)

    def test_wal_journal_mode(self, monkeypatch, tmp_path):
        ae = _point_ledger(monkeypatch, tmp_path)
        ae.append_agentic_efficiency("nueva", "no", 0)
        with sqlite3.connect(ae.AGENTIC_EFFICIENCY_FILE) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(mode).lower() == "wal"

    def test_concurrent_appends_are_safe(self, monkeypatch, tmp_path):
        ae = _point_ledger(monkeypatch, tmp_path)
        errors: list[BaseException] = []

        def worker(i: int) -> None:
            try:
                ae.append_agentic_efficiency("nueva", "no", i)
            except BaseException as exc:  # noqa: BLE001 — collect for assert
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(40)))
        assert errors == []
        rows = _all_rows(ae.AGENTIC_EFFICIENCY_FILE)
        assert len(rows) == 40
        assert sorted(r[2] for r in rows) == list(range(40))


# ---------------------------------------------------------------------------
# Tipo resolution
# ---------------------------------------------------------------------------


class TestResolveTipo:
    def test_default_nueva(self):
        from cron.agentic_efficiency import resolve_agentic_tipo

        assert resolve_agentic_tipo({}) == "nueva"
        assert resolve_agentic_tipo({"prompt": "x"}) == "nueva"

    def test_explicit_valid_values(self):
        from cron.agentic_efficiency import resolve_agentic_tipo

        assert resolve_agentic_tipo({"agentic_execution_type": "nueva"}) == "nueva"
        assert resolve_agentic_tipo({"agentic_execution_type": "retry"}) == "retry"
        assert resolve_agentic_tipo({"agentic_execution_type": "gate"}) == "gate"

    def test_retry_profile_defaults_to_retry(self):
        from cron.agentic_efficiency import resolve_agentic_tipo

        assert (
            resolve_agentic_tipo({"autonomous_profile": "retry"}) == "retry"
        )

    def test_explicit_gate_wins_over_retry_profile(self):
        from cron.agentic_efficiency import resolve_agentic_tipo

        assert (
            resolve_agentic_tipo(
                {
                    "autonomous_profile": "retry",
                    "agentic_execution_type": "gate",
                }
            )
            == "gate"
        )

    def test_explicit_nueva_wins_over_retry_profile(self):
        from cron.agentic_efficiency import resolve_agentic_tipo

        assert (
            resolve_agentic_tipo(
                {
                    "autonomous_profile": "retry",
                    "agentic_execution_type": "nueva",
                }
            )
            == "nueva"
        )

    def test_resolved_profile_name_retry(self):
        from cron.agentic_efficiency import resolve_agentic_tipo

        assert (
            resolve_agentic_tipo({}, profile_name="retry") == "retry"
        )

    def test_invalid_explicit_fails_closed(self):
        from cron.agentic_efficiency import (
            AgenticTipoError,
            resolve_agentic_tipo,
        )

        with pytest.raises(AgenticTipoError):
            resolve_agentic_tipo({"agentic_execution_type": "warmup"})
        with pytest.raises(AgenticTipoError):
            resolve_agentic_tipo({"agentic_execution_type": "RETRY"})  # case-sensitive exact set


# ---------------------------------------------------------------------------
# Material-result classifier (conservative)
# ---------------------------------------------------------------------------


class TestClassifyResultadoMaterial:
    def test_defaults_to_no_on_empty(self):
        from cron.agentic_efficiency import classify_resultado_material

        assert classify_resultado_material(None) == "no"
        assert classify_resultado_material([]) == "no"
        assert classify_resultado_material("not-a-list") == "no"

    def test_final_narrative_never_counts(self):
        from cron.agentic_efficiency import classify_resultado_material

        messages = [
            {"role": "user", "content": "fix the bug"},
            {
                "role": "assistant",
                "content": "Successfully wrote the fix and committed.",
            },
        ]
        assert classify_resultado_material(messages) == "no"

    def test_model_claiming_success_in_logs_does_not_count(self):
        from cron.agentic_efficiency import classify_resultado_material

        messages = [
            {
                "role": "assistant",
                "content": '{"success": true, "files_modified": ["x.py"]}',
            }
        ]
        assert classify_resultado_material(messages) == "no"

    def test_read_only_tool_success_is_no(self):
        from cron.agentic_efficiency import classify_resultado_material

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "a.py"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "read_file",
                "content": '{"content": "print(1)", "total_lines": 1}',
            },
        ]
        assert classify_resultado_material(messages) == "no"

    def test_write_file_bytes_written_is_si(self, tmp_path):
        from cron.agentic_efficiency import classify_resultado_material

        target = tmp_path / "a.py"
        target.write_text("material", encoding="utf-8")
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "w1",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path": "a.py", "content": "x"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "w1",
                "name": "write_file",
                "content": (
                    f'{{"bytes_written": 8, "resolved_path": "{target}"}}'
                ),
            },
        ]
        assert classify_resultado_material(messages, workdir=tmp_path) == "sí"

    def test_write_file_missing_claimed_path_is_no(self, tmp_path):
        from cron.agentic_efficiency import classify_resultado_material

        missing = tmp_path / "missing.txt"
        messages = [
            {
                "role": "tool",
                "name": "write_file",
                "content": (
                    f'{{"bytes_written": 8, "resolved_path": "{missing}"}}'
                ),
            }
        ]
        assert classify_resultado_material(messages, workdir=tmp_path) == "no"

    def test_write_file_with_error_is_no(self):
        from cron.agentic_efficiency import classify_resultado_material

        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "w1",
                        "function": {"name": "write_file", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "w1",
                "name": "write_file",
                "content": '{"bytes_written": 0, "error": "permission denied"}',
            },
        ]
        assert classify_resultado_material(messages) == "no"

    def test_patch_success_with_files_modified_is_si(self):
        from cron.agentic_efficiency import classify_resultado_material

        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "p1",
                        "function": {"name": "patch", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "p1",
                "name": "patch",
                "content": (
                    '{"success": true, "files_modified": ["src/x.py"], '
                    '"diff": "--- a\\n+++ b"}'
                ),
            },
        ]
        assert classify_resultado_material(messages) == "sí"

    def test_patch_success_false_is_no(self):
        from cron.agentic_efficiency import classify_resultado_material

        messages = [
            {
                "role": "tool",
                "name": "patch",
                "content": '{"success": false, "error": "no match"}',
            }
        ]
        assert classify_resultado_material(messages) == "no"

    def test_terminal_git_commit_exit_0_is_si(self):
        from cron.agentic_efficiency import classify_resultado_material

        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "t1",
                        "function": {
                            "name": "terminal",
                            "arguments": (
                                '{"command": "git commit -m \\"fix\\""}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "t1",
                "name": "terminal",
                "content": (
                    '{"output": "[main abcdef0] fix\\n 1 file changed", '
                    '"exit_code": 0}'
                ),
            },
        ]
        assert classify_resultado_material(messages) == "sí"

    def test_terminal_read_only_exit_0_is_no(self):
        from cron.agentic_efficiency import classify_resultado_material

        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "t1",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command": "ls -la"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "t1",
                "name": "terminal",
                "content": '{"output": "a.py\\nb.py", "exit_code": 0}',
            },
        ]
        assert classify_resultado_material(messages) == "no"

    def test_generic_success_true_without_mutation_is_no(self):
        from cron.agentic_efficiency import classify_resultado_material

        messages = [
            {
                "role": "tool",
                "name": "unknown_tool",
                "content": '{"success": true, "results": ["hit"]}',
            }
        ]
        assert classify_resultado_material(messages) == "no"

    def test_structured_web_search_new_data_is_si(self):
        from cron.agentic_efficiency import classify_resultado_material

        messages = [
            {
                "role": "tool",
                "name": "web_search",
                "content": (
                    '{"data": {"web": [{"title": "new fact", '
                    '"url": "https://example.test/fact"}]}}'
                ),
            }
        ]
        assert classify_resultado_material(messages) == "sí"

    def test_verified_external_action_receipt_is_si(self):
        from cron.agentic_efficiency import classify_resultado_material

        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "cronjob",
                            "arguments": '{"action": "create"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "cronjob",
                "content": '{"job_id": "job-123", "status": "created"}',
            },
        ]
        assert classify_resultado_material(messages) == "sí"

    def test_failed_external_action_is_no(self):
        from cron.agentic_efficiency import classify_resultado_material

        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "cronjob",
                            "arguments": '{"action": "create"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "name": "cronjob",
                "content": '{"success": false, "status": "failed"}',
            },
        ]
        assert classify_resultado_material(messages) == "no"

    def test_nested_failed_external_action_is_no(self):
        from cron.agentic_efficiency import classify_resultado_material

        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c-nested",
                        "function": {
                            "name": "cronjob",
                            "arguments": '{"action": "create"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c-nested",
                "name": "cronjob",
                "content": (
                    '{"job_id": "j", "results": [{"success": false}]}'
                ),
            },
        ]
        assert classify_resultado_material(messages) == "no"

    @pytest.mark.parametrize(
        "payload",
        [
            '{"job_id": "j", "failed": true}',
            '{"job_id": "j", "cancelled": true}',
            '{"job_id": "j", "result": {"failed": true}}',
            '{"job_id": "j", "results": [{"failed": true}]}',
            '{"job_id": "j", "ok": false}',
            '{"job_id": "j", "status": 500}',
            '{"job_id": "j", "exit_code": 1}',
        ],
    )
    def test_external_failure_shapes_are_no(self, payload):
        from cron.agentic_efficiency import classify_resultado_material

        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c-failure-shape",
                        "function": {
                            "name": "cronjob",
                            "arguments": '{"action": "create"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c-failure-shape",
                "name": "cronjob",
                "content": payload,
            },
        ]
        assert classify_resultado_material(messages) == "no"

    def test_verified_filesystem_evidence_is_si(self, tmp_path):
        from cron.agentic_efficiency import classify_resultado_material

        target = tmp_path / "created_by_agent.txt"
        target.write_text("material", encoding="utf-8")
        messages = [
            {
                "role": "tool",
                "name": "write_file",
                "content": (
                    f'{{"bytes_written": 8, "resolved_path": "{target}"}}'
                ),
            }
        ]
        assert (
            classify_resultado_material(messages, workdir=tmp_path) == "sí"
        )

    def test_unverified_path_claim_without_tool_shape_is_no(self, tmp_path):
        from cron.agentic_efficiency import classify_resultado_material

        # File exists on disk, but assistant prose alone must not count.
        (tmp_path / "ghost.txt").write_text("x", encoding="utf-8")
        messages = [
            {
                "role": "assistant",
                "content": f"Wrote {tmp_path / 'ghost.txt'} successfully",
            }
        ]
        assert (
            classify_resultado_material(messages, workdir=tmp_path) == "no"
        )


# ---------------------------------------------------------------------------
# Final worker snapshot
# ---------------------------------------------------------------------------


class TestRecordAfterWorker:
    def test_pending_future_records_final_messages_and_tokens(
        self, monkeypatch, tmp_path
    ):
        from cron import agentic_efficiency as ae

        _point_ledger(monkeypatch, tmp_path)
        target = tmp_path / "future.txt"
        target.write_text("material", encoding="utf-8")
        future: concurrent.futures.Future = concurrent.futures.Future()
        agent = MagicMock()
        agent.session_total_tokens = 5
        agent._cron_worker_future = future
        recorder = ae.OnceOnlyAgenticRecorder("gate")

        immediate = ae.record_agentic_efficiency_after_worker(
            recorder,
            agent=agent,
            fallback_result=None,
            workdir=tmp_path,
        )
        assert immediate is False
        assert not ae.AGENTIC_EFFICIENCY_FILE.exists()

        agent.session_total_tokens = 987
        future.set_result(
            {
                "messages": [
                    {
                        "role": "tool",
                        "name": "write_file",
                        "content": (
                            f'{{"bytes_written": 8, "resolved_path": "{target}"}}'
                        ),
                    }
                ]
            }
        )
        assert _all_rows(ae.AGENTIC_EFFICIENCY_FILE) == [
            ("gate", "sí", 987)
        ]


# ---------------------------------------------------------------------------
# Scheduler integration
# ---------------------------------------------------------------------------


class TestSchedulerAgenticLedger:
    def test_agentic_success_appends_exactly_one_row(
        self, monkeypatch, tmp_path
    ):
        ae = _point_ledger(monkeypatch, tmp_path)
        agent = MagicMock()
        agent.session_total_tokens = 1234
        agent.close = MagicMock()

        success, _out, final, error = _run_job_with_agent(
            {"id": "j1", "name": "t", "prompt": "hello"},
            tmp_path=tmp_path,
            agent=agent,
            result={
                "final_response": "done",
                "completed": True,
                "failed": False,
                "messages": [],
            },
        )
        assert success is True
        assert error is None
        assert final == "done"
        rows = _all_rows(ae.AGENTIC_EFFICIENCY_FILE)
        assert rows == [("nueva", "no", 1234)]

    def test_agentic_failure_path_still_appends_once(
        self, monkeypatch, tmp_path
    ):
        ae = _point_ledger(monkeypatch, tmp_path)
        agent = MagicMock()
        agent.session_total_tokens = 50
        agent.close = MagicMock()

        success, _out, _final, error = _run_job_with_agent(
            {"id": "j2", "name": "t", "prompt": "hello"},
            tmp_path=tmp_path,
            agent=agent,
            result={
                "final_response": "boom",
                "completed": False,
                "failed": True,
                "error": "provider exploded",
                "messages": [],
            },
        )
        assert success is False
        assert error
        rows = _all_rows(ae.AGENTIC_EFFICIENCY_FILE)
        assert len(rows) == 1
        assert rows[0] == ("nueva", "no", 50)

    def test_limit_path_appends_once(self, monkeypatch, tmp_path):
        from cron.autonomous_limits import AutonomousLimitError

        ae = _point_ledger(monkeypatch, tmp_path)
        agent = MagicMock()
        agent.session_total_tokens = 999
        agent.close = MagicMock()

        success, _out, _final, error = _run_job_with_agent(
            {"id": "j3", "name": "t", "prompt": "hello"},
            tmp_path=tmp_path,
            agent=agent,
            raise_exc=AutonomousLimitError(
                "max-turns", limit_count=1, action="rescope-permitted"
            ),
        )
        assert success is False
        assert error
        rows = _all_rows(ae.AGENTIC_EFFICIENCY_FILE)
        assert rows == [("nueva", "no", 999)]

    def test_no_agent_does_not_append(self, monkeypatch, tmp_path):
        ae = _point_ledger(monkeypatch, tmp_path)
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        scripts = home / "scripts"
        scripts.mkdir(exist_ok=True)
        script = scripts / "w.sh"
        script.write_text("#!/bin/bash\necho hi\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))

        from cron.scheduler import run_job

        job = {
            "id": "na1",
            "name": "watch",
            "no_agent": True,
            "script": str(script),
        }
        with patch("cron.scheduler._hermes_home", home), \
             patch(
                 "cron.scheduler._run_job_script_with_claim_heartbeat",
                 return_value=(True, "hi"),
             ):
            success, _doc, final, error = run_job(job)

        assert success is True
        assert error is None
        assert final == "hi"
        assert not ae.AGENTIC_EFFICIENCY_FILE.exists() or _all_rows(
            ae.AGENTIC_EFFICIENCY_FILE
        ) == []

    def test_invalid_tipo_fails_before_model_dispatch(
        self, monkeypatch, tmp_path
    ):
        ae = _point_ledger(monkeypatch, tmp_path)
        agent_cls = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=MagicMock()), \
             patch("run_agent.AIAgent", agent_cls) as mock_agent:
            from cron.scheduler import run_job

            success, _out, _final, error = run_job(
                {
                    "id": "bad-tipo",
                    "name": "t",
                    "prompt": "hello",
                    "agentic_execution_type": "warmup",
                }
            )

        assert success is False
        assert error
        assert "agentic_execution_type" in error or "tipo" in error.lower()
        mock_agent.assert_not_called()
        assert not ae.AGENTIC_EFFICIENCY_FILE.exists() or _all_rows(
            ae.AGENTIC_EFFICIENCY_FILE
        ) == []

    def test_retry_profile_sets_tipo_retry(self, monkeypatch, tmp_path):
        ae = _point_ledger(monkeypatch, tmp_path)
        agent = MagicMock()
        agent.session_total_tokens = 10
        agent.close = MagicMock()

        _run_job_with_agent(
            {
                "id": "j4",
                "name": "t",
                "prompt": "hello",
                "autonomous_profile": "retry",
            },
            tmp_path=tmp_path,
            agent=agent,
        )
        rows = _all_rows(ae.AGENTIC_EFFICIENCY_FILE)
        assert rows == [("retry", "no", 10)]

    def test_material_si_from_tool_history(self, monkeypatch, tmp_path):
        ae = _point_ledger(monkeypatch, tmp_path)
        agent = MagicMock()
        agent.session_total_tokens = 7
        agent.close = MagicMock()
        target = tmp_path / "material.txt"
        target.write_text("abc", encoding="utf-8")
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "w1",
                        "function": {
                            "name": "write_file",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "w1",
                "name": "write_file",
                "content": (
                    f'{{"bytes_written": 3, "resolved_path": "{target}"}}'
                ),
            },
        ]
        _run_job_with_agent(
            {
                "id": "j5",
                "name": "t",
                "prompt": "hello",
                "agentic_execution_type": "gate",
            },
            tmp_path=tmp_path,
            agent=agent,
            result={
                "final_response": "I wrote the file successfully!",
                "completed": True,
                "failed": False,
                "messages": messages,
            },
        )
        rows = _all_rows(ae.AGENTIC_EFFICIENCY_FILE)
        assert rows == [("gate", "sí", 7)]

    def test_ledger_failure_is_visible_and_does_not_duplicate(
        self, monkeypatch, tmp_path
    ):
        ae = _point_ledger(monkeypatch, tmp_path)
        agent = MagicMock()
        agent.session_total_tokens = 1
        agent.close = MagicMock()

        calls = {"n": 0}
        real_append = ae.append_agentic_efficiency

        def flaky_once(tipo, resultado_material, tokens):
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("disk I/O error")
            return real_append(tipo, resultado_material, tokens)

        monkeypatch.setattr(ae, "append_agentic_efficiency", flaky_once)
        # Scheduler imports the symbol — patch the name used at call site.
        monkeypatch.setattr(
            "cron.scheduler.append_agentic_efficiency",
            flaky_once,
            raising=False,
        )

        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            _run_job_with_agent(
                {"id": "j6", "name": "t", "prompt": "hello"},
                tmp_path=tmp_path,
                agent=agent,
            )
        assert calls["n"] == 1
        if ae.AGENTIC_EFFICIENCY_FILE.exists():
            assert _all_rows(ae.AGENTIC_EFFICIENCY_FILE) == []

    def test_existing_executions_ledger_unchanged(self, monkeypatch, tmp_path):
        """Rich executions table shape must not grow Task-A columns."""
        import cron.executions as executions

        monkeypatch.setattr(
            executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
        )
        row = executions.create_execution("job-x", source="builtin")
        executions.finish_execution(row["id"], success=True)
        with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
            cols = [
                r[1]
                for r in conn.execute("PRAGMA table_info(executions)").fetchall()
            ]
        assert "tipo" not in cols
        assert "resultado_material" not in cols
        assert "agentic_efficiency" not in cols
        # agentic table must live in its own DB file, not executions.db
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "agentic_efficiency" not in tables
