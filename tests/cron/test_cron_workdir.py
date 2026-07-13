"""Tests for per-job workdir support in cron jobs.

Covers:
  - jobs.create_job: param plumbing, validation, default-None preserved
  - jobs._normalize_workdir: absolute / relative / missing / file-not-dir
  - jobs.update_job: set, clear, re-validate
  - tools.cronjob_tools.cronjob: create + update JSON round-trip, schema
    includes workdir, _format_job exposes it when set
  - scheduler.tick(): runs workdir jobs on the shared non-blocking parallel
    pool (no sequential partition), and run_job enables context-file discovery
    for a workdir WITHOUT mutating the process-global TERMINAL_CWD.  The
    concurrency + per-job cwd-isolation contract lives in
    test_cron_workdir_concurrency.py.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Isolate cron job storage into a temp dir so tests don't stomp on real jobs."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


# ---------------------------------------------------------------------------
# jobs._normalize_workdir
# ---------------------------------------------------------------------------

class TestNormalizeWorkdir:
    def test_none_returns_none(self):
        from cron.jobs import _normalize_workdir
        assert _normalize_workdir(None) is None

    def test_empty_string_returns_none(self):
        from cron.jobs import _normalize_workdir
        assert _normalize_workdir("") is None
        assert _normalize_workdir("   ") is None

    def test_absolute_existing_dir_returns_resolved_str(self, tmp_path):
        from cron.jobs import _normalize_workdir
        result = _normalize_workdir(str(tmp_path))
        assert result == str(tmp_path.resolve())

    def test_tilde_expands(self, tmp_path, monkeypatch):
        from cron.jobs import _normalize_workdir
        monkeypatch.setenv("HOME", str(tmp_path))
        result = _normalize_workdir("~")
        assert result == str(tmp_path.resolve())

    def test_relative_path_rejected(self):
        from cron.jobs import _normalize_workdir
        with pytest.raises(ValueError, match="absolute path"):
            _normalize_workdir("some/relative/path")

    def test_missing_dir_rejected(self, tmp_path):
        from cron.jobs import _normalize_workdir
        missing = tmp_path / "does-not-exist"
        with pytest.raises(ValueError, match="does not exist"):
            _normalize_workdir(str(missing))

    def test_file_not_dir_rejected(self, tmp_path):
        from cron.jobs import _normalize_workdir
        f = tmp_path / "file.txt"
        f.write_text("hi")
        with pytest.raises(ValueError, match="not a directory"):
            _normalize_workdir(str(f))


# ---------------------------------------------------------------------------
# jobs.create_job and update_job
# ---------------------------------------------------------------------------

class TestCreateJobWorkdir:
    def test_workdir_stored_when_set(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job
        job = create_job(
            prompt="hello",
            schedule="every 1h",
            workdir=str(tmp_cron_dir),
        )
        stored = get_job(job["id"])
        assert stored["workdir"] == str(tmp_cron_dir.resolve())

    def test_workdir_none_preserves_old_behaviour(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job
        job = create_job(prompt="hello", schedule="every 1h")
        stored = get_job(job["id"])
        # Field is present on the dict but None — downstream code checks
        # truthiness to decide whether the feature is active.
        assert stored.get("workdir") is None

    def test_create_rejects_invalid_workdir(self, tmp_cron_dir):
        from cron.jobs import create_job
        with pytest.raises(ValueError):
            create_job(
                prompt="hello",
                schedule="every 1h",
                workdir="not/absolute",
            )


class TestUpdateJobWorkdir:
    def test_set_workdir_via_update(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job, update_job
        job = create_job(prompt="x", schedule="every 1h")
        update_job(job["id"], {"workdir": str(tmp_cron_dir)})
        assert get_job(job["id"])["workdir"] == str(tmp_cron_dir.resolve())

    def test_clear_workdir_with_none(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job, update_job
        job = create_job(
            prompt="x", schedule="every 1h", workdir=str(tmp_cron_dir)
        )
        update_job(job["id"], {"workdir": None})
        assert get_job(job["id"])["workdir"] is None

    def test_clear_workdir_with_empty_string(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job, update_job
        job = create_job(
            prompt="x", schedule="every 1h", workdir=str(tmp_cron_dir)
        )
        update_job(job["id"], {"workdir": ""})
        assert get_job(job["id"])["workdir"] is None

    def test_update_rejects_invalid_workdir(self, tmp_cron_dir):
        from cron.jobs import create_job, update_job
        job = create_job(prompt="x", schedule="every 1h")
        with pytest.raises(ValueError):
            update_job(job["id"], {"workdir": "nope/relative"})


# ---------------------------------------------------------------------------
# tools.cronjob_tools: end-to-end JSON round-trip
# ---------------------------------------------------------------------------

class TestCronjobToolWorkdir:
    def test_create_with_workdir_json_roundtrip(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob

        result = json.loads(
            cronjob(
                action="create",
                prompt="hi",
                schedule="every 1h",
                workdir=str(tmp_cron_dir),
            )
        )
        assert result["success"] is True
        assert result["job"]["workdir"] == str(tmp_cron_dir.resolve())

    def test_create_without_workdir_hides_field_in_format(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob

        result = json.loads(
            cronjob(
                action="create",
                prompt="hi",
                schedule="every 1h",
            )
        )
        assert result["success"] is True
        # _format_job omits the field when unset — reduces noise in agent output.
        assert "workdir" not in result["job"]

    def test_update_clears_workdir_with_empty_string(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob

        created = json.loads(
            cronjob(
                action="create",
                prompt="hi",
                schedule="every 1h",
                workdir=str(tmp_cron_dir),
            )
        )
        job_id = created["job_id"]

        updated = json.loads(
            cronjob(action="update", job_id=job_id, workdir="")
        )
        assert updated["success"] is True
        assert "workdir" not in updated["job"]

    def test_schema_advertises_workdir(self):
        from tools.cronjob_tools import CRONJOB_SCHEMA
        assert "workdir" in CRONJOB_SCHEMA["parameters"]["properties"]
        desc = CRONJOB_SCHEMA["parameters"]["properties"]["workdir"]["description"]
        assert "absolute" in desc.lower()


# ---------------------------------------------------------------------------
# scheduler.tick(): workdir jobs run on the shared parallel pool
# ---------------------------------------------------------------------------

class TestTickWorkdirParallel:
    """
    tick() dispatches workdir jobs to the SAME non-blocking parallel pool as
    workdir-less jobs.  They used to be funnelled through a single-thread
    ``cron-seq`` pool because run_job mutated the process-global
    os.environ["TERMINAL_CWD"]; now each job isolates its own cwd (ContextVar +
    per-task terminal env override), so a long workdir job no longer serializes
    the rest of the schedule.
    """

    def test_workdir_jobs_run_on_parallel_pool(self, tmp_path, monkeypatch):
        import cron.scheduler as sched

        sched._parallel_pool = None
        sched._parallel_pool_max_workers = None
        sched._running_job_ids.clear()

        # Two workdir jobs + one workdir-less job — all should run on the
        # parallel pool now (no sequential partition).
        workdir_a = {"id": "a", "name": "A", "workdir": str(tmp_path)}
        workdir_b = {"id": "b", "name": "B", "workdir": str(tmp_path)}
        parallel_job = {"id": "c", "name": "C", "workdir": None}

        monkeypatch.setattr(sched, "get_due_jobs", lambda: [workdir_a, workdir_b, parallel_job])
        monkeypatch.setattr(sched, "advance_next_run", lambda *_a, **_kw: None)

        # Record which thread each job ran on.
        import threading
        calls: list[tuple[str, str]] = []
        order_lock = threading.Lock()

        def fake_run_job(job, *, defer_agent_teardown=None):
            with order_lock:
                calls.append((job["id"], threading.current_thread().name))
            return True, "output", "response", None

        monkeypatch.setattr(sched, "run_job", fake_run_job)
        monkeypatch.setattr(sched, "save_job_output", lambda _jid, _o: None)
        monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
        monkeypatch.setattr(
            sched, "_deliver_result", lambda *_a, **_kw: None
        )

        n = sched.tick(verbose=False)
        assert n == 3
        assert {c[0] for c in calls} == {"a", "b", "c"}

        # Every job — workdir or not — runs on the persistent parallel pool,
        # off the ticker thread, so no single job can block the schedule.
        main_thread_name = threading.current_thread().name
        for jid, tname in calls:
            assert tname != main_thread_name, jid
            assert tname.startswith("cron-parallel"), (jid, tname)

        sched._shutdown_parallel_pool()


# ---------------------------------------------------------------------------
# scheduler.run_job: TERMINAL_CWD + skip_context_files wiring
# ---------------------------------------------------------------------------

class TestRunJobTerminalCwd:
    """
    run_job flips skip_context_files=False when a workdir is set, and does NOT
    mutate the process-global os.environ["TERMINAL_CWD"] — per-job cwd is now
    isolated via the _SESSION_CWD ContextVar + a per-task terminal env override
    (that contract is exercised in test_cron_workdir_concurrency.py).  We stub
    AIAgent so no real API call happens.
    """

    @staticmethod
    def _install_stubs(monkeypatch, observed: dict):
        """Patch enough of run_job's deps that it executes without real creds."""
        import os
        import sys
        import cron.scheduler as sched

        class FakeAgent:
            def __init__(self, **kwargs):
                observed["skip_context_files"] = kwargs.get("skip_context_files")
                observed["load_soul_identity"] = kwargs.get("load_soul_identity")
                observed["terminal_cwd_during_init"] = os.environ.get(
                    "TERMINAL_CWD", "_UNSET_"
                )

            def run_conversation(self, *_a, **_kw):
                observed["terminal_cwd_during_run"] = os.environ.get(
                    "TERMINAL_CWD", "_UNSET_"
                )
                return {"final_response": "done", "messages": []}

            def get_activity_summary(self):
                return {"seconds_since_activity": 0.0}

        fake_mod = type(sys)("run_agent")
        fake_mod.AIAgent = FakeAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_mod)

        # Bypass the real provider resolver — it reads ~/.hermes and credentials.
        from hermes_cli import runtime_provider as _rtp
        monkeypatch.setattr(
            _rtp,
            "resolve_runtime_provider",
            lambda **_kw: {
                "provider": "test",
                "api_key": "k",
                "base_url": "http://test.local",
                "api_mode": "chat_completions",
            },
        )

        # Stub scheduler helpers that would otherwise hit the filesystem / config.
        monkeypatch.setattr(sched, "_build_job_prompt", lambda job, prerun_script=None: "hi")
        monkeypatch.setattr(sched, "_resolve_origin", lambda job: None)
        monkeypatch.setattr(sched, "_resolve_delivery_target", lambda job: None)
        monkeypatch.setattr(sched, "_resolve_cron_enabled_toolsets", lambda job, cfg: None)
        # Unlimited inactivity so the poll loop returns immediately.
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")

        # run_job calls load_dotenv(~/.hermes/.env, override=True), which will
        # happily clobber TERMINAL_CWD out from under us if the real user .env
        # has TERMINAL_CWD set (common on dev boxes).  Stub it out.
        import dotenv
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *_a, **_kw: True)

    def test_workdir_enables_context_files_without_touching_terminal_cwd(
        self, tmp_path, monkeypatch
    ):
        import os
        import cron.scheduler as sched

        # Pin TERMINAL_CWD to a known non-workdir value via monkeypatch so it's
        # restored on teardown regardless of cross-test state.
        monkeypatch.setenv("TERMINAL_CWD", "/original/cwd")

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        job = {
            "id": "abc",
            "name": "wd-job",
            "workdir": str(tmp_path),
            "schedule_display": "manual",
        }

        success, _output, response, error = sched.run_job(job)
        assert success is True, f"run_job failed: error={error!r} response={response!r}"

        # AIAgent was built with skip_context_files=False (context-file
        # discovery ON) so AGENTS.md / CLAUDE.md from the workdir are loaded.
        assert observed["skip_context_files"] is False
        assert observed["load_soul_identity"] is True
        # run_job must NOT point the process-global TERMINAL_CWD at the workdir
        # — that global mutation is exactly what forced serialization.  It
        # stays at whatever it was before the job ran, throughout the run.
        assert observed["terminal_cwd_during_init"] == "/original/cwd"
        assert observed["terminal_cwd_during_run"] == "/original/cwd"
        assert os.environ["TERMINAL_CWD"] == "/original/cwd"

    def test_no_workdir_leaves_terminal_cwd_untouched(self, monkeypatch):
        """When workdir is absent, run_job must not touch TERMINAL_CWD at all —
        whatever value was present before the call should be present after.

        We don't assert on the *content* of TERMINAL_CWD (other tests in the
        same xdist worker may leave it set to something like '.'); we just
        check it's unchanged by run_job.
        """
        import os
        import cron.scheduler as sched

        # Pin TERMINAL_CWD to a sentinel via monkeypatch so we control both
        # the before-value and the after-value regardless of cross-test state.
        monkeypatch.setenv("TERMINAL_CWD", "/cron-test-sentinel")
        before = os.environ["TERMINAL_CWD"]

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        job = {
            "id": "xyz",
            "name": "no-wd-job",
            "workdir": None,
            "schedule_display": "manual",
        }

        success, *_ = sched.run_job(job)
        assert success is True

        # Feature is OFF — skip_context_files stays True.
        assert observed["skip_context_files"] is True
        # Cron still forces SOUL.md identity even when cwd context files stay off.
        assert observed["load_soul_identity"] is True
        # TERMINAL_CWD saw the same value during init as it had before.
        assert observed["terminal_cwd_during_init"] == before
        # And after run_job completes, it's still the sentinel (nothing
        # overwrote or cleared it).
        assert os.environ["TERMINAL_CWD"] == before
