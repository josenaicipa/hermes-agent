"""Concurrency + cwd-isolation contract for per-job cron workdirs.

Regression coverage for the production incident where a due workdir cron job
was claimed/advanced but never executed because *every* workdir job was
funnelled through one global single-thread
``cron-seq`` pool, and ``run_job`` additionally held a process-wide
readers/writer lock while mutating ``os.environ["TERMINAL_CWD"]`` for the
whole agent lifetime.  A long workdir job therefore blocked every other
workdir cron across unrelated projects.

The fix isolates each job's working directory through the existing
per-session / per-task infrastructure instead of process-global state:

  * ``set_session_cwd`` (the ``_SESSION_CWD`` ContextVar, via
    ``set_session_vars(cwd=...)``) — drives system-prompt / context-file
    discovery and the Codex runtime.
  * ``register_task_env_overrides(task_id, {"cwd": ..., "isolate_env": True})``
    — drives the terminal / file / execute_code tools, and gives each job its
    OWN terminal environment so concurrent jobs can't clobber a shared
    ``env.cwd``.
  * ``task_id`` threaded into ``AIAgent.run_conversation`` so the overrides
    above actually reach the tool calls.

With that isolation in place the global ``TERMINAL_CWD`` mutation, the
readers/writer lock, and the sequential pool are all unnecessary — so two
workdir jobs with different directories must run concurrently, each observing
only its own cwd, and one long job must never block another.

These tests are deliberately white-box about the isolation channels
(`agent.runtime_cwd._SESSION_CWD`, `terminal_tool.resolve_task_overrides`)
because that is exactly the contract the fix must honour.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest


_UNSET = "__UNSET__"

# Shared, per-test state read by the injected FakeAgent.  Reset by the
# ``fake_agent_state`` fixture before every test so leakage between tests is
# impossible even under xdist.
_STATE: dict = {}


class _FakeEnvironment:
    def __init__(self):
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


class _FakeAgent:
    """Stand-in for ``run_agent.AIAgent`` that records the cwd-isolation
    signals visible while ``run_conversation`` runs, without touching a real
    model, provider, or tool sandbox."""

    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id")
        _STATE.setdefault("init", {})[self.session_id] = {
            "skip_context_files": kwargs.get("skip_context_files"),
            "load_soul_identity": kwargs.get("load_soul_identity"),
            "terminal_cwd_env": os.environ.get("TERMINAL_CWD", _UNSET),
        }

    def run_conversation(self, user_message, *args, **kwargs):
        import agent.runtime_cwd as rc
        from tools.terminal_tool import resolve_task_overrides

        # task_id is passed as a keyword by the fixed scheduler; on the old
        # code path it is never supplied and therefore reads back as None.
        task_id = kwargs.get("task_id")

        # If a barrier is configured, block until every concurrent job reaches
        # this point.  On the buggy code path a second workdir job can never
        # get here (it is serialized behind the sequential pool / write lock),
        # so the barrier times out and this raises — which is the failure the
        # concurrency test asserts against.
        barrier = _STATE.get("barrier")
        if barrier is not None:
            barrier.wait()  # BrokenBarrierError on timeout -> surfaced as failure

        ctx_cwd = rc.resolve_context_cwd()
        _STATE.setdefault("run", {})[self.session_id] = {
            "task_id": task_id,
            "session_cwd_var": rc._SESSION_CWD.get(),
            "context_cwd": str(ctx_cwd) if ctx_cwd is not None else None,
            "override_cwd": (
                resolve_task_overrides(task_id).get("cwd") if task_id else None
            ),
            "terminal_cwd_env": os.environ.get("TERMINAL_CWD", _UNSET),
        }

        entered = _STATE.get("entered_event")
        if entered is not None:
            entered.set()
        blocker = _STATE.get("block_event")
        if blocker is not None:
            blocker.wait(timeout=10)
        late_env = _STATE.get("late_env")
        if late_env is not None and task_id:
            from tools import terminal_tool

            with terminal_tool._env_lock:
                terminal_tool._active_environments[task_id] = late_env

        if _STATE.get("raise_in_run"):
            raise RuntimeError("boom (injected agent failure)")
        return {"final_response": "done", "messages": []}

    def get_activity_summary(self):
        return {
            "seconds_since_activity": _STATE.get("idle_seconds", 0.0),
            "last_activity_desc": "test agent blocked",
            "current_tool": "terminal",
            "api_call_count": 1,
            "max_iterations": 10,
        }

    def interrupt(self, *_a, **_k):
        pass

    def close(self):
        from tools.terminal_tool import cleanup_vm

        _STATE.setdefault("closed", []).append(self.session_id)
        if self.session_id:
            cleanup_vm(self.session_id)


@pytest.fixture()
def fake_agent_state():
    _STATE.clear()
    yield _STATE
    _STATE.clear()


@pytest.fixture()
def stub_run_job(monkeypatch):
    """Patch enough of ``run_job``'s dependencies that it runs end-to-end with
    ``_FakeAgent`` and no real credentials / model / filesystem config."""
    import cron.scheduler as sched

    fake_mod = type(sys)("run_agent")
    fake_mod.AIAgent = _FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_mod)

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

    monkeypatch.setattr(sched, "_build_job_prompt", lambda job, prerun_script=None: "hi")
    monkeypatch.setattr(sched, "_resolve_origin", lambda job: None)
    monkeypatch.setattr(sched, "_resolve_delivery_target", lambda job: None)
    monkeypatch.setattr(sched, "_resolve_cron_enabled_toolsets", lambda job, cfg: None)
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")

    # run_job re-loads ~/.hermes/.env; keep it from clobbering TERMINAL_CWD.
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *_a, **_kw: True)
    return sched


def _workdir_job(job_id: str, workdir) -> dict:
    return {
        "id": job_id,
        "name": f"wd-{job_id}",
        "workdir": str(workdir),
        "schedule_display": "manual",
    }


# ---------------------------------------------------------------------------
# Per-job isolation channels wired by run_job
# ---------------------------------------------------------------------------

class TestRunJobIsolationWiring:
    def test_registers_task_cwd_override_during_run(
        self, tmp_path, monkeypatch, fake_agent_state, stub_run_job
    ):
        """The workdir must be exposed to the terminal/file/execute_code tools
        through the per-task override registry (keyed by the conversation
        task_id), NOT through process-global env."""
        sched = stub_run_job
        job = _workdir_job("iso1", tmp_path)

        ok, *_ = sched.run_job(job)
        assert ok is True

        run = _only_run(_STATE)
        assert run["override_cwd"] == str(tmp_path)

    def test_sets_session_cwd_contextvar_during_run(
        self, tmp_path, monkeypatch, fake_agent_state, stub_run_job
    ):
        """The workdir must be pinned on the ``_SESSION_CWD`` ContextVar so
        system-prompt / context-file discovery and the Codex runtime resolve
        the job's project dir."""
        sched = stub_run_job
        job = _workdir_job("iso2", tmp_path)

        ok, *_ = sched.run_job(job)
        assert ok is True

        run = _only_run(_STATE)
        assert run["session_cwd_var"] == str(tmp_path)
        assert run["context_cwd"] == str(tmp_path)

    def test_passes_task_id_to_run_conversation(
        self, tmp_path, monkeypatch, fake_agent_state, stub_run_job
    ):
        """run_conversation must receive a non-None task_id (so the overrides
        above reach the tools).  It should match the cron session id used for
        agent teardown / sandbox cleanup."""
        sched = stub_run_job
        job = _workdir_job("iso3", tmp_path)

        ok, *_ = sched.run_job(job)
        assert ok is True

        (sid, run), = _STATE["run"].items()
        assert run["task_id"], "run_conversation was called without a task_id"
        assert run["task_id"] == sid

    def test_does_not_mutate_global_terminal_cwd(
        self, tmp_path, monkeypatch, fake_agent_state, stub_run_job
    ):
        """run_job must not point the process-global TERMINAL_CWD at the
        workdir — that global mutation is exactly what forced serialization."""
        sched = stub_run_job
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        job = _workdir_job("iso4", tmp_path)

        ok, *_ = sched.run_job(job)
        assert ok is True

        run = _only_run(_STATE)
        assert run["terminal_cwd_env"] == _UNSET, (
            "run_job mutated os.environ['TERMINAL_CWD'] during the agent run"
        )
        assert os.environ.get("TERMINAL_CWD", _UNSET) == _UNSET


# ---------------------------------------------------------------------------
# Cleanup — on success and on failure
# ---------------------------------------------------------------------------

class TestRunJobCleanup:
    def test_clears_task_override_after_success(
        self, tmp_path, monkeypatch, fake_agent_state, stub_run_job
    ):
        from tools.terminal_tool import resolve_task_overrides

        sched = stub_run_job
        job = _workdir_job("cl1", tmp_path)

        ok, *_ = sched.run_job(job)
        assert ok is True

        run = _only_run(_STATE)
        assert run["override_cwd"] == str(tmp_path)  # was set during the run
        sid = next(iter(_STATE["run"]))
        assert resolve_task_overrides(sid) == {}  # ...and cleared afterwards

    def test_clears_isolation_on_exception(
        self, tmp_path, monkeypatch, fake_agent_state, stub_run_job
    ):
        import agent.runtime_cwd as rc
        from tools.terminal_tool import resolve_task_overrides

        sched = stub_run_job
        _STATE["raise_in_run"] = True
        job = _workdir_job("cl2", tmp_path)

        ok, _out, _resp, err = sched.run_job(job)
        assert ok is False
        assert err  # failure surfaced

        run = _only_run(_STATE)
        assert run["override_cwd"] == str(tmp_path)  # override was live during run
        sid = next(iter(_STATE["run"]))
        # Cleanup must still run in the finally block on the failure path.
        assert resolve_task_overrides(sid) == {}
        cur = rc._SESSION_CWD.get()
        assert cur in ("", rc._UNSET), f"_SESSION_CWD not cleared after failure: {cur!r}"


    def test_timeout_keeps_override_until_worker_finishes(self, tmp_path, monkeypatch, fake_agent_state, stub_run_job):
        """A timed-out agent may still unwind in its worker thread; its cwd
        override must survive until that future is actually done."""
        from tools.terminal_tool import resolve_task_overrides

        sched = stub_run_job
        entered = threading.Event()
        release = threading.Event()
        _STATE["entered_event"] = entered
        _STATE["block_event"] = release
        _STATE["idle_seconds"] = 999.0
        late_env = _FakeEnvironment()
        _STATE["late_env"] = late_env
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0.01")

        try:
            ok, _out, _resp, err = sched.run_job(_workdir_job("timeout", tmp_path))
            assert ok is False
            assert "TimeoutError" in (err or "")
            assert entered.is_set()
            sid = next(iter(_STATE["run"]))
            assert resolve_task_overrides(sid)["cwd"] == str(tmp_path)
            assert sid not in _STATE.get("closed", [])

            release.set()
            deadline = time.monotonic() + 2
            while (
                (resolve_task_overrides(sid) or not late_env.cleaned)
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert resolve_task_overrides(sid) == {}
            assert late_env.cleaned is True
        finally:
            release.set()


# ---------------------------------------------------------------------------
# The headline: concurrency without cross-job cwd leakage
# ---------------------------------------------------------------------------

class TestWorkdirConcurrency:
    def test_two_workdir_jobs_overlap_without_cwd_leak(
        self, tmp_path, monkeypatch, fake_agent_state, stub_run_job
    ):
        """Two workdir jobs with DIFFERENT directories must be able to execute
        at the same time (proving one long job can't block another), and each
        must observe only its own working directory.

        A ``threading.Barrier(2)`` forces genuine overlap: neither FakeAgent
        can finish until BOTH have entered ``run_conversation``.  On the buggy
        code path the second job is serialized (sequential pool + write lock)
        and never arrives, so the barrier times out and the job fails.
        """
        sched = stub_run_job
        monkeypatch.delenv("TERMINAL_CWD", raising=False)

        wd_a = tmp_path / "proj_a"
        wd_b = tmp_path / "proj_b"
        wd_a.mkdir()
        wd_b.mkdir()

        # timeout so the buggy (serialized) path fails fast instead of hanging.
        _STATE["barrier"] = threading.Barrier(2, timeout=8.0)

        results: dict = {}

        def _run(job):
            results[job["id"]] = sched.run_job(job)

        job_a = _workdir_job("cc_a", wd_a)
        job_b = _workdir_job("cc_b", wd_b)
        t_a = threading.Thread(target=_run, args=(job_a,), name="job-a")
        t_b = threading.Thread(target=_run, args=(job_b,), name="job-b")
        t_a.start()
        t_b.start()
        t_a.join(timeout=30)
        t_b.join(timeout=30)
        assert not t_a.is_alive() and not t_b.is_alive(), "run_job threads hung"

        # Both jobs succeeded → both passed the barrier → they overlapped.
        assert results["cc_a"][0] is True, f"job A failed: {results['cc_a']}"
        assert results["cc_b"][0] is True, f"job B failed: {results['cc_b']}"

        runs = _STATE["run"]
        assert len(runs) == 2, f"expected both jobs to run concurrently, got {runs}"

        # Each concurrently-active job saw ONLY its own cwd — no leakage.
        rec_a = _find_rec(runs, wd_a)
        rec_b = _find_rec(runs, wd_b)
        assert rec_a["session_cwd_var"] == str(wd_a)
        assert rec_a["override_cwd"] == str(wd_a)
        assert rec_b["session_cwd_var"] == str(wd_b)
        assert rec_b["override_cwd"] == str(wd_b)
        # No process-global env carried either workdir.
        for rec in runs.values():
            assert rec["terminal_cwd_env"] == _UNSET


    def test_same_job_same_second_gets_unique_task_ids(
        self, tmp_path, monkeypatch, fake_agent_state, stub_run_job
    ):
        """Manual and scheduled executions of one job can overlap; even with a
        frozen clock they must not share task overrides or sandbox state."""
        sched = stub_run_job
        fixed_now = sched._hermes_now()
        monkeypatch.setattr(sched, "_hermes_now", lambda: fixed_now)

        wd_a = tmp_path / "same_a"
        wd_b = tmp_path / "same_b"
        wd_a.mkdir()
        wd_b.mkdir()
        _STATE["barrier"] = threading.Barrier(2, timeout=8.0)

        results = {}

        def _run(label, workdir):
            results[label] = sched.run_job(_workdir_job("same_job", workdir))

        t1 = threading.Thread(target=_run, args=("a", wd_a), daemon=True)
        t2 = threading.Thread(target=_run, args=("b", wd_b), daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=12)
        t2.join(timeout=12)

        assert not t1.is_alive() and not t2.is_alive()
        assert all(value[0] is True for value in results.values())
        assert len(_STATE["run"]) == 2, "same-second runs reused one task_id"
        task_ids = {record["task_id"] for record in _STATE["run"].values()}
        assert len(task_ids) == 2
        assert {record["override_cwd"] for record in _STATE["run"].values()} == {
            str(wd_a),
            str(wd_b),
        }


# ---------------------------------------------------------------------------
# no_agent path must not mutate the process cwd (races under concurrency)
# ---------------------------------------------------------------------------

class TestNoAgentWorkdirNoChdir:
    def test_no_agent_workdir_job_does_not_chdir_process(
        self, tmp_path, monkeypatch, fake_agent_state
    ):
        import cron.scheduler as sched

        scripts_dir = _get_scripts_dir()
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script = scripts_dir / "probe_workdir.sh"
        script.write_text("#!/usr/bin/env bash\npwd\n")
        script.chmod(0o755)

        chdir_calls: list = []
        real_chdir = os.chdir
        monkeypatch.setattr(os, "chdir", lambda p: chdir_calls.append(p) or real_chdir(p))

        cwd_before = os.getcwd()
        job = {
            "id": "na1",
            "name": "no-agent-wd",
            "no_agent": True,
            "script": "probe_workdir.sh",
            "workdir": str(tmp_path),
            "schedule_display": "manual",
        }

        ok, _doc, resp, _err = sched.run_job(job)
        assert ok is True
        assert resp == str(tmp_path)

        assert os.getcwd() == cwd_before
        assert chdir_calls == [], (
            f"no_agent run_job mutated the process cwd via os.chdir: {chdir_calls}"
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _only_run(state: dict) -> dict:
    runs = state.get("run", {})
    assert len(runs) == 1, f"expected exactly one run, got {list(runs)}"
    return next(iter(runs.values()))


def _find_rec(runs: dict, workdir) -> dict:
    for rec in runs.values():
        if rec["override_cwd"] == str(workdir) or rec["session_cwd_var"] == str(workdir):
            return rec
    raise AssertionError(f"no run recorded for workdir {workdir}: {runs}")


def _get_scripts_dir():
    from pathlib import Path
    import cron.scheduler as sched
    return sched._get_hermes_home() / "scripts"
