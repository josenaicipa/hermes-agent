"""Liveness for buffered external-process auxiliary providers.

A CLI-backed compression provider emits nothing until it is done (measured:
the Kimi Code CLI's first stdout event lands at 106.7s of a 107.4s
synthesis). The caller-side token-inactivity watchdog (gateway session
hygiene, ~30s) therefore cancels a summary that is perfectly healthy.

``external_process_liveness`` fixes exactly that, and nothing more:

* it publishes progress only while a *verified* child process is alive;
* it stops the instant the process exits — never claims liveness afterwards;
* it does not touch total-time bounds (the 600s ceiling still fires);
* streaming providers are untouched.

Both buffered adapters are wired: ``kimi-code-cli`` drives it directly from
its ``Popen`` loop, and ``google-gemini-cli``/agy goes through
``run_text_capture`` — which stays byte-for-byte ``subprocess.run`` when no
progress hook is installed.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from agent import aux_process_liveness as liveness
from agent.auxiliary_client import (
    _create_with_progress,
    aux_external_process_liveness,
    aux_progress_hook,
)
from agent.conversation_compression import CompressionCommitFence


class _FakeProc:
    """Minimal Popen-shaped stand-in with controllable liveness."""

    def __init__(self):
        self._done = threading.Event()

    def poll(self):
        return None if not self._done.is_set() else 0

    def finish(self):
        self._done.set()


class TestSharedModuleWiring:
    """The helper lives in a tiny stdlib-only module (no import cycle)."""

    def test_auxiliary_client_reexports_the_shared_objects(self):
        from agent import auxiliary_client as ac

        assert ac.aux_external_process_liveness is liveness.external_process_liveness
        assert ac.aux_progress_hook is liveness.progress_hook
        assert ac._notify_aux_progress is liveness.notify_progress
        assert ac._aux_progress_active is liveness.progress_active
        assert ac._aux_progress is liveness._progress

    def test_shared_module_imports_only_stdlib(self):
        src = __import__("inspect").getsource(liveness)
        assert "import agent" not in src
        assert "from agent" not in src

    def test_both_buffered_adapters_use_it(self):
        import inspect

        from agent import agy_cli_client, kimi_code_cli_client

        assert "run_text_capture" in inspect.getsource(agy_cli_client)
        assert "external_process_liveness" in inspect.getsource(
            kimi_code_cli_client
        )


class TestLivenessTicking:
    def test_ticks_while_the_child_is_alive(self):
        ticks = []
        proc = _FakeProc()
        with aux_progress_hook(lambda: ticks.append(time.monotonic())):
            with aux_external_process_liveness(
                lambda: proc.poll() is None, interval=0.05
            ):
                time.sleep(0.35)
                proc.finish()
        assert len(ticks) >= 2, "buffered provider must publish liveness"

    def test_stops_immediately_when_the_child_exits(self):
        ticks = []
        proc = _FakeProc()
        with aux_progress_hook(lambda: ticks.append(1)):
            with aux_external_process_liveness(
                lambda: proc.poll() is None, interval=0.05
            ):
                time.sleep(0.2)
                proc.finish()
                after_exit = len(ticks)
                time.sleep(0.4)
        assert len(ticks) == after_exit, "must not claim liveness after exit"

    def test_no_ticks_after_context_exit(self):
        ticks = []
        proc = _FakeProc()  # deliberately never finishes
        with aux_progress_hook(lambda: ticks.append(1)):
            with aux_external_process_liveness(
                lambda: proc.poll() is None, interval=0.05
            ):
                time.sleep(0.2)
            settled = len(ticks)
            time.sleep(0.3)
        assert len(ticks) == settled

    def test_dead_child_from_the_start_never_ticks(self):
        ticks = []
        with aux_progress_hook(lambda: ticks.append(1)):
            with aux_external_process_liveness(lambda: False, interval=0.05):
                time.sleep(0.2)
        assert ticks == []

    def test_noop_without_a_progress_hook(self):
        proc = _FakeProc()
        with aux_external_process_liveness(lambda: proc.poll() is None,
                                           interval=0.05):
            time.sleep(0.1)

    def test_predicate_exception_stops_ticking(self):
        ticks = []

        def _boom():
            raise RuntimeError("cannot poll")

        with aux_progress_hook(lambda: ticks.append(1)):
            with aux_external_process_liveness(_boom, interval=0.05):
                time.sleep(0.25)
        assert ticks == []

    def test_hook_exception_is_swallowed(self):
        proc = _FakeProc()

        def _boom():
            raise RuntimeError("hook blew up")

        with aux_progress_hook(_boom):
            with aux_external_process_liveness(
                lambda: proc.poll() is None, interval=0.05
            ):
                time.sleep(0.15)
                proc.finish()

    def test_thread_is_joined_on_exit(self):
        before = {t.name for t in threading.enumerate()}
        proc = _FakeProc()
        with aux_progress_hook(lambda: None):
            with aux_external_process_liveness(
                lambda: proc.poll() is None, interval=0.05, label="unit"
            ):
                time.sleep(0.1)
        time.sleep(0.2)
        after = {t.name for t in threading.enumerate()}
        assert not [n for n in after - before if "hermes-aux-liveness" in n]


# ── run_text_capture: the agy path ───────────────────────────────────────

_SLEEPER = [sys.executable, "-c", "import time,sys; time.sleep(1.2); print('done')"]
_QUICK = [sys.executable, "-c", "print('quick')"]


class TestRunTextCapture:
    def test_without_hook_it_is_plain_subprocess_run(self, monkeypatch):
        """Byte-for-byte the historical path, including monkeypatchability."""
        seen = {}

        def _fake_run(argv, **kwargs):
            seen["argv"] = argv
            seen["kwargs"] = kwargs
            return subprocess.CompletedProcess(argv, 0, "patched", "")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        result = liveness.run_text_capture(
            ["anything"], timeout=5, env=None, cwd=None, label="agy"
        )
        assert result.stdout == "patched"
        assert seen["kwargs"]["capture_output"] is True
        assert seen["kwargs"]["text"] is True
        assert seen["kwargs"]["shell"] is False
        assert seen["kwargs"]["check"] is False
        assert seen["kwargs"]["timeout"] == 5

    def test_without_hook_no_liveness_thread_is_started(self):
        before = {t.name for t in threading.enumerate()}
        liveness.run_text_capture(
            _QUICK, timeout=30, env=None, cwd=None, label="agy"
        )
        after = {t.name for t in threading.enumerate()}
        assert not [n for n in after - before if "hermes-aux-liveness" in n]

    def test_with_hook_it_publishes_liveness_while_the_child_runs(self):
        ticks = []
        with aux_progress_hook(lambda: ticks.append(1)):
            result = liveness.run_text_capture(
                _SLEEPER, timeout=30, env=None, cwd=None,
                label="agy", interval=0.1,
            )
        assert result.returncode == 0
        assert "done" in result.stdout
        assert len(ticks) >= 2, "agy must not stay silent while its child runs"

    def test_liveness_stops_once_the_child_exits(self):
        ticks = []
        with aux_progress_hook(lambda: ticks.append(1)):
            liveness.run_text_capture(
                _QUICK, timeout=30, env=None, cwd=None,
                label="agy", interval=0.1,
            )
            settled = len(ticks)
            time.sleep(0.4)
        assert len(ticks) == settled

    def test_timeout_still_raises_and_reaps(self):
        with aux_progress_hook(lambda: None):
            with pytest.raises(subprocess.TimeoutExpired):
                liveness.run_text_capture(
                    _SLEEPER, timeout=0.2, env=None, cwd=None,
                    label="agy", interval=0.05,
                )

    def test_output_is_captured_as_text(self):
        with aux_progress_hook(lambda: None):
            result = liveness.run_text_capture(
                _QUICK, timeout=30, env=None, cwd=None, label="agy",
            )
        assert result.stdout.strip() == "quick"
        assert isinstance(result.stdout, str)


class TestAgyAdapterHeartbeat:
    """agy must not remain silent: it is buffered like Kimi."""

    def _client(self):
        from agent.agy_cli_client import AgyCLIClient

        return AgyCLIClient(command=sys.executable)

    def test_agy_publishes_liveness_while_its_child_is_alive(self, tmp_path):
        from agent import agy_cli_client

        ticks = []
        log = tmp_path / "agy.log"
        log.write_text("")
        argv = list(_SLEEPER)

        with aux_progress_hook(lambda: ticks.append(1)):
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(
                    agy_cli_client, "run_text_capture",
                    lambda a, **kw: liveness.run_text_capture(
                        a, **{**kw, "interval": 0.1}
                    ),
                )
                out = self._client()._run_process(
                    argv, timeout_seconds=30, env=None, cwd=None,
                    log_path=str(log),
                )
        assert out == "done"
        assert len(ticks) >= 2

    def test_agy_stops_ticking_after_the_child_exits(self, tmp_path):
        from agent import agy_cli_client

        ticks = []
        log = tmp_path / "agy.log"
        log.write_text("")

        with aux_progress_hook(lambda: ticks.append(1)):
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(
                    agy_cli_client, "run_text_capture",
                    lambda a, **kw: liveness.run_text_capture(
                        a, **{**kw, "interval": 0.1}
                    ),
                )
                self._client()._run_process(
                    list(_QUICK), timeout_seconds=30, env=None, cwd=None,
                    log_path=str(log),
                )
            settled = len(ticks)
            time.sleep(0.4)
        assert len(ticks) == settled

    def test_agy_without_hook_keeps_the_historical_run_path(self, tmp_path,
                                                            monkeypatch):
        """No progress hook -> plain subprocess.run, as tests have always seen."""
        from agent.agy_cli_client import AgyCLIClient

        log = tmp_path / "agy.log"
        log.write_text("")
        calls = []

        def _fake_run(argv, **kwargs):
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, "legacy output", "")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        out = AgyCLIClient(command="/usr/bin/agy")._run_process(
            ["/usr/bin/agy", "--print", "x"], timeout_seconds=12,
            env={"HOME": "/tmp"}, cwd=str(tmp_path), log_path=str(log),
        )
        assert out == "legacy output"
        assert len(calls) == 1
        assert calls[0][1]["timeout"] == 12
        assert calls[0][1]["shell"] is False


class TestCommitFenceIntegration:
    def test_liveness_extends_the_inactivity_clock(self):
        """The exact wiring gateway hygiene polls."""
        fence = CompressionCommitFence()
        proc = _FakeProc()
        time.sleep(0.15)
        assert fence.seconds_since_progress() >= 0.1

        with aux_progress_hook(fence.touch_progress):
            with aux_external_process_liveness(
                lambda: proc.poll() is None, interval=0.05
            ):
                time.sleep(0.25)
                idle_during = fence.seconds_since_progress()
                proc.finish()
        assert idle_during < 0.15, "buffered child must look alive to the fence"

    def test_total_ceiling_is_not_extended_by_liveness(self):
        """Liveness bounds inactivity only — never total wall clock.

        Mirrors gateway/run.py's hygiene loop: it continues *only* while BOTH
        the idle window and the total ceiling hold. A hung-but-alive child
        keeps ticking forever, so the ceiling is what must stop it.
        """
        fence = CompressionCommitFence()
        proc = _FakeProc()  # alive forever == a hung CLI
        idle_window = 0.2
        total_ceiling = 0.6

        with aux_progress_hook(fence.touch_progress):
            with aux_external_process_liveness(
                lambda: proc.poll() is None, interval=0.05
            ):
                started = time.monotonic()
                extensions = 0
                while True:
                    time.sleep(idle_window)
                    waited = time.monotonic() - started
                    if (
                        fence.seconds_since_progress() < idle_window
                        and waited < total_ceiling
                    ):
                        extensions += 1
                        continue
                    break
                proc.finish()

        # Liveness kept the idle watchdog quiet ...
        assert extensions >= 1
        # ... but the total ceiling still terminated the wait.
        assert time.monotonic() - started >= total_ceiling


class TestStreamingProvidersUnaffected:
    """The streaming inactivity policy must not be weakened globally."""

    def test_streaming_path_is_unchanged(self):
        chunks = [
            SimpleNamespace(
                id="c", model="m",
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="hi", reasoning=None,
                                          reasoning_content=None, tool_calls=None),
                    finish_reason=None)],
                usage=None,
            ),
            SimpleNamespace(
                id="c", model="m",
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=None, reasoning=None,
                                          reasoning_content=None, tool_calls=None),
                    finish_reason="stop")],
                usage=None,
            ),
        ]
        seen = []

        class _Client:
            def __init__(self):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs):
                seen.append(kwargs)
                return iter(chunks) if kwargs.get("stream") else None

        ticks = []
        with aux_progress_hook(lambda: ticks.append(1)):
            resp = _create_with_progress(_Client(), {"model": "m", "messages": []})

        assert resp.choices[0].message.content == "hi"
        assert seen[0].get("stream") is True
        # Ticks come from arriving chunks, not from any process-liveness path.
        assert len(ticks) >= 2

    def test_no_progress_hook_means_plain_non_streaming_call(self):
        seen = []
        complete = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="plain"), finish_reason="stop")],
            usage=None, model="m",
        )

        class _Client:
            def __init__(self):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs):
                seen.append(kwargs)
                return complete

        resp = _create_with_progress(_Client(), {"model": "m", "messages": []})
        assert resp is complete
        assert "stream" not in seen[0]
