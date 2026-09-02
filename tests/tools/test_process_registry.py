"""Tests for tools/process_registry.py — ProcessRegistry query methods, pruning, checkpoint."""

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import pytest
from unittest.mock import MagicMock, patch

from tools.environments.local_env_policy import _HERMES_PROVIDER_ENV_FORCE_PREFIX
from tools.process_registry import (
    ProcessNotificationConfig,
    ProcessCheckpointRecoveryError,
    ProcessRegistry,
    ProcessSession,
    FINISHED_TTL_SECONDS,
    MAX_PROCESSES,
)


@pytest.fixture()
def registry():
    """Create a fresh ProcessRegistry."""
    return ProcessRegistry()


@pytest.fixture(autouse=True)
def _reset_systemd_scope_cache():
    """Reset the cached ``systemd-run --user --scope`` availability flag
    before each test so a probe run on a real systemd host (where
    ``INVOCATION_ID`` is set) doesn't leak into tests that mock
    ``subprocess.Popen``. Tests that exercise the probe directly reset the
    cache themselves."""
    import tools.process_registry as _pr

    original = _pr._SYSTEMD_SCOPE_AVAILABLE
    _pr._SYSTEMD_SCOPE_AVAILABLE = False
    yield
    _pr._SYSTEMD_SCOPE_AVAILABLE = original


def _make_session(
    sid="proc_test123",
    command="echo hello",
    task_id="t1",
    exited=False,
    exit_code=None,
    output="",
    started_at=None,
) -> ProcessSession:
    """Helper to create a ProcessSession for testing."""
    s = ProcessSession(
        id=sid,
        command=command,
        task_id=task_id,
        started_at=started_at or time.time(),
        exited=exited,
        exit_code=exit_code,
        output_buffer=output,
    )
    return s


def _spawn_python_sleep(seconds: float) -> subprocess.Popen:
    """Spawn a portable short-lived Python sleep process."""
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
    )


def test_kill_started_since_preserves_preexisting_and_foreign_processes(registry):
    old = _make_session(sid="proc_old", task_id="session-a")
    finished = _make_session(
        sid="proc_finished", task_id="session-a", exited=True, exit_code=0
    )
    registry._running[old.id] = old
    registry._finished[finished.id] = finished
    baseline = registry.snapshot_running_ids("session-a")

    new = _make_session(sid="proc_new", task_id="session-a")
    foreign = _make_session(sid="proc_foreign", task_id="session-b")
    registry._running[new.id] = new
    registry._running[foreign.id] = foreign

    calls = []

    def fake_kill(session_id, **kwargs):
        calls.append((session_id, kwargs))
        return {"status": "killed"}

    registry.kill_process = fake_kill

    assert baseline == frozenset({"proc_old"})
    assert registry.kill_started_since(
        "session-a", baseline, source="gateway_turn_timeout"
    ) == 1
    assert calls == [
        (
            "proc_new",
            {
                "source": "gateway_turn_timeout",
                "consume_output": True,
            },
        )
    ]


def test_kill_started_since_suppresses_missing_control_fallback(
    registry, tmp_path, monkeypatch
):
    """Abandoned-turn cleanup must not revive the work it intentionally kills."""
    import tools.process_registry as pr_module

    checkpoint = tmp_path / "processes.json"
    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
    session = _make_session(sid="proc_abandoned", task_id="turn-a")
    session.pid = 77
    session.pid_scope = "sandbox"
    session.env_ref = MagicMock()
    session.watch_patterns = ["FABLE_WAKE"]
    session.notify_on_failure = True
    session.watcher_platform = "telegram"
    session.watcher_chat_id = "123"
    registry._running[session.id] = session
    assert registry._write_checkpoint() is True

    assert registry.kill_started_since(
        "turn-a", frozenset(), source="gateway_turn_timeout"
    ) == 1

    session.env_ref.execute.assert_called_once()
    assert registry.is_completion_consumed(session.id)
    assert registry.completion_queue.empty()
    assert json.loads(checkpoint.read_text(encoding="utf-8")) == []
    assert not (tmp_path / "process_notifications.json").exists()


def test_kill_started_since_revokes_already_captured_control_wake(
    registry, tmp_path, monkeypatch
):
    """A marker captured just before turn cleanup is not delivered afterward."""
    import tools.process_registry as pr_module

    checkpoint = tmp_path / "processes.json"
    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
    session = _make_session(sid="proc_abandoned_marker", task_id="turn-a")
    session.pid = 88
    session.pid_scope = "sandbox"
    session.env_ref = MagicMock()
    session.watch_patterns = ["FABLE_WAKE"]
    session.notify_on_failure = True
    session.watcher_platform = "telegram"
    session.watcher_chat_id = "123"
    registry._running[session.id] = session
    assert registry._write_checkpoint() is True
    registry._check_watch_patterns(session, "FABLE_WAKE reason=late\n")
    assert registry.completion_queue.qsize() == 1
    outbox = tmp_path / "process_notifications.json"
    assert len(json.loads(outbox.read_text(encoding="utf-8"))) == 1

    assert registry.kill_started_since(
        "turn-a", frozenset(), source="gateway_turn_timeout"
    ) == 1

    assert registry.drain_notifications() == []
    assert registry.completion_queue.empty()
    assert json.loads(outbox.read_text(encoding="utf-8")) == []
    assert json.loads(checkpoint.read_text(encoding="utf-8")) == []


def test_wait_reconcile_marks_consumed_before_missing_control_fallback(
    registry, tmp_path, monkeypatch
):
    """Consumer-driven exit reconciliation cannot enqueue a stale wake."""
    import tools.process_registry as pr_module

    checkpoint = tmp_path / "processes.json"
    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
    session = _make_session(sid="proc_wait_reconcile", task_id="turn-a")
    session.process = MagicMock()
    session.process.poll.return_value = 0
    session.process.stdout = None
    session.pid = 4242
    session.watch_patterns = ["FABLE_WAKE"]
    session.notify_on_failure = True
    session.watcher_platform = "telegram"
    session.watcher_chat_id = "123"
    registry._running[session.id] = session
    assert registry._write_checkpoint() is True

    result = registry.wait(session.id, timeout=1)

    assert result["status"] == "exited"
    assert registry.is_completion_consumed(session.id)
    assert registry.completion_queue.empty()
    assert json.loads(checkpoint.read_text(encoding="utf-8")) == []
    assert not (tmp_path / "process_notifications.json").exists()


@pytest.mark.parametrize("consumer", ["wait", "read_log", "kill"])
def test_detached_terminal_consumers_finalize_without_control_wake(
    registry, tmp_path, monkeypatch, consumer
):
    """Detached refresh folds inline consumption into finalization atomically."""
    import tools.process_registry as pr_module

    checkpoint = tmp_path / "processes.json"
    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
    session = _make_session(sid=f"proc_detached_{consumer}", task_id="turn-a")
    session.pid = 999999999
    session.pid_scope = "host"
    session.host_start_time = 123
    session.detached = True
    session.watch_patterns = ["FABLE_WAKE"]
    session.notify_on_failure = True
    session.watcher_platform = "telegram"
    session.watcher_chat_id = "123"
    registry._running[session.id] = session
    monkeypatch.setattr(registry, "_host_pid_is_ours", lambda *_args: False)
    assert registry._write_checkpoint() is True

    if consumer == "wait":
        result = registry.wait(session.id, timeout=1)
    elif consumer == "read_log":
        result = registry.read_log(session.id, offset=0)
    else:
        result = registry.kill_process(session.id, consume_output=True)

    assert result["status"] in {"exited", "already_exited"}
    assert registry.is_completion_consumed(session.id)
    assert registry.completion_queue.empty()
    assert json.loads(checkpoint.read_text(encoding="utf-8")) == []
    assert not (tmp_path / "process_notifications.json").exists()


def test_kill_all_backward_compat_and_exclude_ids(registry):
    """kill_all keeps its historical default behavior (kill everything for
    the task, consume_output=False, source='kill_all') and honors the new
    exclude_ids kwarg that kill_started_since delegates through (#76188)."""
    a = _make_session(sid="proc_a", task_id="session-a")
    b = _make_session(sid="proc_b", task_id="session-a")
    registry._running[a.id] = a
    registry._running[b.id] = b

    calls = []

    def fake_kill(session_id, **kwargs):
        calls.append((session_id, kwargs))
        return {"status": "killed"}

    registry.kill_process = fake_kill

    assert registry.kill_all("session-a", exclude_ids=frozenset({"proc_a"})) == 1
    assert calls == [
        ("proc_b", {"source": "kill_all", "consume_output": False})
    ]

    calls.clear()
    assert registry.kill_all("session-a") == 2
    assert sorted(c[0] for c in calls) == ["proc_a", "proc_b"]


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """Poll a predicate until it returns truthy or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.mark.windows_only
def test_write_stdin_uses_str_for_windows_pty(registry):
    """pywinpty expects str input; bytes raises a PyString conversion error.

    Windows-only: the str-vs-bytes choice IS the ``_IS_WINDOWS`` branch, and
    the real pty handle it must satisfy (pywinpty) does not exist elsewhere.
    """
    written = []

    class _FakePty:
        def write(self, value):
            written.append(value)

    session = _make_session(sid="pty-win")
    session._pty = _FakePty()
    registry._running[session.id] = session

    result = registry.write_stdin(session.id, "hello\n")

    assert result == {"status": "ok", "bytes_written": 6}
    assert written == ["hello\n"]
    assert isinstance(written[0], str)


@pytest.mark.linux_only
def test_write_stdin_uses_bytes_for_posix_pty(registry):
    """The POSIX counterpart: ptyprocess expects bytes, not str."""
    written = []

    class _FakePty:
        def write(self, value):
            written.append(value)

    session = _make_session(sid="pty-posix")
    session._pty = _FakePty()
    registry._running[session.id] = session

    result = registry.write_stdin(session.id, "hello\n")

    assert result == {"status": "ok", "bytes_written": 6}
    assert written == [b"hello\n"]


@pytest.mark.windows_only
def test_submit_stdin_uses_crlf_for_windows_pty(registry):
    """Enter on a Windows PTY is a carriage return, not a bare LF.

    ConPTY cooked input only ends a line on ``\\r``; a bare ``\\n`` through
    pywinpty is never delivered to a blocking line read (Python readline,
    Go bufio.Scanner — the exact hang seen live with ``gh auth login``'s
    "Press Enter to open the browser" prompt). submit_stdin must append
    ``\\r\\n`` for Windows PTY sessions.
    """
    written = []

    class _FakePty:
        def write(self, value):
            written.append(value)

    session = _make_session(sid="pty-win-submit")
    session._pty = _FakePty()
    registry._running[session.id] = session

    result = registry.submit_stdin(session.id, "Y")

    assert result["status"] == "ok"
    assert written == ["Y\r\n"]


@pytest.mark.windows_only
def test_submit_stdin_keeps_lf_for_windows_pipe(registry):
    """Non-PTY (Popen pipe) sessions keep the plain LF on Windows."""
    session = _make_session(sid="pipe-win-submit")
    fake_stdin = MagicMock()
    session.process = MagicMock()
    session.process.stdin = fake_stdin
    registry._running[session.id] = session

    result = registry.submit_stdin(session.id, "Y")

    assert result["status"] == "ok"
    fake_stdin.write.assert_called_once_with("Y\n")


# =========================================================================
# Get / Poll
# =========================================================================

class TestGetAndPoll:
    def test_poll_running(self, registry):
        s = _make_session(output="some output here")
        registry._running[s.id] = s
        result = registry.poll(s.id)
        assert result["status"] == "running"
        assert "some output" in result["output_preview"]
        assert result["command"] == "echo hello"

    def test_poll_exited(self, registry):
        s = _make_session(exited=True, exit_code=0, output="done")
        registry._finished[s.id] = s
        result = registry.poll(s.id)
        assert result["status"] == "exited"
        assert result["exit_code"] == 0


def test_request_close_terminal_invokes_sink_without_killing(registry):
    """With a sink wired, close routes (session, process_id) to the UI and leaves
    the process running — close is a view drop, not a kill."""
    s = _make_session(sid="proc_close_live")
    registry._running[s.id] = s
    calls = []
    registry.on_close = lambda session, pid: calls.append((session, pid))

    result = registry.request_close_terminal(s.id)

    assert result["status"] == "ok"
    assert result["closed"] == "proc_close_live"
    assert calls == [(s, "proc_close_live")]
    # Still tracked as running — closing the tab must not reap the process.
    assert s.id in registry._running


def test_reader_loop_streams_incremental_chunks_from_read1(registry, monkeypatch):
    """Local reader must emit live chunks, not one EOF burst.

    Regression for desktop agent terminals: ``stdout.read(4096)`` can buffer
    until process exit for small periodic output. ``buffer.read1(4096)`` should
    surface each chunk as it arrives.
    """

    class _FakeBuffer:
        def __init__(self, chunks):
            self._chunks = list(chunks)

        def read1(self, _n):
            if self._chunks:
                return self._chunks.pop(0)
            return b""

    class _FakeStdout:
        def __init__(self, chunks):
            self.buffer = _FakeBuffer(chunks)

    class _FakeProcess:
        def __init__(self, chunks):
            self.stdout = _FakeStdout(chunks)
            self.returncode = 0

        def wait(self, timeout=None):
            return 0

    session = _make_session(sid="proc_reader_live")
    session.process = _FakeProcess([b"tick 1\n", b"tick 2\n", b"tick 3\n", b""])
    emitted = []
    moved = []

    monkeypatch.setattr(registry, "_check_watch_patterns", lambda _s, _c: None)
    monkeypatch.setattr(registry, "_emit_output", lambda _s, chunk: emitted.append(chunk))
    monkeypatch.setattr(registry, "_move_to_finished", lambda _s: moved.append(_s.id))

    registry._reader_loop(session)

    assert emitted == ["tick 1\n", "tick 2\n", "tick 3\n"]
    assert session.output_buffer == "tick 1\ntick 2\ntick 3\n"
    assert session.exited is True
    assert session.exit_code == 0
    assert moved == ["proc_reader_live"]


# =========================================================================
# Incremental UTF-8 decoding across chunk boundaries
# (ported from openclaw/openclaw#112325)
# =========================================================================


class _FakeChunkBuffer:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read1(self, _n):
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _FakeChunkStdout:
    def __init__(self, chunks):
        self.buffer = _FakeChunkBuffer(chunks)


class _FakeChunkProcess:
    def __init__(self, chunks):
        self.stdout = _FakeChunkStdout(chunks)
        self.returncode = 0

    def wait(self, timeout=None):
        return 0


def _run_reader(registry, monkeypatch, chunks, sid="proc_utf8"):
    session = _make_session(sid=sid)
    session.process = _FakeChunkProcess(chunks)
    monkeypatch.setattr(registry, "_check_watch_patterns", lambda _s, _c: None)
    monkeypatch.setattr(registry, "_emit_output", lambda _s, _c: None)
    monkeypatch.setattr(registry, "_move_to_finished", lambda _s: None)
    registry._reader_loop(session)
    return session


def test_reader_loop_reassembles_multibyte_char_split_across_chunks(registry, monkeypatch):
    """A UTF-8 char split across two read1() chunks must not become U+FFFD.

    Before the incremental decoder, each chunk was decoded statelessly with
    ``errors="replace"``, so ``é`` (0xC3 0xA9) straddling a 4096-byte read
    boundary decoded as two replacement characters.
    """
    session = _run_reader(registry, monkeypatch, [b"caf\xc3", b"\xa9 ok\n"])
    assert session.output_buffer == "café ok\n"
    assert "\ufffd" not in session.output_buffer


def test_reader_loop_reassembles_four_byte_char_split_three_ways(registry, monkeypatch):
    """A 4-byte emoji fragmented across three reads reassembles cleanly."""
    session = _run_reader(registry, monkeypatch, [b"\xf0", b"\x9f\x92", b"\xa9\n"])
    assert session.output_buffer == "\U0001f4a9\n"


def test_reader_loop_flushes_truncated_multibyte_tail_at_eof(registry, monkeypatch):
    """A sequence truncated by process exit flushes as a single U+FFFD."""
    session = _run_reader(registry, monkeypatch, [b"ok \xe2\x82"])
    assert session.output_buffer == "ok \ufffd"


def test_reader_loop_still_replaces_genuinely_invalid_bytes(registry, monkeypatch):
    """Truly invalid bytes keep the errors="replace" behavior."""
    session = _run_reader(registry, monkeypatch, [b"ok\xffdone\n"])
    assert session.output_buffer == "ok\ufffddone\n"


def test_pty_reader_loop_reassembles_multibyte_char_split_across_chunks(registry, monkeypatch):
    """The PTY reader gets the same incremental-decode treatment."""

    class _FakePty:
        def __init__(self, chunks):
            self._chunks = list(chunks)
            self.exitstatus = 0

        def isalive(self):
            return bool(self._chunks)

        def read(self, _n):
            if self._chunks:
                return self._chunks.pop(0)
            raise EOFError

        def wait(self):
            return 0

    session = _make_session(sid="proc_pty_utf8")
    session._pty = _FakePty([b"caf\xc3", b"\xa9\n"])
    monkeypatch.setattr(registry, "_check_watch_patterns", lambda _s, _c: None)
    monkeypatch.setattr(registry, "_emit_output", lambda _s, _c: None)
    monkeypatch.setattr(registry, "_move_to_finished", lambda _s: None)

    registry._pty_reader_loop(session)

    assert session.output_buffer == "café\n"
    assert "\ufffd" not in session.output_buffer


# =========================================================================
# Orphaned-pipe reconciliation (issue #17327)
# =========================================================================

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: uses setsid/fcntl")
class TestOrphanedPipeReconciliation:
    """Regression tests for issue #17327.

    `hermes update` in Feishu spawned a background subprocess that restarted
    the gateway; the direct child exited quickly but a descendant daemon
    held the stdout pipe open. `_reader_loop.finally` never ran, so
    `session.exited` stayed False and the agent polled 74 times over 7
    minutes, all returning `status: running`.

    The fix is `_reconcile_local_exit()`: poll() and wait() now check the
    direct `Popen.poll()` before trusting `session.exited`.
    """

    def test_reconcile_flips_exited_when_direct_child_done(self, registry):
        """Direct child exited but reader thread is blocked on orphaned pipe."""
        # Simulate the orphaned-pipe scenario: direct child exited, but a
        # descendant holds stdout open so the reader never sees EOF.
        # Approach: spawn `sh -c 'sleep 10 &'` with setsid — sh forks the
        # sleep into a new session group, exits immediately, but sleep
        # inherits the stdout pipe and keeps it open.
        proc = subprocess.Popen(
            ["sh", "-c", "exec 1>&2; ( sleep 30 ) & disown; exit 0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        s = _make_session(sid="proc_orphan_test")
        s.process = proc
        s.pid = proc.pid
        registry._running[s.id] = s

        # Wait for the direct child to exit. We don't start a reader thread,
        # so session.exited stays False (mimicking the stuck-reader state).
        assert _wait_until(lambda: proc.poll() is not None, timeout=5.0), (
            "Direct child should exit quickly (sh exits, sleep descendant "
            "holds the pipe open)"
        )

        # Before the fix: poll would return "running" forever.
        # After the fix: poll reconciles against proc.poll() and flips.
        assert s.exited is False  # Precondition: reader hasn't updated it.
        result = registry.poll(s.id)
        assert result["status"] == "exited", (
            f"Expected reconciled 'exited' status; got {result!r}. "
            "This is issue #17327 — reader is blocked on orphaned pipe."
        )
        assert result["exit_code"] == 0
        assert s.exited is True
        assert s.id in registry._finished
        assert s.id not in registry._running

        # Clean up the orphaned descendant.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def test_wait_returns_when_reader_blocked(self, registry):
        """wait() must also reconcile — not just poll()."""
        proc = subprocess.Popen(
            ["sh", "-c", "( sleep 30 ) & disown; exit 0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        s = _make_session(sid="proc_wait_orphan")
        s.process = proc
        s.pid = proc.pid
        registry._running[s.id] = s

        assert _wait_until(lambda: proc.poll() is not None, timeout=5.0)

        start = time.monotonic()
        result = registry.wait(s.id, timeout=10)
        elapsed = time.monotonic() - start

        assert result["status"] == "exited", result
        assert elapsed < 5.0, (
            f"wait() should return ~immediately via reconcile; took {elapsed:.1f}s"
        )

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def test_wait_wakes_when_session_moves_to_finished(self, registry):
        """wait() should not sleep for the old 1s polling tick after exit."""
        s = _make_session(sid="proc_wait_event", output="done")
        registry._running[s.id] = s

        def finish_later():
            time.sleep(0.05)
            s.exited = True
            s.exit_code = 0
            with patch.object(registry, "_write_checkpoint"):
                registry._move_to_finished(s)

        t = threading.Thread(target=finish_later)
        t.start()
        start = time.monotonic()
        try:
            result = registry.wait(s.id, timeout=5)
        finally:
            t.join(timeout=1)
        elapsed = time.monotonic() - start

        assert result["status"] == "exited", result
        assert result["exit_code"] == 0
        assert elapsed < 0.9  # must stay under the old 1s poll tick being regression-tested, f"wait() should wake on completion; took {elapsed:.3f}s"


# =========================================================================
# Read log
# =========================================================================

class TestReadLog:
    def test_read_full_log(self, registry):
        lines = "\n".join([f"line {i}" for i in range(50)])
        s = _make_session(output=lines)
        registry._running[s.id] = s
        result = registry.read_log(s.id)
        assert result["total_lines"] == 50

    def test_read_with_offset(self, registry):
        lines = "\n".join([f"line {i}" for i in range(100)])
        s = _make_session(output=lines)
        registry._running[s.id] = s
        result = registry.read_log(s.id, offset=10, limit=5)
        assert "5 lines" in result["showing"]


# =========================================================================
# Stdin helpers
# =========================================================================

class TestStdinHelpers:
    def test_close_stdin_pipe_mode(self, registry):
        proc = MagicMock()
        proc.stdin = MagicMock()
        s = _make_session()
        s.process = proc
        registry._running[s.id] = s

        result = registry.close_stdin(s.id)

        proc.stdin.close.assert_called_once()
        assert result["status"] == "ok"

    def test_close_stdin_allows_eof_driven_process_to_finish(self, registry, tmp_path):
        """PTY mode: writing data + sending EOF lets an EOF-driven child finish.

        Background non-PTY mode used to expose subprocess stdin via a pipe,
        but PR #214b95392 detached non-PTY stdin to DEVNULL to fix keyboard
        lockout (#17959). For interactive stdin → PTY mode is now the only
        supported path.
        """
        session = registry.spawn_local(
            'python3 -c "import sys; print(sys.stdin.read().strip())"',
            cwd=str(tmp_path),
            use_pty=True,
        )

        try:
            # Wait for the PTY child to be up rather than sleeping blindly.
            assert _wait_until(
                lambda: registry.poll(session.id)["status"] == "running",
                timeout=5.0,
                interval=0.02,
            ), "PTY session never reached running"
            assert registry.submit_stdin(session.id, "hello")["status"] == "ok"
            assert registry.close_stdin(session.id)["status"] == "ok"

            deadline = time.time() + 5
            while time.time() < deadline:
                poll = registry.poll(session.id)
                if poll["status"] == "exited":
                    assert poll["exit_code"] == 0
                    assert "hello" in poll["output_preview"]
                    return
                time.sleep(0.02)

            pytest.fail("process did not exit after stdin was closed")
        finally:
            registry.kill_process(session.id)


# =========================================================================
# List sessions
# =========================================================================

class TestListSessions:
    def test_filter_by_task_id(self, registry):
        s1 = _make_session(sid="proc_1", task_id="t1")
        s2 = _make_session(sid="proc_2", task_id="t2")
        registry._running[s1.id] = s1
        registry._running[s2.id] = s2
        result = registry.list_sessions(task_id="t1")
        assert len(result) == 1
        assert result[0]["session_id"] == "proc_1"

    def test_session_key_surfaces_cross_task_processes(self, registry):
        """A bg process under the same gateway session but a DIFFERENT task is
        surfaced when session_key is passed, and flagged session_scoped (#29177).
        """
        # Current turn's task = "t_now"; forgotten preview server = "t_old"
        # but both share gateway session_key "gw1".
        own = _make_session(sid="proc_own", task_id="t_now")
        own.session_key = "gw1"
        forgotten = _make_session(sid="proc_forgotten", task_id="t_old")
        forgotten.session_key = "gw1"
        other = _make_session(sid="proc_other", task_id="t_x")
        other.session_key = "gw_other"
        registry._running[own.id] = own
        registry._running[forgotten.id] = forgotten
        registry._running[other.id] = other

        # Task-only (legacy) view sees just the current task's process.
        legacy = registry.list_sessions(task_id="t_now")
        assert {r["session_id"] for r in legacy} == {"proc_own"}

        # With session_key, the forgotten process under the same gateway
        # session is surfaced and flagged; the unrelated session is not.
        result = registry.list_sessions(task_id="t_now", session_key="gw1")
        by_id = {r["session_id"]: r for r in result}
        assert set(by_id) == {"proc_own", "proc_forgotten"}
        assert by_id["proc_forgotten"].get("session_scoped") is True
        assert "session_scoped" not in by_id["proc_own"]

# =========================================================================
# Active process queries
# =========================================================================

class TestActiveQueries:
    def test_has_active_processes(self, registry):
        s = _make_session(task_id="t1")
        registry._running[s.id] = s
        assert registry.has_active_processes("t1") is True
        assert registry.has_active_processes("t2") is False

    def test_has_active_for_session_with_max_age_stale(self, registry):
        """Stale process (older than max_active_age) is ignored."""
        s = _make_session(started_at=time.time() - 90000)  # 25 hours ago
        s.session_key = "gw_session_1"
        registry._running[s.id] = s
        assert registry.has_active_for_session("gw_session_1", max_active_age=86400) is False

# =========================================================================
# Pruning
# =========================================================================

class TestPruning:
    def test_prune_expired_finished(self, registry):
        old_session = _make_session(
            sid="proc_old",
            exited=True,
            started_at=time.time() - FINISHED_TTL_SECONDS - 100,
        )
        registry._finished[old_session.id] = old_session
        registry._prune_if_needed()
        assert "proc_old" not in registry._finished

    def test_prune_over_max_removes_oldest(self, registry):
        # Fill up to MAX_PROCESSES
        for i in range(MAX_PROCESSES):
            s = _make_session(
                sid=f"proc_{i}",
                exited=True,
                started_at=time.time() - i,  # older as i increases
            )
            registry._finished[s.id] = s

        # Add one more running to trigger prune
        s = _make_session(sid="proc_new")
        registry._running[s.id] = s
        registry._prune_if_needed()

        total = len(registry._running) + len(registry._finished)
        assert total <= MAX_PROCESSES

    def test_expiry_prune_keeps_consumed_fable_wake_suppressed(
        self, registry, tmp_path, monkeypatch
    ):
        """A queued control wake cannot revive after its process row expires."""
        import tools.process_registry as pr_module

        monkeypatch.setattr(
            pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json"
        )
        session = _make_session(
            sid="proc_consumed_expired",
            exited=True,
            exit_code=1,
            started_at=time.time() - FINISHED_TTL_SECONDS - 100,
        )
        session.watch_patterns = ["FABLE_WAKE"]
        session.notify_on_failure = True
        session.watcher_platform = "telegram"
        registry._finished[session.id] = session
        queued = registry._build_reliable_control_failure_event(session)
        registry.completion_queue.put(queued)

        registry._consume_completion_result(session)
        registry._prune_if_needed()

        assert session.id not in registry._finished
        assert session.id not in registry._completion_consumed
        assert registry.is_completion_consumed(session.id) is True
        assert registry.is_notification_consumed(
            registry.completion_queue.get_nowait()
        ) is True

    def test_capacity_prune_keeps_consumed_fable_wake_suppressed(
        self, registry, tmp_path, monkeypatch
    ):
        """The MAX_PROCESSES LRU cannot release a queued control fence."""
        import tools.process_registry as pr_module

        monkeypatch.setattr(
            pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json"
        )
        consumed = _make_session(
            sid="proc_consumed_oldest",
            exited=True,
            exit_code=1,
            started_at=time.time() - 100,
        )
        consumed.watch_patterns = ["FABLE_WAKE"]
        consumed.notify_on_failure = True
        consumed.watcher_platform = "telegram"
        registry._finished[consumed.id] = consumed
        for i in range(MAX_PROCESSES - 1):
            session = _make_session(
                sid=f"proc_capacity_{i}",
                exited=True,
                started_at=time.time() - i,
            )
            registry._finished[session.id] = session
        queued = registry._build_reliable_control_failure_event(consumed)
        registry.completion_queue.put(queued)

        registry._consume_completion_result(consumed)
        registry._prune_if_needed()

        assert consumed.id not in registry._finished
        assert consumed.id not in registry._completion_consumed
        assert registry.is_completion_consumed(consumed.id) is True
        assert registry.is_notification_consumed(
            registry.completion_queue.get_nowait()
        ) is True


class TestFinishedHandleRelease:
    """Finished sessions must release their Popen/PTY OS handles immediately.

    Regression for the "file descriptor limit" symptom: a finished-but-
    unpruned session previously kept its Popen stdout pipe (or PTY master)
    FD open until the finished-process TTL (FINISHED_TTL_SECONDS) elapsed.
    Under heavy background churn the gateway could exhaust its FD limit even
    though the registry never rejects spawns (it prunes oldest-finished at
    MAX_PROCESSES instead) — the symptom was a retained-handle leak, not a
    registry-cap rejection. poll()/wait()/read_log() serve from the buffered
    output_buffer, never from the pipe, so closing the handles at finish is
    lossless.
    """

    def test_move_to_finished_closes_popen_pipes(self, registry):
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(0.2)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        session = _make_session(sid="proc_handle_close", exited=False)
        session.process = proc
        registry._running[session.id] = session

        assert proc.stdout is not None
        assert not proc.stdout.closed

        # Simulate the reader loop finishing (process exits, EOF drained).
        proc.wait(timeout=5)
        session.exited = True
        session.exit_code = proc.returncode
        session.completion_reason = "exited"
        registry._move_to_finished(session)

        assert session.id in registry._finished
        assert proc.stdout.closed, "finished session must release its stdout pipe FD"  # type: ignore[union-attr]

    def test_move_to_finished_closes_pty(self, registry):
        """PTY-backed sessions release the PTY master on finish too."""
        pty_closed = {"closed": False}

        class _FakePty:
            def close(self):
                pty_closed["closed"] = True

        session = _make_session(sid="proc_pty_close", exited=True)
        session._pty = _FakePty()
        registry._finished[session.id] = session

        registry._move_to_finished(session)
        assert pty_closed["closed"]

    def test_poll_still_serves_output_after_handle_release(self, registry):
        """Output remains queryable after the pipes close — poll() reads the
        buffered output, never the (now-closed) pipe."""
        proc = subprocess.Popen(
            [sys.executable, "-c", "print('hello-finish'); import time; time.sleep(0.2)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        session = _make_session(sid="proc_poll_after_close", exited=False)
        session.process = proc
        registry._running[session.id] = session

        # Drain output like the reader loop would.
        proc.wait(timeout=5)
        try:
            tail = proc.stdout.read() if proc.stdout else ""
        except ValueError:
            tail = ""
        session.output_buffer = tail or ""
        session.exited = True
        session.exit_code = proc.returncode
        session.completion_reason = "exited"
        registry._move_to_finished(session)

        assert proc.stdout.closed
        result = registry.poll("proc_poll_after_close")
        assert result["status"] == "exited"
        assert "hello-finish" in result["output_preview"]

    def test_prune_releases_handles_of_dropped_sessions(self, registry):
        """TTL-prune must release handles of sessions that landed in
        _finished without passing through _move_to_finished (direct inserts).
        The release is idempotent, so double-close on the normal path is safe.
        """
        import time as _time

        pty_closed = {"closed": False}

        class _FakePty:
            def close(self):
                pty_closed["closed"] = True

        session = _make_session(sid="proc_prune_release", exited=True)
        session._pty = _FakePty()
        # Force TTL expiry.
        session.started_at = _time.time() - (FINISHED_TTL_SECONDS + 60)
        registry._finished[session.id] = session

        with registry._lock:
            registry._prune_if_needed()

        assert session.id not in registry._finished
        assert pty_closed["closed"], "pruned session must release its PTY handle"



# =========================================================================
# Spawn env sanitization
# =========================================================================

class TestSpawnEnvSanitization:
    def test_notification_checkpoint_precedes_local_reader_start(
        self, registry, tmp_path
    ):
        """The reader sees a fully configured, durable, registered session."""
        checkpoint = tmp_path / "processes.json"
        observed = {}

        proc = MagicMock()
        proc.pid = 4321
        proc.stdout = iter([])
        proc.stdin = MagicMock()
        proc.poll.return_value = None

        class InspectingThread:
            def __init__(self, *, target, args, **_kwargs):
                # Reader executes through copy_context().run, preserving the
                # producer profile; its first argument is the reader function.
                self.session = args[1]

            def start(self):
                session = self.session
                assert registry._running.get(session.id) is session
                persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
                assert persisted == [{
                    "session_id": session.id,
                    "checkpoint_owner_id": registry._checkpoint_owner_id,
                    "command": "printf ready",
                    "pid": 4321,
                    "pid_scope": "host",
                    "host_start_time": None,
                    "systemd_unit": "",
                    "cwd": str(tmp_path),
                    "started_at": session.started_at,
                    "task_id": "task",
                    "owner_task_id": "task",
                    "handoff_note": "",
                    "session_key": "agent:main:telegram:dm:123",
                    "watcher_platform": "telegram",
                    "watcher_chat_id": "123",
                    "watcher_user_id": "7",
                    "watcher_user_name": "Ada",
                    "watcher_thread_id": "42",
                    "watcher_message_id": "99",
                    "watcher_interval": 5,
                    "parent_session_id": "sess-parent",
                    "notify_on_complete": False,
                    "notify_on_failure": True,
                    "watch_patterns": ["FABLE_WAKE"],
                    "reliable_control_watch_delivered": False,
                    "reliable_control_close_seen": False,
                    "reliable_control_delivery_consumed": False,
                }]
                assert registry.pending_watchers[0]["session_id"] == session.id
                assert registry.pending_watchers[0]["notify_on_failure"] is True
                observed["reader_started"] = True

        notification = ProcessNotificationConfig(
            watcher_platform="telegram",
            watcher_chat_id="123",
            watcher_user_id="7",
            watcher_user_name="Ada",
            watcher_thread_id="42",
            watcher_message_id="99",
            watcher_interval=5,
            parent_session_id="sess-parent",
            notify_on_failure=True,
            watch_patterns=("FABLE_WAKE",),
        )
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint), \
            patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
            patch("tools.process_registry.subprocess.Popen", return_value=proc), \
            patch("tools.process_registry.threading.Thread", InspectingThread), \
            patch.object(registry, "_safe_host_start_time", return_value=None):
            session = registry.spawn_local(
                "printf ready",
                cwd=str(tmp_path),
                task_id="task",
                session_key="agent:main:telegram:dm:123",
                notification=notification,
            )

        assert observed == {"reader_started": True}
        assert session.notify_on_failure is True
        assert session.watch_patterns == ["FABLE_WAKE"]

    def test_ultrafast_marker_is_captured_once_without_spawn_sleep(
        self, registry, tmp_path
    ):
        """A child may print+exit immediately; the activation barrier suffices."""
        checkpoint = tmp_path / "processes.json"
        notification = ProcessNotificationConfig(
            notify_on_failure=True,
            watch_patterns=("FABLE_WAKE",),
        )
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            session = registry.spawn_local(
                "printf 'FABLE_WAKE ready\\n'",
                cwd=str(tmp_path),
                notification=notification,
            )
            assert session._completion_event.wait(5), "ultrafast child never reaped"

        events = []
        while not registry.completion_queue.empty():
            events.append(registry.completion_queue.get_nowait())
        matches = [event for event in events if event.get("type") == "watch_match"]
        completions = [event for event in events if event.get("type") == "completion"]
        assert len(matches) == 1
        assert matches[0]["pattern"] == "FABLE_WAKE"
        assert "FABLE_WAKE ready" in matches[0]["output"]
        assert completions == [], "the delivered sentinel suppresses failure fallback"
        assert session.id not in registry._running
        assert registry._finished.get(session.id) is session

    def test_notification_config_round_trips_through_checkpoint_recovery(
        self, registry, tmp_path, monkeypatch
    ):
        """Every routing/failure field needed after restart is recovered."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        session = ProcessSession(
            id="proc_roundtrip",
            command="watcher",
            pid=4242,
            pid_scope="host",
            host_start_time=111,
            started_at=1234.5,
            task_id="task",
            session_key="agent:main:telegram:dm:123:42",
            watcher_platform="telegram",
            watcher_chat_id="123",
            watcher_user_id="7",
            watcher_user_name="Ada",
            watcher_thread_id="42",
            watcher_message_id="99",
            watcher_interval=5,
            parent_session_id="sess-parent",
            notify_on_failure=True,
            watch_patterns=["FABLE_WAKE"],
        )
        registry._running[session.id] = session
        assert registry._write_checkpoint() is True

        recovered_registry = ProcessRegistry()
        monkeypatch.setattr(
            recovered_registry, "_host_pid_is_ours", lambda *_args: True
        )
        assert recovered_registry.recover_from_checkpoint() == 1
        recovered = recovered_registry._running.get(session.id)
        assert recovered is not None
        for field in (
            "session_key",
            "watcher_platform",
            "watcher_chat_id",
            "watcher_user_id",
            "watcher_user_name",
            "watcher_thread_id",
            "watcher_message_id",
            "watcher_interval",
            "parent_session_id",
            "notify_on_failure",
            "watch_patterns",
        ):
            assert getattr(recovered, field) == getattr(session, field)
        assert recovered_registry.pending_watchers == [{
            "session_id": session.id,
            "check_interval": 5,
            "session_key": session.session_key,
            "platform": "telegram",
            "chat_id": "123",
            "user_id": "7",
            "user_name": "Ada",
            "thread_id": "42",
            "message_id": "99",
            "notify_on_complete": False,
            "notify_on_failure": True,
            "parent_session_id": "sess-parent",
        }]

    def test_spawn_local_strips_blocked_vars_from_background_env(self, registry):
        captured = {}

        def fake_popen(cmd, **kwargs):
            captured["env"] = kwargs["env"]
            proc = MagicMock()
            proc.pid = 4321
            proc.stdout = iter([])
            proc.stdin = MagicMock()
            proc.poll.return_value = None
            return proc

        fake_thread = MagicMock()

        with patch.dict(os.environ, {
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/user",
            "USER": "tester",
            "TELEGRAM_BOT_TOKEN": "bot-secret",
            "FIRECRAWL_API_KEY": "fc-secret",
        }, clear=True), \
            patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
            patch("subprocess.Popen", side_effect=fake_popen), \
            patch("threading.Thread", return_value=fake_thread), \
            patch.object(registry, "_write_checkpoint"):
            registry.spawn_local(
                "echo hello",
                cwd="/tmp",
                env_vars={
                    "MY_CUSTOM_VAR": "keep-me",
                    "TELEGRAM_BOT_TOKEN": "drop-me",
                    f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}TELEGRAM_BOT_TOKEN": "forced-bot-token",
                },
            )

        env = captured["env"]
        assert env["MY_CUSTOM_VAR"] == "keep-me"
        assert env["TELEGRAM_BOT_TOKEN"] == "forced-bot-token"
        assert "FIRECRAWL_API_KEY" not in env
        assert f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}TELEGRAM_BOT_TOKEN" not in env
        assert env["PYTHONUNBUFFERED"] == "1"

    def test_spawn_via_env_checks_returncode_when_wrapper_fails(self, registry):
        class FakeEnv:
            def __init__(self):
                self.commands = []

            def execute(self, command, **kwargs):
                self.commands.append((command, kwargs))
                return {"output": "syntax error", "returncode": 2}

        env = FakeEnv()
        fake_thread = MagicMock()

        with patch("tools.process_registry.threading.Thread", return_value=fake_thread), \
            patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_via_env(env, "echo hello")

        assert session.exited is True
        assert session.exit_code == 2
        assert session.pid is None
        assert session.output_buffer == "syntax error"
        fake_thread.start.assert_not_called()
        # A failed launch must not be exposed as a running/tracked session.
        assert session.id not in registry._running

    def test_env_poller_quotes_temp_paths_with_spaces(self, registry):
        session = _make_session(sid="proc_space")
        session.exited = False

        class FakeEnv:
            def __init__(self):
                self.commands = []
                self._responses = iter([
                    {"output": "6 0\nhello\n"},
                    {"output": "1\n"},
                    {"output": "0\n"},
                ])

            def execute(self, command, **kwargs):
                self.commands.append((command, kwargs))
                return next(self._responses)

        env = FakeEnv()

        with patch("tools.process_registry.time.sleep", return_value=None), \
            patch.object(registry, "_move_to_finished"):
            registry._env_poller_loop(
                session,
                env,
                "/path with spaces/hermes_bg.log",
                "/path with spaces/hermes_bg.pid",
                "/path with spaces/hermes_bg.exit",
            )

        assert "'/path with spaces/hermes_bg.log'" in env.commands[0][0]
        assert "cat '/path with spaces/hermes_bg.log'" not in env.commands[0][0]
        assert env.commands[1][0] == "kill -0 \"$(cat '/path with spaces/hermes_bg.pid' 2>/dev/null)\" 2>/dev/null; echo $?"
        assert env.commands[2][0] == "cat '/path with spaces/hermes_bg.exit' 2>/dev/null"


class TestEnvPollerIncrementalRead:
    """The sandbox log poller must read only new bytes, not the whole file.

    Reading the whole file every poll made one poll cost grow with the total
    output so far, so a long noisy job re-sent all of its output over the
    docker or SSH channel every two seconds.
    """

    @staticmethod
    def _run_poller(registry, session, responses):
        """Drive one poll cycle and hand back the commands the env saw."""

        class FakeEnv:
            def __init__(self):
                self.commands = []
                self._responses = iter(responses)

            def execute(self, command, **kwargs):
                self.commands.append(command)
                return next(self._responses)

        env = FakeEnv()
        with patch("tools.process_registry.time.sleep", return_value=None), \
            patch.object(registry, "_move_to_finished"):
            registry._env_poller_loop(
                session, env, "/tmp/bg.log", "/tmp/bg.pid", "/tmp/bg.exit"
            )
        return env.commands

    def test_read_command_asks_only_for_new_bytes(self):
        cmd = ProcessRegistry._log_delta_command("'/tmp/bg.log'", 4096)
        # The offset is carried into the command, and the file is opened with
        # tail rather than cat.
        assert "O=4096" in cmd
        assert "tail -c +$((O+1)) '/tmp/bg.log'" in cmd
        assert "cat '/tmp/bg.log'" not in cmd

    def test_read_command_starts_from_zero_on_first_poll(self):
        cmd = ProcessRegistry._log_delta_command("'/tmp/bg.log'", 0)
        assert "O=0" in cmd

    @pytest.mark.skipif(not shutil.which("sh"), reason="needs a POSIX sh")
    def test_read_command_holds_back_a_split_utf8_sequence(self, tmp_path):
        """A multibyte character straddling two polls must not be split.

        The backend decodes each execute() result on its own, so returning
        the first byte of an 'é' in one poll and the rest in the next would
        yield replacement characters in the transcript (and break watch
        patterns at the seam). Every prefix of a mixed ASCII/2/3/4-byte
        string must come back decodable, with at most 3 bytes held back and
        nothing held back once the trailing character is complete.
        """
        full = "hé😀中a\n€bz🚀".encode()
        log = tmp_path / "bg.log"
        quoted = shlex.quote(str(log))
        for n in range(1, len(full) + 1):
            log.write_bytes(full[:n])
            out = subprocess.run(
                ["sh", "-c", ProcessRegistry._log_delta_command(quoted, 0)],
                capture_output=True, timeout=30,
            ).stdout
            header, _, delta = out.partition(b"\n")
            size, _offset = map(int, header.split())
            delta.decode("utf-8")  # must not raise
            assert delta == full[:size]
            complete = full[:n].decode("utf-8", "ignore").encode() == full[:n]
            assert (n - size) == 0 if complete else 0 < (n - size) <= 3

    def test_first_poll_reads_from_the_start(self, registry):
        session = _make_session(sid="proc_delta")
        session.exited = False
        commands = self._run_poller(
            registry,
            session,
            [
                {"output": "11 0\nfirst chunk"},
                {"output": "1\n"},
                {"output": "0\n"},
            ],
        )
        assert "O=0" in commands[0]
        assert session.output_buffer == "first chunk"

    def test_delta_is_appended_not_replaced(self, registry):
        session = _make_session(sid="proc_append", output="already here ")
        session.exited = False
        self._run_poller(
            registry,
            session,
            [
                {"output": "8 0\nand new"},
                {"output": "1\n"},
                {"output": "0\n"},
            ],
        )
        assert session.output_buffer == "already here and new"

    def test_second_poll_asks_from_where_the_first_one_stopped(self, registry):
        session = _make_session(sid="proc_two_polls")
        session.exited = False
        commands = self._run_poller(
            registry,
            session,
            [
                {"output": "11 0\nfirst chunk"},
                {"output": "0\n"},          # still running, poll again
                {"output": "17 11\n and more"},
                {"output": "1\n"},          # gone now
                {"output": "0\n"},
            ],
        )
        assert "O=0" in commands[0]
        # The second read starts at byte 11, so the first chunk is not sent
        # a second time.
        assert "O=11" in commands[2]
        assert session.output_buffer == "first chunk and more"

    def test_truncated_log_drops_the_stale_buffer(self, registry):
        session = _make_session(sid="proc_rotate")
        session.exited = False
        # The second read reports offset 0 even though the first one left off
        # at byte 11. The file no longer reaches that byte, so it was rotated
        # or truncated and the buffer we hold no longer matches it.
        self._run_poller(
            registry,
            session,
            [
                {"output": "11 0\nfirst chunk"},
                {"output": "0\n"},          # still running, poll again
                {"output": "5 0\nfresh"},
                {"output": "1\n"},
                {"output": "0\n"},
            ],
        )
        assert session.output_buffer == "fresh"

    def test_unreadable_header_leaves_the_buffer_alone(self, registry):
        session = _make_session(sid="proc_bad", output="keep me")
        session.exited = False
        # No header at all, for example when the shell is missing one of the
        # tools the command needs.
        self._run_poller(
            registry,
            session,
            [
                {"output": ""},
                {"output": "1\n"},
                {"output": "0\n"},
            ],
        )
        assert session.output_buffer == "keep me"

    def test_buffer_stays_within_the_cap(self, registry):
        session = _make_session(sid="proc_cap")
        session.exited = False
        session.max_output_chars = 10
        self._run_poller(
            registry,
            session,
            [
                {"output": "20 0\n" + "x" * 20},
                {"output": "1\n"},
                {"output": "0\n"},
            ],
        )
        assert session.output_buffer == "x" * 10


# =========================================================================
# Popen leak prevention
# =========================================================================

class TestPopenLeakOnSetupFailure:
    """Regression for issue #2749: subprocess orphaned when post-Popen setup raises."""

    def test_popen_killed_when_thread_creation_fails(self, registry):
        """If Thread() raises after Popen, proc must be killed — not orphaned."""
        killed = []

        proc = MagicMock()
        proc.pid = 9999
        proc.stdout = iter([])
        proc.stdin = MagicMock()
        proc.poll.return_value = None

        def fake_kill():
            killed.append(True)

        proc.kill = fake_kill
        proc.wait = MagicMock()

        def boom(*args, **kwargs):
            raise RuntimeError("Thread creation failed")

        # proc.pid is a MagicMock-backed fake; os.getpgid(fake_pid) would query
        # the real OS for an arbitrary PID. On a busy host that PID may exist,
        # in which case spawn_local's primary cleanup path
        # (os.killpg(os.getpgid(pid), SIGKILL)) succeeds against an UNRELATED
        # real process group and proc.kill() is never reached — flaky failure,
        # and a real risk of SIGKILLing an innocent process group. Force the
        # ProcessLookupError fallback so the test deterministically exercises
        # proc.kill() and never issues a real killpg.
        with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("threading.Thread", side_effect=boom), \
             patch("os.getpgid", side_effect=ProcessLookupError), \
             patch.object(registry, "_write_checkpoint"):
            with pytest.raises(RuntimeError, match="Thread creation failed"):
                registry.spawn_local("echo hello", cwd="/tmp")

        assert killed, "proc.kill() must be called when post-Popen setup raises"

# =========================================================================
# Spawn rewrite regression (issue #68915)
# =========================================================================


class TestSpawnRewriteCompoundBackground:
    """Verify that spawn_local rewrites `A && B &` patterns to avoid subshell deadlocks.

    Issue #68915: when bash parses ``A && B &`` it forks a subshell ``(A && B) &``.
    If B is a long-running server, the subshell never exits and holds the stdout
    pipe open, causing a permanent deadlock. The rewriter wraps the tail to
    ``A && { B & }`` so no subshell fork occurs.
    """

    def test_compound_and_background_gets_rewritten(self, registry):
        """A && B & must be rewritten to A && { B & } before Popen."""
        captured_cmd = []

        def fake_popen(args, **kwargs):
            captured_cmd.append(args)
            proc = MagicMock()
            proc.pid = 1111
            proc.stdout = MagicMock()
            return proc

        fake_thread = MagicMock()
        fake_thread.daemon = False

        with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
             patch("subprocess.Popen", side_effect=fake_popen), \
             patch("threading.Thread", return_value=fake_thread), \
             patch.object(registry, "_write_checkpoint"):
            registry.spawn_local("cd /app && node server.js &>/tmp/srv.log &", cwd="/tmp")

        assert len(captured_cmd) == 1
        shell_cmd = captured_cmd[0]
        # The command passed to Popen should be the REWRITTEN version
        assert "&& { node server.js &>/tmp/srv.log & }" in shell_cmd[2]

    def test_simple_background_preserved(self, registry):
        """Simple cmd & (no &&) must NOT be rewritten — no subshell bug."""
        captured_cmd = []

        def fake_popen(args, **kwargs):
            captured_cmd.append(args)
            proc = MagicMock()
            proc.pid = 2222
            proc.stdout = MagicMock()
            return proc

        fake_thread = MagicMock()
        fake_thread.daemon = False

        with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
             patch("subprocess.Popen", side_effect=fake_popen), \
             patch("threading.Thread", return_value=fake_thread), \
             patch.object(registry, "_write_checkpoint"):
            registry.spawn_local("sleep 5 &", cwd="/tmp")

        assert len(captured_cmd) == 1
        shell_cmd = captured_cmd[0][2]
        # Simple background must remain as-is
        assert "sleep 5 &" in shell_cmd

    def test_pty_path_uses_rewritten_command(self, registry):
        """PTY spawn path must also use the rewritten command (issue #68915)."""
        mock_pty_proc = MagicMock()
        mock_pty_proc.pid = 5555

        mock_pty_module = MagicMock()
        mock_pty_module.PtyProcess.spawn = MagicMock(return_value=mock_pty_proc)

        fake_thread = MagicMock()
        fake_thread.daemon = False

        with patch("tools.process_registry._find_shell", return_value="/bin/bash"), \
             patch.dict("sys.modules", {"ptyprocess": mock_pty_module}), \
             patch("threading.Thread", return_value=fake_thread), \
             patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_local(
                "cd /app && node server.js &",
                cwd="/tmp",
                use_pty=True,
            )

        assert mock_pty_module.PtyProcess.spawn.called, \
            "PTY spawn should have been attempted"
        pty_args = mock_pty_module.PtyProcess.spawn.call_args[0][0]
        assert "&& { node server.js & }" in pty_args[2], \
            f"PTY path should use rewritten command, got: {pty_args[2]}"
        assert session.command == "cd /app && node server.js &"


# =========================================================================
# Checkpoint
# =========================================================================

class TestCheckpoint:
    def test_checkpoint_merges_disjoint_process_owners(self, tmp_path):
        """Gateway and CLI snapshots sharing HERMES_HOME never erase each other."""
        checkpoint = tmp_path / "procs.json"
        first = ProcessRegistry()
        second = ProcessRegistry()
        first_session = _make_session(sid="proc_first")
        second_session = _make_session(sid="proc_second")
        first_session.pid = os.getpid()
        second_session.pid = os.getpid()
        first._running[first_session.id] = first_session
        second._running[second_session.id] = second_session

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            assert first._write_checkpoint() is True
            assert second._write_checkpoint() is True
            entries = json.loads(checkpoint.read_text(encoding="utf-8"))
            assert {entry["session_id"] for entry in entries} == {
                "proc_first", "proc_second",
            }
            assert len({entry["checkpoint_owner_id"] for entry in entries}) == 2

            first._running.clear()
            assert first._write_checkpoint() is True
            entries = json.loads(checkpoint.read_text(encoding="utf-8"))
            assert [entry["session_id"] for entry in entries] == ["proc_second"]

    def test_checkpoint_file_lock_serializes_disjoint_registry_writers(
        self, tmp_path
    ):
        """Independent registries cannot overlap snapshot replacement."""
        from utils import atomic_json_write as real_atomic_json_write

        checkpoint = tmp_path / "procs.json"
        first = ProcessRegistry()
        second = ProcessRegistry()
        first._running["proc_first"] = _make_session(sid="proc_first")
        second._running["proc_second"] = _make_session(sid="proc_second")
        first._running["proc_first"].pid = os.getpid()
        second._running["proc_second"].pid = os.getpid()
        start = threading.Barrier(2)
        counter_lock = threading.Lock()
        active_writes = 0
        max_active_writes = 0
        results = []

        def slow_atomic_write(path, data):
            nonlocal active_writes, max_active_writes
            with counter_lock:
                active_writes += 1
                max_active_writes = max(max_active_writes, active_writes)
            try:
                time.sleep(0.05)
                real_atomic_json_write(path, data)
            finally:
                with counter_lock:
                    active_writes -= 1

        def publish(registry):
            start.wait()
            results.append(registry._write_checkpoint())

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint), patch(
            "utils.atomic_json_write", side_effect=slow_atomic_write
        ):
            threads = [
                threading.Thread(target=publish, args=(first,)),
                threading.Thread(target=publish, args=(second,)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

        assert all(not thread.is_alive() for thread in threads)
        assert results == [True, True]
        assert max_active_writes == 1
        entries = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert {entry["session_id"] for entry in entries} == {
            "proc_first", "proc_second",
        }

    def test_checkpoint_write_never_replaces_corrupt_shared_snapshot(
        self, tmp_path, monkeypatch
    ):
        """Unknown existing state cannot be treated as an empty owner set."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        checkpoint.write_text("{corrupt shared state", encoding="utf-8")
        registry = ProcessRegistry()
        session = _make_session(sid="proc_would_overwrite")
        registry._running[session.id] = session

        assert registry._write_checkpoint() is False
        assert checkpoint.read_text(encoding="utf-8") == "{corrupt shared state"

    def test_recovery_refuses_corrupt_checkpoint(self, tmp_path, monkeypatch):
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        checkpoint.write_text("{corrupt", encoding="utf-8")

        with pytest.raises(ProcessCheckpointRecoveryError):
            ProcessRegistry().recover_from_checkpoint()

    def test_recovery_refuses_unreadable_checkpoint(self, tmp_path, monkeypatch):
        from pathlib import Path
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        checkpoint.write_text("[]", encoding="utf-8")
        original_read_text = Path.read_text

        def unreadable_checkpoint(path, *args, **kwargs):
            if path == checkpoint:
                raise PermissionError("simulated checkpoint permission error")
            return original_read_text(path, *args, **kwargs)

        with patch.object(Path, "read_text", unreadable_checkpoint):
            with pytest.raises(ProcessCheckpointRecoveryError):
                ProcessRegistry().recover_from_checkpoint()

    def test_recovery_read_waits_for_checkpoint_file_lock(
        self, tmp_path, monkeypatch
    ):
        """Recovery and snapshot replace share one inter-process lock."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        checkpoint.write_text("[]", encoding="utf-8")
        lock_path = checkpoint.with_name(f"{checkpoint.name}.lock")
        registry = ProcessRegistry()
        entered = threading.Event()
        finished = threading.Event()
        result = []

        def recover():
            entered.set()
            result.append(registry.recover_from_checkpoint())
            finished.set()

        thread = threading.Thread(target=recover)
        from tools.process_registry_control import _CheckpointFileLock
        with _CheckpointFileLock(lock_path):
            thread.start()
            assert entered.wait(5)
            assert finished.wait(0.1) is False
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert finished.is_set()
        assert result == [0]

    def test_reliable_watch_outbox_round_trips_until_ack(
        self, tmp_path, monkeypatch
    ):
        """A captured marker survives gateway death and is removed only on ack."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        producer = ProcessRegistry()
        session = _make_session(sid="proc_durable", started_at=1234.5)
        session.pid = os.getpid()
        session.watch_patterns = ["FABLE_WAKE"]
        session.notify_on_failure = True
        session.watcher_platform = "telegram"
        session.watcher_chat_id = "123"
        session.session_key = "agent:main:telegram:dm:123"
        session.parent_session_id = "sess-parent"
        producer._running[session.id] = session
        assert producer._write_checkpoint() is True

        # Simulate a crash window where the outbox fsync succeeded but the
        # companion process snapshot did not. Recovery must repair that
        # half-transaction before the event becomes acknowledgeable.
        with patch.object(producer, "_write_checkpoint", return_value=False):
            producer._check_watch_patterns(
                session, "FABLE_WAKE reason=decision\n"
            )
        live_event = producer.completion_queue.get_nowait()
        assert live_event["checkpoint_confirmed"] is False
        assert json.loads(checkpoint.read_text())[0][
            "reliable_control_watch_delivered"
        ] is False
        outbox = tmp_path / "process_notifications.json"
        assert [item["delivery_id"] for item in json.loads(outbox.read_text())] == [
            live_event["delivery_id"]
        ]

        restarted = ProcessRegistry()
        monkeypatch.setattr(restarted, "_host_pid_is_ours", lambda *_args: True)
        assert restarted.recover_from_checkpoint() == 1
        restored_event = restarted.completion_queue.get_nowait()
        assert restored_event["delivery_id"] == live_event["delivery_id"]
        assert restored_event["restored"] is True
        assert restored_event["checkpoint_confirmed"] is True
        assert json.loads(checkpoint.read_text())[0][
            "reliable_control_watch_delivered"
        ] is True
        restored_session = restarted.get(session.id)
        assert restored_session._reliable_control_watch_delivered is True
        assert restored_session._reliable_control_watch_persisted is True

        assert restarted.acknowledge_watch_event(restored_event) is True
        assert json.loads(outbox.read_text()) == []
        after_ack = ProcessRegistry()
        monkeypatch.setattr(after_ack, "_host_pid_is_ours", lambda *_args: True)
        assert after_ack.recover_from_checkpoint() == 1
        assert after_ack.completion_queue.empty()

    def test_live_recovered_fable_watcher_wakes_when_stream_is_unrecoverable(
        self, tmp_path, monkeypatch
    ):
        """A surviving PID without a reattachable reader fails closed now."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        producer = ProcessRegistry()
        session = _make_session(sid="proc_live_unobservable", started_at=2222.0)
        session.pid = 4242
        session.host_start_time = 111
        session.watch_patterns = ["FABLE_WAKE"]
        session.notify_on_failure = True
        session.watcher_platform = "telegram"
        session.watcher_chat_id = "123"
        session.parent_session_id = "sess-parent"
        producer._running[session.id] = session
        assert producer._write_checkpoint() is True

        restarted = ProcessRegistry()
        monkeypatch.setattr(restarted, "_host_pid_is_ours", lambda *_args: True)
        assert restarted.recover_from_checkpoint() == 1

        recovered = restarted._running[session.id]
        assert recovered.detached is True
        assert recovered._reliable_control_watch_delivered is True
        event = restarted.completion_queue.get_nowait()
        assert event["pattern"] == "FABLE_WAKE"
        assert event["control_reason"] == "output_stream_unrecoverable"
        assert event["termination_source"] == (
            "checkpoint_output_stream_unrecoverable"
        )
        assert event["checkpoint_confirmed"] is True
        assert restarted.completion_queue.empty()
        persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert persisted[0]["reliable_control_watch_delivered"] is True

    def test_live_recovery_deduplicates_existing_fable_outbox_event(
        self, tmp_path, monkeypatch
    ):
        """An outbox half-transaction is restored, not replaced by fallback."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        producer = ProcessRegistry()
        session = _make_session(sid="proc_live_outbox", started_at=3333.0)
        session.pid = 4242
        session.host_start_time = 111
        session.watch_patterns = ["FABLE_WAKE"]
        session.notify_on_failure = True
        session.watcher_platform = "telegram"
        session.watcher_chat_id = "123"
        producer._running[session.id] = session
        assert producer._write_checkpoint() is True
        existing = producer._build_reliable_control_failure_event(
            session,
            control_reason="primary_captured_before_restart",
        )
        assert producer._persist_reliable_watch_event(existing) is True

        restarted = ProcessRegistry()
        monkeypatch.setattr(restarted, "_host_pid_is_ours", lambda *_args: True)
        assert restarted.recover_from_checkpoint() == 1

        restored = restarted.completion_queue.get_nowait()
        assert restored["delivery_id"] == existing["delivery_id"]
        assert restored["control_reason"] == "primary_captured_before_restart"
        assert restarted.completion_queue.empty()
        outbox = tmp_path / "process_notifications.json"
        assert [
            item["delivery_id"]
            for item in json.loads(outbox.read_text(encoding="utf-8"))
        ] == [existing["delivery_id"]]
        assert json.loads(checkpoint.read_text(encoding="utf-8"))[0][
            "reliable_control_watch_delivered"
        ] is True

    def test_live_recovered_scope_retains_cleanup_row_when_reap_fails(
        self, tmp_path, monkeypatch
    ):
        """Later wrapper death cannot orphan an unreaped recovered scope."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        producer = ProcessRegistry()
        session = _make_session(sid="proc_live_scope", started_at=4444.0)
        session.pid = 4242
        session.host_start_time = 111
        session.systemd_unit = "hermes-worker-proc_live_scope.scope"
        session.watch_patterns = ["FABLE_WAKE"]
        session.notify_on_failure = True
        session.watcher_platform = "telegram"
        session.watcher_chat_id = "123"
        producer._running[session.id] = session
        assert producer._write_checkpoint() is True

        restarted = ProcessRegistry()
        monkeypatch.setattr(restarted, "_host_pid_is_ours", lambda *_args: True)
        assert restarted.recover_from_checkpoint() == 1
        recovered = restarted._running[session.id]
        assert restarted.completion_queue.qsize() == 1

        monkeypatch.setattr(restarted, "_host_pid_is_ours", lambda *_args: False)
        with patch(
            "tools.process_registry._stop_systemd_unit", return_value=False
        ) as stop_unit:
            restarted._refresh_detached_session(recovered)

        stop_unit.assert_called_once_with(session.systemd_unit)
        assert recovered.exited is True
        assert recovered.completion_reason == "lost"
        assert recovered.termination_source == "checkpoint_scope_reap_failed"
        assert recovered.id in restarted._finished
        assert restarted.completion_queue.qsize() == 1
        retained = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert retained[0]["session_id"] == session.id
        assert retained[0]["systemd_unit"] == session.systemd_unit
        assert retained[0]["reliable_control_watch_delivered"] is True

    def test_consumed_watch_tombstone_prevents_restart_revival_when_delete_fails(
        self, tmp_path, monkeypatch
    ):
        """Inline consumption is durable even if its outbox delete loses I/O."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        producer = ProcessRegistry()
        session = _make_session(sid="proc_consumed_io_fail", started_at=4321.5)
        session.pid = 999999999
        session.host_start_time = 111
        session.watch_patterns = ["FABLE_WAKE"]
        session.notify_on_failure = True
        session.watcher_platform = "telegram"
        session.watcher_chat_id = "123"
        producer._running[session.id] = session
        assert producer._write_checkpoint() is True
        producer._check_watch_patterns(session, "FABLE_WAKE reason=attention\n")
        producer.completion_queue.get_nowait()

        session.exited = True
        with patch.object(
            producer,
            "discard_reliable_watch_events_for_session",
            return_value=False,
        ):
            producer._consume_completion_result(session)

        persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert persisted[0]["reliable_control_delivery_consumed"] is True
        outbox = tmp_path / "process_notifications.json"
        assert json.loads(outbox.read_text(encoding="utf-8"))

        restarted = ProcessRegistry()
        monkeypatch.setattr(restarted, "_host_pid_is_ours", lambda *_args: False)
        monkeypatch.setattr(restarted, "_is_host_pid_alive", lambda *_args: False)
        assert restarted.recover_from_checkpoint() == 0
        assert restarted.completion_queue.empty()
        assert json.loads(outbox.read_text(encoding="utf-8")) == []
        assert json.loads(checkpoint.read_text(encoding="utf-8")) == []

    def test_corrupt_outbox_cannot_falsely_ack_consumed_tombstone(
        self, tmp_path, monkeypatch
    ):
        """Unreadable outbox aborts startup without clearing consumption."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        producer = ProcessRegistry()
        session = _make_session(sid="proc_consumed_corrupt", started_at=8765.5)
        session.pid = 999999999
        session.host_start_time = 111
        session.exited = True
        session.watch_patterns = ["FABLE_WAKE"]
        session.notify_on_failure = True
        session.watcher_platform = "telegram"
        session.watcher_chat_id = "123"
        event = producer._build_reliable_control_failure_event(session)
        assert producer._persist_reliable_watch_event(event) is True
        producer._completion_consumed.add(session.id)
        consumed_entry = producer._checkpoint_entry_for_session(session)
        assert producer._write_checkpoint(extra_entries=[consumed_entry]) is True

        outbox = tmp_path / "process_notifications.json"
        valid_outbox = outbox.read_text(encoding="utf-8")
        outbox.write_text("{corrupt", encoding="utf-8")

        first_restart = ProcessRegistry()
        monkeypatch.setattr(
            first_restart, "_host_pid_is_ours", lambda *_args: False
        )
        monkeypatch.setattr(
            first_restart, "_is_host_pid_alive", lambda *_args: False
        )
        with pytest.raises(ProcessCheckpointRecoveryError):
            first_restart.recover_from_checkpoint()
        assert first_restart.completion_queue.empty()
        retained = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert retained[0]["reliable_control_delivery_consumed"] is True

        # Once storage becomes readable, recovery deletes the stale event and
        # only then prunes the consume tombstone. It never exposes a wake.
        outbox.write_text(valid_outbox, encoding="utf-8")
        second_restart = ProcessRegistry()
        monkeypatch.setattr(
            second_restart, "_host_pid_is_ours", lambda *_args: False
        )
        monkeypatch.setattr(
            second_restart, "_is_host_pid_alive", lambda *_args: False
        )
        assert second_restart.recover_from_checkpoint() == 0
        assert second_restart.completion_queue.empty()
        assert json.loads(outbox.read_text(encoding="utf-8")) == []
        assert json.loads(checkpoint.read_text(encoding="utf-8")) == []

    def test_dead_delivered_checkpoint_fails_closed_when_outbox_is_corrupt(
        self, tmp_path, monkeypatch
    ):
        """Corrupt durable wake storage aborts startup until repaired."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        producer = ProcessRegistry()
        session = _make_session(sid="proc_delivered_corrupt", started_at=7777.0)
        session.pid = 999999999
        session.host_start_time = 111
        session.watch_patterns = ["FABLE_WAKE"]
        session.notify_on_failure = True
        session.watcher_platform = "telegram"
        session.watcher_chat_id = "123"
        producer._running[session.id] = session
        assert producer._write_checkpoint() is True
        producer._check_watch_patterns(session, "FABLE_WAKE reason=attention\n")
        primary = producer.completion_queue.get_nowait()
        outbox = tmp_path / "process_notifications.json"
        valid_outbox = outbox.read_text(encoding="utf-8")
        assert json.loads(checkpoint.read_text(encoding="utf-8"))[0][
            "reliable_control_watch_delivered"
        ] is True
        outbox.write_text("{corrupt", encoding="utf-8")

        first_restart = ProcessRegistry()
        monkeypatch.setattr(
            first_restart, "_host_pid_is_ours", lambda *_args: False
        )
        monkeypatch.setattr(
            first_restart, "_is_host_pid_alive", lambda *_args: False
        )
        with pytest.raises(ProcessCheckpointRecoveryError):
            first_restart.recover_from_checkpoint()
        assert first_restart.completion_queue.empty()
        retained = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert retained[0]["session_id"] == session.id

        # Once storage is repaired, the original durable marker is restored
        # exactly as-is; startup never served while its state was unknown.
        outbox.write_text(valid_outbox, encoding="utf-8")
        second_restart = ProcessRegistry()
        monkeypatch.setattr(
            second_restart, "_host_pid_is_ours", lambda *_args: False
        )
        monkeypatch.setattr(
            second_restart, "_is_host_pid_alive", lambda *_args: False
        )
        assert second_restart.recover_from_checkpoint() == 0
        restored = second_restart.completion_queue.get_nowait()
        assert restored["delivery_id"] == primary["delivery_id"]
        assert second_restart.completion_queue.empty()

    def test_live_delivered_checkpoint_fails_closed_on_outbox_read_error(
        self, tmp_path, monkeypatch
    ):
        """A permission/read failure aborts before adopting a live watcher."""
        from pathlib import Path
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        producer = ProcessRegistry()
        session = _make_session(sid="proc_delivered_unreadable", started_at=8888.0)
        session.pid = 4242
        session.host_start_time = 111
        session.watch_patterns = ["FABLE_WAKE"]
        session.notify_on_failure = True
        session.watcher_platform = "telegram"
        session.watcher_chat_id = "123"
        producer._running[session.id] = session
        assert producer._write_checkpoint() is True
        producer._check_watch_patterns(session, "FABLE_WAKE reason=attention\n")
        primary = producer.completion_queue.get_nowait()
        outbox = tmp_path / "process_notifications.json"

        original_read_text = Path.read_text

        def unreadable_outbox(path, *args, **kwargs):
            if path == outbox:
                raise PermissionError("simulated unreadable outbox")
            return original_read_text(path, *args, **kwargs)

        first_restart = ProcessRegistry()
        monkeypatch.setattr(
            first_restart, "_host_pid_is_ours", lambda *_args: True
        )
        with patch.object(Path, "read_text", unreadable_outbox):
            with pytest.raises(ProcessCheckpointRecoveryError):
                first_restart.recover_from_checkpoint()

        assert first_restart.completion_queue.empty()
        assert first_restart._running == {}
        assert json.loads(checkpoint.read_text(encoding="utf-8"))[0][
            "reliable_control_watch_delivered"
        ] is True

        second_restart = ProcessRegistry()
        monkeypatch.setattr(
            second_restart, "_host_pid_is_ours", lambda *_args: True
        )
        assert second_restart.recover_from_checkpoint() == 1
        restored = second_restart.completion_queue.get_nowait()
        assert restored["delivery_id"] == primary["delivery_id"]
        assert second_restart.completion_queue.empty()

    def test_outbox_only_corruption_aborts_recovery(self, tmp_path, monkeypatch):
        """Outbox-only is a valid crash state and cannot degrade to empty."""
        import tools.process_registry as pr_module

        monkeypatch.setattr(
            pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json"
        )
        (tmp_path / "process_notifications.json").write_text(
            "{corrupt", encoding="utf-8"
        )

        restarted = ProcessRegistry()
        with pytest.raises(ProcessCheckpointRecoveryError):
            restarted.recover_from_checkpoint()
        assert restarted.completion_queue.empty()

    def test_outbox_only_read_error_aborts_recovery(self, tmp_path, monkeypatch):
        """A permission failure cannot let an outbox-only wake disappear."""
        from pathlib import Path
        import tools.process_registry as pr_module

        monkeypatch.setattr(
            pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json"
        )
        outbox = tmp_path / "process_notifications.json"
        outbox.write_text("[]", encoding="utf-8")
        original_read_text = Path.read_text

        def unreadable_outbox(path, *args, **kwargs):
            if path == outbox:
                raise PermissionError("simulated unreadable outbox")
            return original_read_text(path, *args, **kwargs)

        restarted = ProcessRegistry()
        with patch.object(Path, "read_text", unreadable_outbox):
            with pytest.raises(ProcessCheckpointRecoveryError):
                restarted.recover_from_checkpoint()
        assert restarted.completion_queue.empty()

    @pytest.mark.parametrize(
        "pid",
        [os.getpid(), 999999999],
        ids=["live-checkpoint", "dead-checkpoint"],
    )
    def test_semantically_invalid_outbox_aborts_before_recovery_mutation(
        self, tmp_path, monkeypatch, pid
    ):
        """A JSON object that is not a FABLE event is corruption, not noise."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        producer = ProcessRegistry()
        session = _make_session(sid=f"proc_bad_event_{pid}", started_at=8890.0)
        session.pid = pid
        session.host_start_time = 111
        session.watch_patterns = ["FABLE_WAKE"]
        session.notify_on_failure = True
        session.watcher_platform = "telegram"
        session.watcher_chat_id = "123"
        producer._running[session.id] = session
        assert producer._write_checkpoint() is True
        producer._check_watch_patterns(session, "FABLE_WAKE reason=attention\n")
        producer.completion_queue.get_nowait()

        outbox = tmp_path / "process_notifications.json"
        malformed = json.loads(outbox.read_text(encoding="utf-8"))
        malformed[0].pop("pattern")
        outbox.write_text(json.dumps(malformed), encoding="utf-8")
        checkpoint_before = checkpoint.read_text(encoding="utf-8")

        restarted = ProcessRegistry()
        with pytest.raises(ProcessCheckpointRecoveryError):
            restarted.recover_from_checkpoint()

        assert restarted.completion_queue.empty()
        assert restarted._running == {}
        assert checkpoint.read_text(encoding="utf-8") == checkpoint_before

    @pytest.mark.parametrize(
        "entry",
        [
            {
                "session_id": "proc_missing_pid",
                "command": "sleep 1",
            },
            {
                "session_id": "proc_zero_pid",
                "command": "sleep 1",
                "pid": 0,
            },
            {
                "session_id": "proc_bad_fable_state",
                "command": "agent-wait-job.sh",
                "pid": 123,
                "watch_patterns": ["FABLE_WAKE"],
                "reliable_control_watch_delivered": "yes",
            },
            {
                "session_id": "proc_bad_tombstone",
                "command": "agent-wait-job.sh",
                "reliable_control_delivery_consumed": True,
                "watch_patterns": [],
            },
        ],
        ids=["missing-pid", "zero-pid", "bad-bool", "bad-tombstone"],
    )
    def test_semantically_invalid_checkpoint_aborts_without_rewrite(
        self, tmp_path, monkeypatch, entry
    ):
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        raw = json.dumps([entry])
        checkpoint.write_text(raw, encoding="utf-8")

        restarted = ProcessRegistry()
        with pytest.raises(ProcessCheckpointRecoveryError):
            restarted.recover_from_checkpoint()

        assert checkpoint.read_text(encoding="utf-8") == raw
        assert restarted.completion_queue.empty()
        assert restarted._running == {}

    def test_consumed_control_tombstone_may_omit_pid(self, tmp_path, monkeypatch):
        """The explicit suppression row is state, not a process to adopt."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        checkpoint.write_text(
            json.dumps(
                [
                    {
                        "session_id": "proc_consumed_tombstone",
                        "command": "agent-wait-job.sh",
                        "watch_patterns": ["FABLE_WAKE"],
                        "reliable_control_delivery_consumed": True,
                    }
                ]
            ),
            encoding="utf-8",
        )

        restarted = ProcessRegistry()
        assert restarted.recover_from_checkpoint() == 0
        assert restarted.completion_queue.empty()
        assert json.loads(checkpoint.read_text(encoding="utf-8")) == []

    def test_failed_primary_outbox_write_keeps_one_ram_wake_and_recovery_tombstone(
        self, tmp_path, monkeypatch
    ):
        """A captured marker is not durable until its outbox row exists."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        producer = ProcessRegistry()
        session = _make_session(sid="proc_primary_io_fail", started_at=2468.0)
        session.pid = 999999999
        session.host_start_time = 111
        session.watch_patterns = ["FABLE_WAKE"]
        session.notify_on_failure = True
        session.watcher_platform = "telegram"
        session.watcher_chat_id = "123"
        session.session_key = "agent:main:telegram:dm:123"
        session.parent_session_id = "sess-parent"
        producer._running[session.id] = session
        assert producer._write_checkpoint() is True

        with patch.object(
            producer, "_persist_reliable_watch_event", return_value=False
        ):
            producer._check_watch_patterns(
                session, "FABLE_WAKE reason=needs_attention\n"
            )
            session.exited = True
            session.exit_code = 0
            producer._move_to_finished(session)

        # The primary marker remains the only live event; completion must not
        # add a second fallback turn in this gateway lifecycle.
        assert producer.completion_queue.qsize() == 1
        primary = producer.completion_queue.get_nowait()
        assert primary["type"] == "watch_match"
        assert primary["pattern"] == "FABLE_WAKE"
        assert primary["checkpoint_confirmed"] is False
        assert session._reliable_control_watch_delivered is True
        assert session._reliable_control_watch_persisted is False

        persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert [item["session_id"] for item in persisted] == [session.id]
        assert persisted[0]["reliable_control_watch_delivered"] is False

        # An unrelated later snapshot from this same registry cannot erase the
        # last recovery source.
        assert producer._write_checkpoint() is True
        assert json.loads(checkpoint.read_text(encoding="utf-8"))[0][
            "session_id"
        ] == session.id

        restarted = ProcessRegistry()
        monkeypatch.setattr(restarted, "_host_pid_is_ours", lambda *_args: False)
        monkeypatch.setattr(restarted, "_is_host_pid_alive", lambda *_args: False)
        assert restarted.recover_from_checkpoint() == 0
        fallback = restarted.completion_queue.get_nowait()
        assert fallback["type"] == "watch_match"
        assert fallback["control_reason"] == "missing_reliable_control_sentinel"
        assert fallback["checkpoint_confirmed"] is True
        assert json.loads(checkpoint.read_text(encoding="utf-8")) == []

    def test_second_move_cannot_erase_checkpoint_before_fallback_persists(
        self, tmp_path, monkeypatch
    ):
        """A racing idempotent finalizer cannot publish an empty snapshot."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        registry = ProcessRegistry()
        session = _make_session(sid="proc_move_race", started_at=7654.0)
        session.pid = os.getpid()
        session.watch_patterns = ["FABLE_WAKE"]
        session.notify_on_failure = True
        session.watcher_platform = "telegram"
        session.watcher_chat_id = "123"
        registry._running[session.id] = session
        assert registry._write_checkpoint() is True

        persist_entered = threading.Event()
        allow_persist = threading.Event()
        real_persist = registry._persist_reliable_watch_event

        def blocked_persist(event):
            persist_entered.set()
            assert allow_persist.wait(5)
            return real_persist(event)

        session.exited = True
        session.exit_code = 0
        mover = threading.Thread(target=registry._move_to_finished, args=(session,))
        second_started = threading.Event()
        second_done = threading.Event()

        def second_move():
            second_started.set()
            registry._move_to_finished(session)
            second_done.set()

        second_mover = threading.Thread(target=second_move)
        with patch.object(
            registry, "_persist_reliable_watch_event", side_effect=blocked_persist
        ):
            mover.start()
            try:
                assert persist_entered.wait(5)
                second_mover.start()
                assert second_started.wait(5)
                # The first mover has not committed its outbox yet. The second
                # call must be a true serialized no-op and leave the original
                # recovery row on disk.
                persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
                assert [item["session_id"] for item in persisted] == [session.id]
                assert not second_done.is_set()
            finally:
                allow_persist.set()
                mover.join(timeout=5)
                second_mover.join(timeout=5)

        assert not mover.is_alive()
        assert not second_mover.is_alive()
        assert second_done.is_set()
        assert registry.completion_queue.qsize() == 1
        assert json.loads(checkpoint.read_text(encoding="utf-8")) == []
        outbox = tmp_path / "process_notifications.json"
        assert len(json.loads(outbox.read_text(encoding="utf-8"))) == 1

    def test_unrelated_checkpoint_cannot_erase_exited_session_mid_finalization(
        self, tmp_path, monkeypatch
    ):
        """The process row survives until fallback durability is committed."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        registry = ProcessRegistry()
        exiting = _make_session(sid="proc_finalizing", started_at=8766.0)
        exiting.pid = os.getpid()
        exiting.watch_patterns = ["FABLE_WAKE"]
        exiting.notify_on_failure = True
        exiting.watcher_platform = "telegram"
        exiting.watcher_chat_id = "123"
        unrelated = _make_session(sid="proc_unrelated", started_at=8767.0)
        unrelated.pid = os.getpid()
        registry._running[exiting.id] = exiting
        registry._running[unrelated.id] = unrelated
        assert registry._write_checkpoint() is True

        persist_entered = threading.Event()
        allow_persist = threading.Event()
        real_persist = registry._persist_reliable_watch_event

        def blocked_persist(event):
            persist_entered.set()
            assert allow_persist.wait(5)
            return real_persist(event)

        exiting.exited = True
        exiting.exit_code = 0
        mover = threading.Thread(
            target=registry._move_to_finished,
            args=(exiting,),
        )
        with patch.object(
            registry, "_persist_reliable_watch_event", side_effect=blocked_persist
        ):
            mover.start()
            try:
                assert persist_entered.wait(5)
                # Simulate a spawn/checkpoint for another session while A is
                # between exit observation and its durable outbox commit.
                assert registry._write_checkpoint() is True
                crash_snapshot = json.loads(checkpoint.read_text(encoding="utf-8"))
                assert {item["session_id"] for item in crash_snapshot} == {
                    exiting.id,
                    unrelated.id,
                }
                assert not (tmp_path / "process_notifications.json").exists()
            finally:
                allow_persist.set()
                mover.join(timeout=5)

        assert not mover.is_alive()
        assert {item["session_id"] for item in json.loads(
            checkpoint.read_text(encoding="utf-8")
        )} == {unrelated.id}
        assert len(json.loads(
            (tmp_path / "process_notifications.json").read_text(encoding="utf-8")
        )) == 1

    def test_checkpoint_writer_cannot_reinsert_clean_close_after_remove_commit(
        self, tmp_path, monkeypatch
    ):
        """Terminal checkpoint removal atomically transfers running ownership."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        registry = ProcessRegistry()
        closing = _make_session(sid="proc_clean_close", started_at=8768.0)
        closing.pid = os.getpid()
        closing.watch_patterns = ["FABLE_WAKE"]
        closing.notify_on_failure = True
        closing.watcher_platform = "telegram"
        closing.watcher_chat_id = "123"
        closing._reliable_control_close_seen = True
        unrelated = _make_session(sid="proc_after_commit", started_at=8769.0)
        unrelated.pid = os.getpid()
        registry._running[closing.id] = closing
        registry._running[unrelated.id] = unrelated
        assert registry._write_checkpoint() is True

        remove_committed = threading.Event()
        allow_finalizer = threading.Event()
        real_write = registry._write_checkpoint

        def pause_after_remove_commit(*args, **kwargs):
            result = real_write(*args, **kwargs)
            if kwargs.get("remove_session_ids") == {closing.id}:
                remove_committed.set()
                assert allow_finalizer.wait(5)
            return result

        closing.exited = True
        closing.exit_code = 0
        mover = threading.Thread(
            target=registry._move_to_finished,
            args=(closing,),
        )
        with patch.object(
            registry, "_write_checkpoint", side_effect=pause_after_remove_commit
        ):
            mover.start()
            try:
                assert remove_committed.wait(5)
                assert closing.id not in registry._running
                # A later writer observes the atomic ownership transfer and
                # cannot resurrect the clean terminal row.
                assert real_write() is True
                persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
                assert {item["session_id"] for item in persisted} == {
                    unrelated.id
                }
            finally:
                allow_finalizer.set()
                mover.join(timeout=5)

        assert not mover.is_alive()
        assert registry.completion_queue.empty()
        assert not (tmp_path / "process_notifications.json").exists()

    def test_primary_persist_serializes_with_consumption_before_outbox_write(
        self, tmp_path, monkeypatch
    ):
        """Consume cannot clear its tombstone ahead of an in-flight producer."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        registry = ProcessRegistry()
        session = _make_session(sid="proc_primary_race", started_at=8765.0)
        session.pid = os.getpid()
        session.watch_patterns = ["FABLE_WAKE"]
        session.notify_on_failure = True
        session.watcher_platform = "telegram"
        session.watcher_chat_id = "123"
        registry._running[session.id] = session
        assert registry._write_checkpoint() is True

        persist_entered = threading.Event()
        allow_persist = threading.Event()
        real_persist = registry._persist_reliable_watch_event

        def pause_then_persist(event):
            persist_entered.set()
            assert allow_persist.wait(5)
            return real_persist(event)

        scanner = threading.Thread(
            target=registry._check_watch_patterns,
            args=(session, "FABLE_WAKE reason=attention\n"),
        )
        consume_done = threading.Event()

        def consume_exit():
            registry._move_to_finished(session, consume_output=True)
            consume_done.set()

        consumer = threading.Thread(target=consume_exit)
        with patch.object(
            registry, "_persist_reliable_watch_event", side_effect=pause_then_persist
        ):
            scanner.start()
            try:
                assert persist_entered.wait(5)
                session.exited = True
                session.exit_code = 0
                consumer.start()
                # Primary persistence owns session._lock across its durable
                # transaction. Consumption cannot observe an empty outbox and
                # clear its tombstone while that producer is paused.
                assert consume_done.wait(0.1) is False
            finally:
                allow_persist.set()
                scanner.join(timeout=5)
                consumer.join(timeout=5)

        assert not scanner.is_alive()
        assert not consumer.is_alive()
        assert consume_done.is_set()
        assert registry.is_completion_consumed(session.id) is True
        assert json.loads(checkpoint.read_text(encoding="utf-8")) == []
        outbox = tmp_path / "process_notifications.json"
        assert json.loads(outbox.read_text(encoding="utf-8")) == []
        while not registry.completion_queue.empty():
            assert registry.is_notification_consumed(
                registry.completion_queue.get_nowait()
            ) is True

        restarted = ProcessRegistry()
        assert restarted.recover_from_checkpoint() == 0
        assert restarted.completion_queue.empty()

    def test_dead_pid_without_sentinel_recovers_as_durable_control_wake(
        self, tmp_path, monkeypatch
    ):
        """Gateway-down process exit cannot erase the selective fail-safe."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        producer = ProcessRegistry()
        session = _make_session(sid="proc_lost", started_at=4321.0)
        session.pid = 999999999
        session.host_start_time = 111
        session.watch_patterns = ["FABLE_WAKE"]
        session.notify_on_failure = True
        session.watcher_platform = "telegram"
        session.watcher_chat_id = "123"
        session.session_key = "agent:main:telegram:dm:123"
        session.parent_session_id = "sess-parent"
        producer._running[session.id] = session
        assert producer._write_checkpoint() is True

        restarted = ProcessRegistry()
        monkeypatch.setattr(restarted, "_host_pid_is_ours", lambda *_args: False)
        monkeypatch.setattr(restarted, "_is_host_pid_alive", lambda *_args: False)
        assert restarted.recover_from_checkpoint() == 0
        event = restarted.completion_queue.get_nowait()
        assert event["type"] == "watch_match"
        assert event["pattern"] == "FABLE_WAKE"
        assert event["control_reason"] == "missing_reliable_control_sentinel"
        assert event["parent_session_id"] == "sess-parent"
        assert event["checkpoint_confirmed"] is True
        assert json.loads(checkpoint.read_text()) == []

        # Crash again before adapter acceptance: the outbox alone replays it.
        replay = ProcessRegistry()
        assert replay.recover_from_checkpoint() == 0
        replayed = replay.completion_queue.get_nowait()
        assert replayed["delivery_id"] == event["delivery_id"]
        assert replayed["checkpoint_confirmed"] is True

    def test_unrecoverable_sandbox_watcher_generates_durable_control_wake(
        self, tmp_path, monkeypatch
    ):
        """A sandbox-local PID is unobservable after restart, not disposable."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        producer = ProcessRegistry()
        session = _make_session(sid="proc_sandbox_lost", started_at=5555.0)
        session.pid = 77
        session.pid_scope = "sandbox"
        session.watch_patterns = ["FABLE_WAKE"]
        session.notify_on_failure = True
        session.watcher_platform = "telegram"
        session.watcher_chat_id = "123"
        session.session_key = "agent:main:telegram:dm:123"
        session.parent_session_id = "sess-parent"
        producer._running[session.id] = session
        assert producer._write_checkpoint() is True

        restarted = ProcessRegistry()
        monkeypatch.setattr(
            restarted,
            "_host_pid_is_ours",
            lambda *_args: pytest.fail("sandbox PID must not be probed on host"),
        )
        assert restarted.recover_from_checkpoint() == 0

        event = restarted.completion_queue.get_nowait()
        assert event["type"] == "watch_match"
        assert event["pattern"] == "FABLE_WAKE"
        assert event["control_reason"] == "missing_reliable_control_sentinel"
        assert event["parent_session_id"] == "sess-parent"
        assert event["checkpoint_confirmed"] is True
        assert json.loads(checkpoint.read_text(encoding="utf-8")) == []
        outbox = tmp_path / "process_notifications.json"
        assert json.loads(outbox.read_text(encoding="utf-8"))[0][
            "delivery_id"
        ] == event["delivery_id"]

    def test_sandbox_fallback_persist_failure_retains_checkpoint_tombstone(
        self, tmp_path, monkeypatch
    ):
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        producer = ProcessRegistry()
        session = _make_session(sid="proc_sandbox_retry", started_at=6666.0)
        session.pid = 88
        session.pid_scope = "sandbox"
        session.watch_patterns = ["FABLE_WAKE"]
        session.notify_on_failure = True
        session.watcher_platform = "telegram"
        session.watcher_chat_id = "123"
        producer._running[session.id] = session
        assert producer._write_checkpoint() is True

        restarted = ProcessRegistry()
        with patch.object(
            restarted, "_persist_reliable_watch_event", return_value=False
        ), patch.object(
            restarted,
            "_host_pid_is_ours",
            side_effect=AssertionError("sandbox PID ownership probe"),
        ):
            assert restarted.recover_from_checkpoint() == 0

        event = restarted.completion_queue.get_nowait()
        assert event["checkpoint_confirmed"] is False
        persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert [item["session_id"] for item in persisted] == [session.id]
        assert persisted[0]["reliable_control_watch_delivered"] is False

        # Tombstone is registry state, not a one-write extra that disappears on
        # the next checkpoint publication.
        assert restarted._write_checkpoint() is True
        assert json.loads(checkpoint.read_text(encoding="utf-8"))[0][
            "session_id"
        ] == session.id

    def test_dead_pid_after_auto_close_wakes_when_exit_status_was_lost(
        self, tmp_path, monkeypatch
    ):
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        producer = ProcessRegistry()
        session = _make_session(sid="proc_closed", started_at=9876.0)
        session.pid = 999999999
        session.host_start_time = 111
        session.watch_patterns = ["FABLE_WAKE"]
        session.notify_on_failure = True
        session.watcher_platform = "telegram"
        session.watcher_chat_id = "123"
        session._reliable_control_close_seen = True
        producer._running[session.id] = session
        assert producer._write_checkpoint() is True

        restarted = ProcessRegistry()
        monkeypatch.setattr(restarted, "_host_pid_is_ours", lambda *_args: False)
        monkeypatch.setattr(restarted, "_is_host_pid_alive", lambda *_args: False)
        assert restarted.recover_from_checkpoint() == 0
        event = restarted.completion_queue.get_nowait()
        assert event["pattern"] == "FABLE_WAKE"
        assert event["control_reason"] == "checkpoint_exit_status_unavailable"
        assert event["checkpoint_confirmed"] is True
        assert json.loads(checkpoint.read_text()) == []

    def test_recover_dead_pid(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_dead",
            "command": "sleep 999",
            "pid": 999999999,  # almost certainly not running
            "task_id": "t1",
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 0

    def test_recover_dead_wrapper_retries_unreaped_systemd_scope(
        self, registry, tmp_path, monkeypatch
    ):
        checkpoint = tmp_path / "procs.json"
        entry = {
            "session_id": "proc_dead_scope",
            "command": "daemonize",
            "pid": 999999999,
            "pid_scope": "host",
            "host_start_time": 123.0,
            "systemd_unit": "hermes-worker-proc_dead_scope.scope",
        }
        checkpoint.write_text(json.dumps([entry]))
        monkeypatch.setattr(registry, "_host_pid_is_ours", lambda *_args: False)
        monkeypatch.setattr(registry, "_is_host_pid_alive", lambda *_args: False)

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint), patch(
            "tools.process_registry._stop_systemd_unit", return_value=False
        ) as stop_unit:
            assert registry.recover_from_checkpoint() == 0

        stop_unit.assert_called_once_with(entry["systemd_unit"])
        retained = json.loads(checkpoint.read_text())
        assert len(retained) == 1
        assert {
            key: value
            for key, value in retained[0].items()
            if key != "checkpoint_owner_id"
        } == entry
        assert retained[0]["checkpoint_owner_id"] == registry._checkpoint_owner_id

    def test_unreaped_systemd_scope_emits_one_durable_failure_wake(
        self, tmp_path, monkeypatch
    ):
        """Scope cleanup failure wakes once, even after a clean sentinel."""
        import tools.process_registry as pr_module

        checkpoint = tmp_path / "procs.json"
        monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", checkpoint)
        entry = {
            "session_id": "proc_dead_fable_scope",
            "command": "agent-wait-job.sh",
            "pid": 999999999,
            "pid_scope": "host",
            "host_start_time": 123.0,
            "systemd_unit": "hermes-worker-proc_dead_fable_scope.scope",
            "started_at": 9999.0,
            "watcher_platform": "telegram",
            "watcher_chat_id": "123",
            "notify_on_failure": True,
            "watch_patterns": ["FABLE_WAKE"],
            "reliable_control_watch_delivered": False,
            "reliable_control_close_seen": True,
        }
        checkpoint.write_text(json.dumps([entry]))

        first = ProcessRegistry()
        monkeypatch.setattr(first, "_host_pid_is_ours", lambda *_args: False)
        monkeypatch.setattr(first, "_is_host_pid_alive", lambda *_args: False)
        with patch(
            "tools.process_registry._stop_systemd_unit", return_value=False
        ) as stop_unit:
            assert first.recover_from_checkpoint() == 0
        stop_unit.assert_called_once_with(entry["systemd_unit"])

        event = first.completion_queue.get_nowait()
        assert event["pattern"] == "FABLE_WAKE"
        assert event["control_reason"] == "checkpoint_scope_reap_failed"
        assert event["checkpoint_confirmed"] is True
        retained = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert retained[0]["reliable_control_watch_delivered"] is True
        assert retained[0]["reliable_control_close_seen"] is True
        assert first.acknowledge_watch_event(event) is True

        # The cleanup row remains retryable, but the acknowledged control edge
        # must not fire again on every startup while systemctl remains broken.
        second = ProcessRegistry()
        monkeypatch.setattr(second, "_host_pid_is_ours", lambda *_args: False)
        monkeypatch.setattr(second, "_is_host_pid_alive", lambda *_args: False)
        with patch(
            "tools.process_registry._stop_systemd_unit", return_value=False
        ):
            assert second.recover_from_checkpoint() == 0
        assert second.completion_queue.empty()
        assert json.loads(checkpoint.read_text(encoding="utf-8"))[0][
            "reliable_control_watch_delivered"
        ] is True

    def test_recover_dead_wrapper_drops_reaped_systemd_scope(
        self, registry, tmp_path, monkeypatch
    ):
        checkpoint = tmp_path / "procs.json"
        entry = {
            "session_id": "proc_dead_scope",
            "command": "daemonize",
            "pid": 999999999,
            "pid_scope": "host",
            "host_start_time": 123.0,
            "systemd_unit": "hermes-worker-proc_dead_scope.scope",
        }
        checkpoint.write_text(json.dumps([entry]))
        monkeypatch.setattr(registry, "_host_pid_is_ours", lambda *_args: False)
        monkeypatch.setattr(registry, "_is_host_pid_alive", lambda *_args: False)

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint), patch(
            "tools.process_registry._stop_systemd_unit", return_value=True
        ) as stop_unit:
            assert registry.recover_from_checkpoint() == 0

        stop_unit.assert_called_once_with(entry["systemd_unit"])
        assert json.loads(checkpoint.read_text()) == []


    def test_recovery_skips_explicit_sandbox_backed_entries(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        original = [{
            "session_id": "proc_remote",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "t1",
            "pid_scope": "sandbox",
        }]
        checkpoint.write_text(json.dumps(original))

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 0
            assert registry.get("proc_remote") is None

            data = json.loads(checkpoint.read_text())
            assert data == []

    def test_checkpoint_redacts_command_with_inline_secret(self, registry, tmp_path):
        """Issue #77484: the checkpoint file persists raw commands; inline
        credentials (e.g. ``curl -H 'Authorization: Bearer sk-...'``) must be
        redacted before write. Recovery only uses command for display/logging
        (the process is already running), so masking is lossless."""
        checkpoint = tmp_path / "procs.json"
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            secret = "sk-secret1234567890"
            command = f"curl -H 'Authorization: Bearer {secret}' http://x"
            s = _make_session(sid="proc_secret", command=command)
            s.pid = 12345
            s.host_start_time = int(time.time())
            registry._running[s.id] = s
            registry._write_checkpoint()

            data = json.loads(checkpoint.read_text())
            assert data[0]["session_id"] == "proc_secret"
            assert secret not in data[0]["command"]
            assert data[0]["command"] != command

# =========================================================================
# Kill process
# =========================================================================

class TestKillProcess:
    def test_kill_already_exited(self, registry):
        s = _make_session(exited=True, exit_code=0)
        registry._finished[s.id] = s
        result = registry.kill_process(s.id)
        assert result["status"] == "already_exited"


    def test_kill_detached_session_uses_host_pid(self, registry):
        s = _make_session(sid="proc_detached", command="sleep 999")
        s.pid = 424242
        s.detached = True
        registry._running[s.id] = s

        terminate_calls = []

        class FakeProcess:
            def __init__(self, pid):
                self.pid = pid
            def children(self, recursive=False):
                return []
            def terminate(self):
                terminate_calls.append(("terminate", self.pid))

        import psutil as _psutil

        try:
            # Post-#21561: liveness probe routes through
            # ``ProcessRegistry._is_host_pid_alive`` (→
            # ``gateway.status._pid_exists``), and the actual kill on POSIX
            # routes through ``psutil.Process(pid).terminate()``. Neither
            # touches ``os.kill`` directly. Mock both seams.  Disable the
            # SIGKILL-escalation step (grace=0) so it doesn't call
            # ``psutil.wait_procs`` on the FakeProcess.
            with patch("gateway.status._pid_exists", return_value=True), \
                 patch.object(ProcessRegistry, "_daemon_term_grace_seconds",
                              staticmethod(lambda: 0.0)), \
                 patch.object(_psutil, "Process", side_effect=lambda pid: FakeProcess(pid)):
                result = registry.kill_process(s.id)

            assert result["status"] == "killed"
            assert ("terminate", 424242) in terminate_calls
        finally:
            registry._running.pop(s.id, None)


# =========================================================================
# Tool handler
# =========================================================================

class TestProcessToolHandler:
    def test_unknown_action(self):
        from tools.process_registry import _handle_process
        result = json.loads(_handle_process({"action": "unknown_action"}))
        assert "error" in result


# =========================================================================
# format_process_notification + drain_notifications (shared helpers)
# =========================================================================

from tools.process_registry_notifications import format_process_notification


def test_drain_notifications_completion_callback_exception_fails_closed(registry):
    event = {
        "type": "completion",
        "session_id": "proc_callback_error",
        "session_key": "session-a",
        "command": "safe-test-command",
        "exit_code": 0,
        "output": "done",
    }
    registry.completion_queue.put(event)

    def broken(_event):
        raise RuntimeError("ownership check exploded")

    results = registry.drain_notifications(
        session_key="session-a",
        owns_event=broken,
    )

    assert results == []
    assert registry.completion_queue.get_nowait() == event
    assert registry.completion_queue.empty()


def test_drain_notifications_filters_async_delegation_by_session_key():
    """Async-delegation events should only be consumed by the matching session's drain.

    Regression test for issue #58684: background delegation results delivered
    to the wrong session when the user switches sessions while a subagent runs.
    """
    from tools.process_registry import process_registry

    # Clear the queue first
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    try:
        # Put events for different sessions
        process_registry.completion_queue.put({
            "type": "async_delegation",
            "delegation_id": "deleg_session_a",
            "session_key": "telegram:dm:111:user_a",
            "goal": "task A",
            "status": "completed",
            "summary": "done A",
            "api_calls": 1,
            "duration_seconds": 0.5,
        })
        process_registry.completion_queue.put({
            "type": "async_delegation",
            "delegation_id": "deleg_session_b",
            "session_key": "telegram:dm:222:user_b",
            "goal": "task B",
            "status": "completed",
            "summary": "done B",
            "api_calls": 1,
            "duration_seconds": 0.3,
        })

        # Drain for session A — should only get deleg_session_a
        results_a = process_registry.drain_notifications(session_key="telegram:dm:111:user_a")
        assert len(results_a) == 1, (
            f"Expected 1 event for session A, got {len(results_a)}"
        )
        assert results_a[0][0]["delegation_id"] == "deleg_session_a"
        assert "done A" in results_a[0][1]

        # Session B's event should have been re-queued — drain for session B
        results_b = process_registry.drain_notifications(session_key="telegram:dm:222:user_b")
        assert len(results_b) == 1, (
            f"Expected 1 event for session B, got {len(results_b)}"
        )
        assert results_b[0][0]["delegation_id"] == "deleg_session_b"
        assert "done B" in results_b[0][1]

        # No more events should remain
        assert process_registry.completion_queue.empty()
    finally:
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def test_drain_notifications_owns_event_callback_beats_key_equality():
    """The positive-proof ownership callback consumes ONLY approved events —
    including across a compression rotation where bare key equality would
    wrongly re-queue the session's own pre-compression dispatch (#55578)."""
    from tools.process_registry import process_registry

    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    try:
        # Pre-compression dispatch: event carries the OLD key.
        process_registry.completion_queue.put({
            "type": "async_delegation",
            "delegation_id": "deleg_precompress",
            "session_key": "old_parent_key",
            "goal": "task", "status": "completed", "summary": "mine",
            "api_calls": 1, "duration_seconds": 0.1,
        })
        # Foreign event that plain key equality would also reject.
        process_registry.completion_queue.put({
            "type": "async_delegation",
            "delegation_id": "deleg_foreign",
            "session_key": "someone_else",
            "goal": "task", "status": "completed", "summary": "not mine",
            "api_calls": 1, "duration_seconds": 0.1,
        })

        # Chain-aware ownership: this session's lineage includes old_parent_key.
        lineage = {"old_parent_key", "new_child_key"}
        results = process_registry.drain_notifications(
            session_key="new_child_key",
            owns_event=lambda e: e.get("session_key") in lineage,
        )
        assert [r[0]["delegation_id"] for r in results] == ["deleg_precompress"]

        # The foreign event was re-queued, not consumed.
        leftover = process_registry.completion_queue.get_nowait()
        assert leftover["delegation_id"] == "deleg_foreign"
    finally:
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


# ---------------------------------------------------------------------------
# _terminate_host_pid — cross-platform process-tree termination
# ---------------------------------------------------------------------------


class TestTerminateHostPidWindows:
    """Windows branch uses ``taskkill /T /F`` — the documented MS tree-kill
    primitive. We can't use psutil's ``children(recursive=True)`` /
    ``.terminate()`` path on Windows because (1) Windows doesn't maintain
    a Unix-style process tree so the walk is unreliable, and (2)
    ``Process.terminate()`` on Windows is ``TerminateProcess()`` for the
    target handle only, not the tree.
    """

    @pytest.mark.windows_only
    def test_windows_invokes_taskkill_with_tree_and_force_flags(self, monkeypatch):
        """The Windows branch must shell out to ``taskkill /PID N /T /F``.

        Windows-only: ``taskkill.exe`` is the thing under test and only exists
        here — with a faked ``_IS_WINDOWS`` the argv was asserted against a
        binary that could never have run.
        """
        from tools import process_registry as pr

        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return MagicMock(returncode=0, stderr="", stdout="")

        monkeypatch.setattr(pr.subprocess, "run", fake_run)

        pr.ProcessRegistry._terminate_host_pid(12345)

        assert captured["args"][0] == "taskkill"
        assert "/PID" in captured["args"]
        assert "12345" in captured["args"]
        assert "/T" in captured["args"], "Tree flag required to reach descendants"
        assert "/F" in captured["args"], "Force flag required for headless Chromium"

class TestTerminateHostPidPosix:
    """POSIX branch walks the tree via psutil and SIGTERMs children first."""

    def test_posix_walks_tree_and_terminates_children_then_parent(self, monkeypatch):
        from tools import process_registry as pr
        import psutil

        terminate_order = []

        class _FakeChild:
            def __init__(self, pid):
                self.pid = pid

            def terminate(self):
                terminate_order.append(self.pid)

        class _FakeParent:
            def __init__(self, pid):
                self.pid = pid

            def children(self, recursive=False):
                assert recursive is True
                return [_FakeChild(101), _FakeChild(102), _FakeChild(103)]

            def terminate(self):
                terminate_order.append(self.pid)

        monkeypatch.setattr(psutil, "Process", _FakeParent)
        # This test covers only the SIGTERM tree-walk ordering; disable the
        # SIGKILL-escalation step (which would call psutil.wait_procs on the
        # fakes) by setting the grace to 0.
        monkeypatch.setattr(pr.ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 0.0))

        pr.ProcessRegistry._terminate_host_pid(12345)

        assert terminate_order == [101, 102, 103, 12345], (
            "Children must be terminated before the parent"
        )

    def test_posix_oserror_falls_back_to_os_kill(self, monkeypatch):
        from tools import process_registry as pr
        import psutil

        def boom(pid):
            raise PermissionError("can't read /proc")

        kill_calls = []

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))

        monkeypatch.setattr(psutil, "Process", boom)
        monkeypatch.setattr(pr.os, "kill", fake_kill)

        pr.ProcessRegistry._terminate_host_pid(12345)

        assert kill_calls == [(12345, signal.SIGTERM)]


# =========================================================================
# PID-reuse guard — a recycled PID/PGID must never be signalled.
#
# Regression: once a background-session process exits and is reaped, the kernel
# can recycle its PID onto an unrelated process (observed in the wild landing on
# a desktop browser's session leader, whose whole tree we then SIGTERMed —
# Firefox dying at irregular intervals).  Identity is re-validated via the
# kernel start time captured at spawn before any signal is sent.
# =========================================================================

class TestPidReuseGuard:
    def test_terminate_refuses_when_start_time_mismatches(self, registry):
        """A live PID whose start time changed (recycled) is NOT killed."""
        proc = _spawn_python_sleep(30)
        try:
            real_start = ProcessRegistry._safe_host_start_time(proc.pid)
            assert real_start is not None, "no /proc start time on this platform?"
            # Simulate recycling: the recorded baseline no longer matches.
            registry._terminate_host_pid(proc.pid, expected_start=real_start + 1)
            # The process must still be alive — the guard refused to signal it.
            assert not _wait_until(lambda: proc.poll() is not None, timeout=0.3)
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait()


    def test_refresh_detached_marks_recycled_pid_exited(self, registry):
        """A detached session whose PID got recycled is moved to finished."""
        wrong_start = (ProcessRegistry._safe_host_start_time(os.getpid()) or 0) + 999
        s = _make_session(sid="proc_detached")
        s.pid = os.getpid()          # alive, but...
        s.pid_scope = "host"
        s.detached = True
        s.host_start_time = wrong_start  # ...identity no longer matches
        registry._running[s.id] = s
        refreshed = registry._refresh_detached_session(s)
        assert refreshed.exited is True
        assert s.id in registry._finished


@pytest.mark.skipif(sys.platform == "win32",
                    reason="POSIX SIGTERM→SIGKILL escalation; Windows uses taskkill /F")
class TestSigkillEscalation:
    """Bounded SIGTERM→SIGKILL escalation in _terminate_host_pid.

    A daemon that ignores/stalls on SIGTERM must be force-killed after the
    configured grace window so it can't leak indefinitely — while well-behaved
    processes still exit cleanly on SIGTERM and the recycled-PID guard is never
    bypassed.
    """

    # A process that traps SIGTERM (ignores it): only SIGKILL stops it.
    # It prints "ready" AFTER installing the handler so the parent never
    # signals it during the startup window (before SIG_IGN is in place).
    _TRAP = (
        "import signal, sys, time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "sys.stdout.write('ready\\n'); sys.stdout.flush();"
        "[time.sleep(0.2) for _ in iter(int, 1)]"
    )

    def _spawn_trap(self):
        proc = subprocess.Popen(
            [sys.executable, "-c", self._TRAP],
            stdout=subprocess.PIPE, text=True,
        )
        # Wait until the handler is installed before returning.
        line = proc.stdout.readline()
        assert line.strip() == "ready", "trap process failed to start"
        return proc

    def test_sigterm_ignoring_daemon_is_sigkilled(self, monkeypatch):
        monkeypatch.setattr(ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 0.3))
        proc = self._spawn_trap()
        try:
            ProcessRegistry._terminate_host_pid(proc.pid)
            assert _wait_until(lambda: proc.poll() is not None, timeout=4.0), \
                "SIGTERM-ignoring daemon should be SIGKILLed after grace"
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def test_escalation_does_not_bypass_recycled_pid_guard(self, monkeypatch):
        """A start-time mismatch must still spare the PID — no SIGTERM, no SIGKILL."""
        monkeypatch.setattr(ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 0.3))
        proc = self._spawn_trap()
        try:
            real_start = ProcessRegistry._safe_host_start_time(proc.pid)
            ProcessRegistry._terminate_host_pid(
                proc.pid, expected_start=(real_start or 0) + 1)
            assert not _wait_until(lambda: proc.poll() is not None, timeout=0.3)
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait()

    def test_grace_reader_floors_at_zero(self, monkeypatch):
        """A negative configured grace is clamped to 0 (no escalation)."""
        import hermes_cli.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "read_raw_config",
                            lambda: {"terminal": {"daemon_term_grace_seconds": -5}})
        assert ProcessRegistry._daemon_term_grace_seconds() == 0.0

    @pytest.mark.live_system_guard_bypass
    def test_entire_tree_is_sigkilled_not_just_parent(self, monkeypatch):
        """A SIGTERM-ignoring parent + children are ALL force-killed.

        Regression: an earlier implementation trusted psutil.wait_procs's
        gone/alive partition, which mis-partitioned across a parent/child tree
        and left survivors un-killed (flaky — sometimes the parent lived,
        sometimes a child). The escalation now re-probes every target directly.
        """
        import psutil
        # 2.0s grace (not 1.0): with three interpreters mid-startup on a
        # loaded runner, a 1s SIGTERM->partition window races child spawn and
        # is how a child PID escaped the live-system guard in CI.
        monkeypatch.setattr(ProcessRegistry, "_daemon_term_grace_seconds",
                            staticmethod(lambda: 2.0))
        # Parent spawns 2 children; all trap SIGTERM. Parent prints child pids
        # after the handler is installed.
        parent_src = (
            "import signal, subprocess, sys, time;"
            "child='import signal,time\\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
            "[time.sleep(0.2) for _ in iter(int,1)]';"
            "kids=[subprocess.Popen([sys.executable,'-c',child]) for _ in range(2)];"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            "sys.stdout.write(' '.join(str(k.pid) for k in kids)+'\\n'); sys.stdout.flush();"
            "[time.sleep(0.2) for _ in iter(int,1)]"
        )
        parent = subprocess.Popen([sys.executable, "-c", parent_src],
                                  stdout=subprocess.PIPE, text=True)
        # Bound the readline: if the parent wedges before printing, fail THIS
        # test with a clear message instead of letting the per-file timeout
        # SIGKILL the whole pytest process (opaque rc=124 in CI).
        import select as _select
        ready, _, _ = _select.select([parent.stdout], [], [], 20.0)
        assert ready, "parent process failed to print child pids within 20s"
        child_pids = [int(x) for x in parent.stdout.readline().split()]
        all_pids = [parent.pid] + child_pids
        try:
            ProcessRegistry._terminate_host_pid(parent.pid)

            def _pid_dead(p: int) -> bool:
                # A pid is "dead" for our purposes if it no longer exists OR
                # exists only as an unreaped zombie (already terminated, just
                # not reaped by its reparented parent yet). psutil can also
                # raise mid-probe if the pid vanishes between the existence
                # check and the status read — treat any such race as dead.
                try:
                    if not psutil.pid_exists(p):
                        return True
                    return not ProcessRegistry._proc_alive(psutil.Process(p))
                except Exception:
                    return True

            def _all_dead():
                return all(_pid_dead(p) for p in all_pids)

            # _terminate_host_pid SIGKILLs synchronously before returning, so
            # the kill signals are already delivered here. The only remaining
            # wait is the kernel tearing down 3 processes and the reparented
            # children transitioning to zombie — which can lag on a loaded CI
            # runner. Give a generous budget (matches the wait() test's 10s)
            # so this asserts the escalation BEHAVIOR, not the runner's
            # scheduling latency. The assertion itself never weakens: every
            # tree member must end up dead/zombie.
            assert _wait_until(_all_dead, timeout=15.0, interval=0.02), (
                "entire SIGTERM-ignoring tree (parent + children) must be SIGKILLed"
            )
        finally:
            for p in all_pids:
                try:
                    os.kill(p, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            parent.wait()


class TestHandleProcessRedaction:
    """`_handle_process` redacts background-process output before it reaches the
    model / session.db / CLI display — issue #43025.

    Mirrors the foreground `terminal` redaction so the two surfaces can't
    diverge. Env-dump commands (`printenv`/`env`) get the ENV-assignment pass
    so opaque tokens are masked; other commands stay on the code_file path.
    """

    def _setup(self, monkeypatch, command, output):
        import agent.redact as _r
        monkeypatch.setattr(_r, "_REDACT_ENABLED", True)
        from tools import process_registry as pr
        reg = ProcessRegistry()
        sess = _make_session(sid="proc_redact1", command=command)
        sess.output_buffer = output
        sess.exited = True
        sess.exit_code = 0
        reg._running.clear()
        reg._finished[sess.id] = sess
        reg._running[sess.id] = sess
        monkeypatch.setattr(pr, "process_registry", reg)
        return pr, sess

    def test_log_redacts_env_dump_opaque_token(self, monkeypatch):
        pr, sess = self._setup(
            monkeypatch, "printenv",
            "MY_SERVICE_TOKEN=abc123randomopaquetokenvalue999\nHOME=/home/u",
        )
        out = json.loads(pr._handle_process({"action": "log", "session_id": sess.id}))
        assert "abc123randomopaquetokenvalue999" not in out["output"]
        assert "HOME=/home/u" in out["output"]

    def test_poll_redacts_prefix_key(self, monkeypatch):
        pr, sess = self._setup(
            monkeypatch, "python app.py",
            "leaked OPENAI_API_KEY sk-proj-abc123def456ghi789jkl012 here",
        )
        out = json.loads(pr._handle_process({"action": "poll", "session_id": sess.id}))
        assert "abc123def456" not in out["output_preview"]

    def test_list_redacts_command_and_output(self, monkeypatch):
        """`process(action=list)` redacts command + output_preview — issue #77484.

        The list branch previously returned raw ``command[:200]`` and
        ``output_preview[-200:]`` with no redaction wrap, leaking inline
        secrets (unlike poll/log/wait/kill).
        """
        pr, sess = self._setup(
            monkeypatch, "curl -H 'Authorization: Bearer sk-abc123def456ghi789jkl012345'",
            "opaque token sk-proj-AAAABBBBCCCCDDDDEEEEFFFFGGGG output",
        )
        out = json.loads(pr._handle_process({"action": "list"}))
        assert len(out["processes"]) >= 1
        entry = out["processes"][0]
        assert "sk-abc123def456ghi789jkl012345" not in entry["command"]
        assert "sk-proj-AAAABBBBCCCCDDDDEEEEFFFFGGGG" not in entry["output_preview"]
        assert "curl" in entry["command"]

    def test_disabled_passes_through(self, monkeypatch):
        import agent.redact as _r
        monkeypatch.setattr(_r, "_REDACT_ENABLED", False)
        from tools import process_registry as pr
        reg = ProcessRegistry()
        sess = _make_session(sid="proc_redact2", command="printenv")
        sess.output_buffer = "CUSTOM_TOKEN=zzzopaque1234567890abcdef"
        sess.exited = True
        sess.exit_code = 0
        reg._running[sess.id] = sess
        monkeypatch.setattr(pr, "process_registry", reg)
        out = json.loads(pr._handle_process({"action": "log", "session_id": sess.id}))
        assert "zzzopaque1234567890abcdef" in out["output"]


# =========================================================================
# Reader loop: orphaned grandchild holding the stdout pipe (issue #68915)
# =========================================================================

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: select() on pipes")
class TestReaderLoopOrphanedPipe:
    """Regression tests for issue #68915.

    When an agent command backgrounds a long-lived process (``node server.js
    &``), the grandchild inherits the write end of the reader's stdout pipe.
    The direct bash child exits, but the pipe never EOFs — the old blocking
    ``read1()`` parked the reader thread forever, ``session.exited`` never
    flipped on its own, and ``notify_on_complete`` never fired. The reader
    must instead terminate shortly after the direct child exits, even while
    a descendant still holds the pipe open.
    """

    def test_reader_exits_when_orphan_holds_pipe(self, registry):
        """Reader loop must return promptly after the direct child exits,
        even though a backgrounded descendant keeps the pipe open."""
        proc = subprocess.Popen(
            ["sh", "-c", "echo started; sleep 30 & exit 0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            preexec_fn=os.setsid,
        )
        s = _make_session(sid="proc_orphan_reader")
        s.process = proc
        s.pid = proc.pid
        registry._running[s.id] = s

        done = threading.Event()

        def _run():
            registry._reader_loop(s)
            done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        try:
            # The direct child exits immediately; the reader must notice and
            # return well before the 30s descendant releases the pipe.
            assert done.wait(timeout=10.0), (
                "_reader_loop is still blocked on the orphan-held pipe "
                "(issue #68915) — session.exited would never flip and "
                "notify_on_complete would never fire"
            )
            assert s.exited is True
            assert s.exit_code == 0
            assert s.completion_reason == "exited"
            assert "started" in s.output_buffer
            assert s.id in registry._finished
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def test_reader_exit_fires_notify_on_complete(self, registry):
        """The autonomous completion notification must not depend on a
        poll()/wait() call when an orphan holds the pipe."""
        proc = subprocess.Popen(
            ["sh", "-c", "sleep 30 & echo bg-started"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            preexec_fn=os.setsid,
        )
        s = _make_session(sid="proc_orphan_notify")
        s.process = proc
        s.pid = proc.pid
        s.notify_on_complete = True
        registry._running[s.id] = s

        done = threading.Event()

        def _run():
            registry._reader_loop(s)
            done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        try:
            assert done.wait(timeout=10.0), (
                "_reader_loop blocked — completion notification lost (#68915)"
            )
            # Exactly one completion event must have been queued.
            item = registry.completion_queue.get_nowait()
            assert item["type"] == "completion"
            assert item["session_id"] == s.id
            assert item["exit_code"] == 0
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

# =========================================================================
# systemd cgroup isolation for gateway-spawned local executors (#70716)
# =========================================================================
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: systemd scopes")
class TestSystemdCgroupIsolation:
    """Verify spawn_local wraps the worker in ``systemd-run --user --scope``
    when running under a supervisor and systemd-run is available, and falls
    back to the legacy ``start_new_session`` path otherwise.

    Issue #70716: local background terminal executors inherit the gateway's
    cgroup, so an OOM in a memory-heavy worker lets systemd-oomd kill the
    ENTIRE gateway cgroup, taking down the messaging control plane.
    """

    @pytest.fixture()
    def _gateway_identity(self, monkeypatch):
        """Opt-in: mark this test as running AS the live gateway process."""
        monkeypatch.setenv("_HERMES_GATEWAY", "1")
        monkeypatch.setattr(
            "gateway.status.get_running_pid",
            lambda *, cleanup_stale=False: os.getpid(),
        )

    def _fake_popen_capture(self):
        """Return (fake_popen, captured) where captured["argv"] gets the
        argv passed to subprocess.Popen."""
        captured = {}

        def fake_popen(argv, **kwargs):
            captured["argv"] = list(argv)
            captured["start_new_session"] = kwargs.get("start_new_session")
            proc = MagicMock()
            proc.pid = 4321
            proc.stdout = iter([])
            proc.stdin = MagicMock()
            proc.poll.return_value = None
            return proc

        return fake_popen, captured

    @pytest.mark.linux_only
    def test_wraps_in_systemd_scope_when_supervisor_and_available(
        self, registry, monkeypatch, _gateway_identity
    ):
        """Under a supervisor with systemd-run available, the spawn argv is
        wrapped in ``systemd-run --user --scope --unit=hermes-worker-<id>``."""
        fake_popen, captured = self._fake_popen_capture()

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )
        # _build_systemd_scope_argv calls shutil.which — point it at a stub.
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch("threading.Thread", return_value=MagicMock()),
            patch.object(registry, "_write_checkpoint"),
        ):
            session = registry.spawn_local("echo hello", cwd="/tmp")

        argv = captured["argv"]
        assert argv[0] == "/usr/bin/systemd-run", argv
        assert "--user" in argv
        assert "--scope" in argv
        assert "--quiet" in argv, (
            "systemd-run argv must include --quiet (#70716 gap #3)"
        )
        assert "--unit" in argv
        unit_idx = argv.index("--unit")
        assert argv[unit_idx + 1].startswith("hermes-worker-"), argv
        assert argv[unit_idx + 1] == f"hermes-worker-{session.id}", (
            argv
        )  # _build_systemd_scope_argv uses bare name
        properties = [
            argv[index + 1]
            for index, value in enumerate(argv[:-1])
            if value == "--property"
        ]
        assert "MemoryAccounting=yes" in properties
        # systemd rejects OOMPolicy= on transient --scope units across the versions
        # users run (239/245/249, #102486); emitting it fails the probe and every
        # cron worker dispatch. MemoryMax + MemoryAccounting carry the isolation.
        assert not any(p.startswith("OOMPolicy=") for p in properties), properties
        memory_max = next(
            value for value in properties if value.startswith("MemoryMax=")
        )
        assert int(memory_max.split("=", 1)[1]) > 0
        # The original shell command must still be present at the tail,
        # after the ``--`` separator that prevents systemd-run from
        # interpreting command flags as its own.
        assert "--" in argv, "systemd-run argv must use -- to separate command"
        sep_idx = argv.index("--")
        assert "/bin/bash" in argv[sep_idx:]
        assert "set +m; echo hello" in argv[sep_idx:]
        # systemd-run --scope gives the worker a new cgroup but NOT a new
        # session (#70716 regression: start_new_session was False, so the
        # worker kept the parent's session + controlling terminal → SIGTTIN/
        # SIGTTOU stopped the TUI).  start_new_session=True gives systemd-run
        # (and the scoped worker below it) a private session.
        assert captured["start_new_session"] is True
        # The session must record the unit name so kill_process can stop it.
        assert session.systemd_unit == f"hermes-worker-{session.id}.scope"

    def test_falls_back_when_systemd_run_unavailable(self, registry, monkeypatch, _gateway_identity):
        """Under a supervisor but without systemd-run, fall back to the
        legacy ``start_new_session=True`` path (worker shares the gateway
        cgroup)."""
        fake_popen, captured = self._fake_popen_capture()

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: False,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch("threading.Thread", return_value=MagicMock()),
            patch.object(registry, "_write_checkpoint"),
        ):
            registry.spawn_local("echo hello", cwd="/tmp")

        argv = captured["argv"]
        # No systemd-run wrapping — direct shell invocation.
        assert argv == ["/bin/bash", "-lic", "set +m; echo hello"], argv
        assert captured["start_new_session"] is True

    def test_falls_back_when_not_under_supervisor(self, registry, monkeypatch):
        """CLI mode (no supervisor) must NOT wrap in a systemd scope even if
        systemd-run is available — isolation is a gateway concern."""
        fake_popen, captured = self._fake_popen_capture()

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: False,
        )

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch("threading.Thread", return_value=MagicMock()),
            patch.object(registry, "_write_checkpoint"),
        ):
            registry.spawn_local("echo hello", cwd="/tmp")

        argv = captured["argv"]
        assert argv == ["/bin/bash", "-lic", "set +m; echo hello"], argv
        assert captured["start_new_session"] is True

    @pytest.mark.parametrize("use_pty", [False, True])
    def test_inherited_systemd_marker_does_not_scope_interactive_cli(
        self, registry, monkeypatch, use_pty
    ):
        """A CLI inside a supervised terminal must keep workers off its tty.

        INVOCATION_ID is inherited by every descendant, so its presence
        alone must not activate the gateway-only systemd scope path.
        """
        monkeypatch.setenv("INVOCATION_ID", "herdr-service-inherited-marker")
        monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        if use_pty:
            from ptyprocess import PtyProcess

            fake_pty = MagicMock(pid=4321)
            with (
                patch.object(PtyProcess, "spawn", return_value=fake_pty) as pty_spawn,
                patch("threading.Thread", return_value=MagicMock()),
                patch.object(registry, "_write_checkpoint"),
            ):
                session = registry.spawn_local("codex", cwd="/tmp", use_pty=True)
            assert pty_spawn.call_args.args[0] == [
                "/bin/bash", "-lic", "set +m; codex",
            ]
        else:
            fake_popen, captured = self._fake_popen_capture()
            with (
                patch("subprocess.Popen", side_effect=fake_popen),
                patch("threading.Thread", return_value=MagicMock()),
                patch.object(registry, "_write_checkpoint"),
            ):
                session = registry.spawn_local("echo hello", cwd="/tmp")
            assert captured["argv"] == [
                "/bin/bash", "-lic", "set +m; echo hello",
            ]
            assert captured["start_new_session"] is True

        assert session.systemd_unit == ""

    @pytest.mark.parametrize("use_pty", [False, True])
    def test_inherited_gateway_tree_markers_do_not_scope_child_cli(
        self, registry, monkeypatch, use_pty
    ):
        """Gateway descendants are not the gateway process that owns the PID file.

        _HERMES_GATEWAY is inherited (and set by importing gateway.run), so
        both it and INVOCATION_ID may be present in a child process. The
        PID-ownership gate must still keep the scope path off.
        """
        monkeypatch.setenv("INVOCATION_ID", "inherited-systemd-marker")
        monkeypatch.setenv("_HERMES_GATEWAY", "1")
        monkeypatch.setattr(
            "gateway.status.get_running_pid",
            lambda *, cleanup_stale=False: os.getpid() + 1,
        )
        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        if use_pty:
            from ptyprocess import PtyProcess

            fake_pty = MagicMock(pid=4321)
            with (
                patch.object(PtyProcess, "spawn", return_value=fake_pty) as pty_spawn,
                patch("threading.Thread", return_value=MagicMock()),
                patch.object(registry, "_write_checkpoint"),
            ):
                session = registry.spawn_local("codex", cwd="/tmp", use_pty=True)
            assert pty_spawn.call_args.args[0] == [
                "/bin/bash", "-lic", "set +m; codex",
            ]
        else:
            fake_popen, captured = self._fake_popen_capture()
            with (
                patch("subprocess.Popen", side_effect=fake_popen),
                patch("threading.Thread", return_value=MagicMock()),
                patch.object(registry, "_write_checkpoint"),
            ):
                session = registry.spawn_local("echo hello", cwd="/tmp")
            assert captured["argv"] == [
                "/bin/bash", "-lic", "set +m; echo hello",
            ]
            assert captured["start_new_session"] is True

        assert session.systemd_unit == ""

    @pytest.mark.linux_only
    def test_systemd_post_spawn_failure_never_kills_gateway_process_group(
        self, registry, monkeypatch, _gateway_identity
    ):
        """Cleanup must not killpg: scope teardown is the authoritative path."""
        fake_popen, _captured = self._fake_popen_capture()
        fake_proc = fake_popen(["placeholder"])

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        broken_reader = MagicMock()
        broken_reader.start.side_effect = RuntimeError("reader failed")

        with patch("subprocess.Popen", return_value=fake_proc), \
            patch("threading.Thread", return_value=broken_reader), \
            patch("tools.process_registry._stop_systemd_unit", return_value=True) as stop_unit, \
            patch("os.killpg") as killpg, \
            patch.object(registry, "_write_checkpoint"):
            with pytest.raises(RuntimeError, match="reader failed"):
                registry.spawn_local("echo hello", cwd="/tmp")

        stop_unit.assert_called_once()
        assert stop_unit.call_args.args[0].startswith("hermes-worker-proc_")
        assert stop_unit.call_args.args[0].endswith(".scope")
        killpg.assert_not_called()

    @pytest.mark.linux_only
    def test_pty_spawn_is_wrapped_in_systemd_scope(self, registry, monkeypatch, _gateway_identity):
        """Interactive executors receive the same sibling-cgroup isolation."""
        from ptyprocess import PtyProcess

        fake_pty = MagicMock()
        fake_pty.pid = 4321

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        with patch.object(PtyProcess, "spawn", return_value=fake_pty) as pty_spawn, \
            patch("threading.Thread", return_value=MagicMock()), \
            patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_local("codex", cwd="/tmp", use_pty=True)

        argv = pty_spawn.call_args.args[0]
        assert argv[0] == "/usr/bin/systemd-run"
        assert "--scope" in argv
        assert "--unit" in argv
        assert "--" in argv
        assert argv[-3:] == ["/bin/bash", "-lic", "set +m; codex"]
        assert session.systemd_unit == f"hermes-worker-{session.id}.scope"

    @pytest.mark.linux_only
    def test_pty_spawn_failure_reaps_scope_before_distinct_pipe_fallback(
        self, registry, monkeypatch, _gateway_identity
    ):
        """A failed PTY scope must not collide with the pipe fallback scope."""
        from ptyprocess import PtyProcess

        events = []
        fake_proc = MagicMock()
        fake_proc.pid = 4321
        fake_proc.stdout = iter([])
        fake_proc.stdin = MagicMock()
        fake_proc.poll.return_value = None

        def fake_popen(argv, **_kwargs):
            events.append(("pipe", list(argv)))
            return fake_proc

        def fake_stop(unit_name):
            events.append(("stop", unit_name))
            return True

        def fail_pty(*_args, **_kwargs):
            events.append(("pty", None))
            raise RuntimeError("PTY wrapper failed after scope creation")

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        with patch.object(PtyProcess, "spawn", side_effect=fail_pty), \
            patch("subprocess.Popen", side_effect=fake_popen), \
            patch("tools.process_registry._stop_systemd_unit", side_effect=fake_stop), \
            patch("threading.Thread", return_value=MagicMock()), \
            patch.object(registry, "_write_checkpoint"):
            session = registry.spawn_local("codex", cwd="/tmp", use_pty=True)

        assert [event[0] for event in events] == ["pty", "stop", "pipe"]
        stopped_unit = events[1][1]
        fallback_argv = events[2][1]
        assert stopped_unit == f"hermes-worker-{session.id}.scope"
        unit_idx = fallback_argv.index("--unit")
        assert fallback_argv[unit_idx + 1] == (
            f"hermes-worker-{session.id}-pipe-fallback"
        )
        assert session.systemd_unit == (
            f"hermes-worker-{session.id}-pipe-fallback.scope"
        )

    @pytest.mark.linux_only
    def test_pty_spawn_failure_does_not_fallback_when_scope_reap_fails(
        self, registry, monkeypatch, _gateway_identity
    ):
        """Do not launch a duplicate command while the failed PTY scope may live."""
        from ptyprocess import PtyProcess

        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "tools.process_registry._systemd_run_user_scope_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process",
            lambda: True,
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        with patch.object(
            PtyProcess,
            "spawn",
            side_effect=RuntimeError("PTY wrapper failed after scope creation"),
        ), patch("subprocess.Popen") as pipe_spawn, patch(
            "tools.process_registry._stop_systemd_unit", return_value=False
        ) as stop_unit:
            with pytest.raises(RuntimeError, match="could not be reaped"):
                registry.spawn_local("codex", cwd="/tmp", use_pty=True)

        stop_unit.assert_called_once()
        pipe_spawn.assert_not_called()

    def test_worker_memory_limit_honors_local_guard_mb_override(self, monkeypatch):
        import tools.process_registry as pr

        monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "123")
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

        with patch("tools.process_registry.logger.warning") as warning:
            argv = pr._build_systemd_scope_argv(
                ["/bin/bash", "-lc", "true"],
                unit_suffix="test",
            )

        warning.assert_not_called()
        assert f"MemoryMax={123 * 1024 * 1024}" in argv

    def test_worker_memory_limit_caps_oversized_local_guard_override(
        self, monkeypatch
    ):
        import tools.process_registry as pr

        monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "999999")
        monkeypatch.setattr(
            pr.Path,
            "read_text",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no cgroup")),
        )
        monkeypatch.setattr(
            pr.os,
            "sysconf",
            lambda *_args: (_ for _ in ()).throw(OSError("no sysconf")),
        )

        assert pr._worker_memory_max_bytes() == pr._DEFAULT_WORKER_MEMORY_MAX_BYTES

    def test_kill_recovered_detached_already_exited_stops_persisted_scope(
        self, registry, monkeypatch
    ):
        """Recovered detached sessions whose wrapper PID is gone/recycled must
        still stop their persisted systemd scope before the already_exited
        return, while retaining the PID-reuse guard (no PID tree kill)."""
        session = _make_session(sid="proc_recovered_scope", command="daemonize")
        session.detached = True
        session.pid_scope = "host"
        session.pid = 12345
        session.host_start_time = 67890
        session.systemd_unit = "hermes-worker-proc_recovered_scope.scope"
        registry._running[session.id] = session

        stopped = []
        terminated = []
        monkeypatch.setattr(registry, "_host_pid_is_ours", lambda pid, start: False)
        monkeypatch.setattr(registry, "_terminate_host_pid", lambda pid, start: terminated.append((pid, start)))
        monkeypatch.setattr("tools.process_registry._stop_systemd_unit", lambda unit: stopped.append(unit) or True)

        with patch.object(registry, "_write_checkpoint"):
            result = registry.kill_process(session.id)

        assert result["status"] == "already_exited"
        assert stopped == ["hermes-worker-proc_recovered_scope.scope"]
        assert terminated == []
        assert session.exited is True
        assert session.id in registry._finished
        assert session.id not in registry._running

    @pytest.mark.linux_only
    def test_systemd_run_user_scope_available_caches_after_probe(
        self, registry, monkeypatch
    ):
        """The availability check probes once and caches — a second call must
        not re-probe (and must return the same value)."""
        import tools.process_registry as pr

        # Reset the cache.
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        probe_calls = []

        def fake_run(*args, **kwargs):
            probe_calls.append(args)
            return subprocess.CompletedProcess(args=args[0], returncode=0)

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        monkeypatch.setattr("subprocess.run", fake_run)

        first = pr._systemd_run_user_scope_available()
        second = pr._systemd_run_user_scope_available()
        assert first is True
        assert second is True
        assert len(probe_calls) == 1, "probe must run only once (cached)"
        # The probe must not carry OOMPolicy= either: that is the argv systemd
        # rejected on scope units and cached as "unavailable" (#102486).
        probe_argv = probe_calls[0][0]
        assert not any(
            value.startswith("OOMPolicy=") for value in probe_argv if isinstance(value, str)
        ), probe_argv

    @pytest.mark.linux_only
    def test_systemd_probe_derives_owned_user_bus_env_for_system_gateway(
        self, registry, monkeypatch, request
    ):
        """A system service running as an unprivileged user has no login env,
        but may still have a valid lingering user manager and D-Bus socket."""
        import socket
        import tempfile

        import tools.process_registry as pr

        # Short path: AF_UNIX socket paths are capped at ~104 bytes, longer than most tmp_path values.
        runtime_dir = pr.Path(tempfile.mkdtemp(prefix="hbus-", dir="/tmp"))
        runtime_dir.chmod(0o700)
        bus_path = runtime_dir / "bus"
        bus_socket = socket.socket(socket.AF_UNIX)
        bus_socket.bind(str(bus_path))

        def _cleanup():
            bus_socket.close()
            bus_path.unlink(missing_ok=True)
            runtime_dir.rmdir()

        request.addfinalizer(_cleanup)

        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr(pr, "_default_user_runtime_dir", lambda: runtime_dir)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        derived = pr.systemd_user_bus_env(
            {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/tmp/untrusted-bus"}
        )
        assert derived["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path={bus_path}"
        probe_kwargs = []

        def fake_run(*args, **kwargs):
            probe_kwargs.append(kwargs)
            return subprocess.CompletedProcess(args=args[0], returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)

        assert pr._systemd_run_user_scope_available() is True
        env = probe_kwargs[0]["env"]
        assert env["XDG_RUNTIME_DIR"] == str(runtime_dir)
        assert env["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path={bus_path}"
        assert "XDG_RUNTIME_DIR" not in os.environ
        assert "DBUS_SESSION_BUS_ADDRESS" not in os.environ

    @pytest.mark.linux_only
    def test_probe_succeeds_without_bin_true(self, monkeypatch):
        """An absent ``/bin/true`` must not make a usable scope fail its probe."""
        import tools.process_registry as pr

        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_PROBED_AT", 0.0)
        real_run = subprocess.run
        executed = []

        def systemd_run_on_nixos_shaped_root(argv, **kwargs):
            # Simulate NixOS's missing executable, but run the selected replacement.
            payload = argv[argv.index("--") + 1 :]
            if payload[0] == "/bin/true":
                return subprocess.CompletedProcess(payload, 127, stderr=b"No such file or directory")
            executed.append(payload)
            return real_run(payload, **kwargs)

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        monkeypatch.setattr("subprocess.run", systemd_run_on_nixos_shaped_root)

        assert pr._systemd_run_user_scope_available() is True
        assert len(executed) == 1, "payload must really run (exit 0) on the host, not just be spelled right"

    @pytest.mark.linux_only
    def test_systemd_scope_first_probe_is_serialized(self, monkeypatch):
        """Concurrent first-use callers must wait for one definitive probe.

        A temporary cached ``False`` would let a racing worker spawn inside the
        gateway cgroup, defeating the OOM isolation guarantee.
        """
        import tools.process_registry as pr

        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        probe_started = threading.Event()
        release_probe = threading.Event()
        probe_calls = []
        results = []

        def fake_run(*args, **kwargs):
            probe_calls.append(args)
            probe_started.set()
            assert release_probe.wait(timeout=2)
            return subprocess.CompletedProcess(args=args[0], returncode=0)

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        monkeypatch.setattr("subprocess.run", fake_run)

        first = threading.Thread(
            target=lambda: results.append(pr._systemd_run_user_scope_available())
        )
        second = threading.Thread(
            target=lambda: results.append(pr._systemd_run_user_scope_available())
        )
        first.start()
        assert probe_started.wait(timeout=2)
        second.start()

        # The racing caller must be blocked behind the probe, not observe a
        # temporary False cache value.
        second.join(timeout=0.05)
        assert second.is_alive()

        release_probe.set()
        first.join(timeout=2)
        second.join(timeout=2)

        assert not first.is_alive()
        assert not second.is_alive()
        assert results == [True, True]
        assert len(probe_calls) == 1

    @pytest.mark.linux_only
    def test_failed_systemd_probe_retries_after_cache_ttl(self, monkeypatch):
        import tools.process_registry as pr

        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_PROBED_AT", 0.0, raising=False)
        clock = [100.0]
        probe_results = [1, 0]
        probe_calls = []

        def fake_run(*args, **kwargs):
            probe_calls.append(args)
            return subprocess.CompletedProcess(
                args=args[0], returncode=probe_results.pop(0)
            )

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")
        monkeypatch.setattr("tools.process_registry.time.monotonic", lambda: clock[0])
        monkeypatch.setattr("subprocess.run", fake_run)

        assert pr._systemd_run_user_scope_available() is False
        assert pr._systemd_run_user_scope_available() is False
        assert len(probe_calls) == 1

        clock[0] += 61
        assert pr._systemd_run_user_scope_available() is True
        assert len(probe_calls) == 2

    def test_stop_systemd_unit_treats_absent_unit_as_clean(self, monkeypatch):
        import tools.process_registry as pr

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemctl")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args=args[0],
                returncode=5,
                stderr=b"Unit hermes-worker-gone.scope not loaded.\n",
            ),
        )

        assert pr._stop_systemd_unit("hermes-worker-gone.scope") is True

    def test_darwin_never_takes_scope_path_even_with_systemd_run_on_path(
        self, registry, monkeypatch, _gateway_identity
    ):
        """macOS no-op guarantee (#70716 cross-platform audit).

        With ``_IS_LINUX = False`` (darwin), the spawn path must be
        byte-identical to the legacy path even when a ``systemd-run``
        binary is somehow on PATH and the gateway identity checks pass:
        no probe, no wrapping, no unit recorded.
        """
        import tools.process_registry as pr

        fake_popen, captured = self._fake_popen_capture()

        monkeypatch.setattr(pr, "_IS_LINUX", False)
        monkeypatch.setattr(pr, "_IS_WINDOWS", False)
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr("tools.process_registry._find_shell", lambda: "/bin/bash")
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process", lambda: True
        )
        # If any branch consults the probe or builds a scope argv on darwin,
        # fail loudly.
        monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/systemd-run")
        scope_builds = []
        real_build = pr._build_systemd_scope_argv
        monkeypatch.setattr(
            pr,
            "_build_systemd_scope_argv",
            lambda *a, **k: scope_builds.append(a) or real_build(*a, **k),
        )
        probe_runs = []

        def fake_probe_run(argv, **kwargs):
            probe_runs.append(argv)
            return subprocess.CompletedProcess(args=argv, returncode=0)

        monkeypatch.setattr("subprocess.run", fake_probe_run)

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch("threading.Thread", return_value=MagicMock()),
            patch.object(registry, "_write_checkpoint"),
        ):
            session = registry.spawn_local("echo hello", cwd="/tmp")

        argv = captured["argv"]
        assert argv == ["/bin/bash", "-lic", "set +m; echo hello"], argv
        assert captured["start_new_session"] is True
        assert session.systemd_unit == ""
        assert scope_builds == [], "darwin must never build a systemd scope argv"
        assert probe_runs == [], "darwin must never run the systemd-run probe"

    def test_probe_returns_false_off_linux(self, monkeypatch):
        """``_systemd_run_user_scope_available`` is False on non-Linux even
        when a ``systemd-run`` binary exists on PATH."""
        import tools.process_registry as pr

        monkeypatch.setattr(pr, "_IS_LINUX", False)
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/systemd-run")
        probe_runs = []
        monkeypatch.setattr(
            "subprocess.run",
            lambda argv, **kwargs: probe_runs.append(argv)
            or subprocess.CompletedProcess(args=argv, returncode=0),
        )

        assert pr._systemd_run_user_scope_available() is False
        assert probe_runs == [], "non-Linux must not exec the probe"


class TestNotificationRedaction:
    """Background-process notification delivery (completion_queue) applies the
    same redaction as the explicit process tool — issue #43025 gap.

    The _move_to_finished() and _check_watch_patterns() paths enqueue raw
    output into the completion_queue.  After the fix, _redact_process_result()
    is called before enqueueing so secrets are masked in the [IMPORTANT: ...]
    messages delivered to the LLM.
    """

    def test_completion_notification_redacts_secret(self, monkeypatch):
        """_move_to_finished completion notification redacts API keys."""
        import agent.redact as _r
        monkeypatch.setattr(_r, "_REDACT_ENABLED", True)
        from tools import process_registry as pr

        reg = ProcessRegistry()
        sess = _make_session(sid="proc_notif1", command="env")
        sess.output_buffer = "OPENAI_API_KEY=sk-proj-secret123\nHOME=/home/u"
        sess.notify_on_complete = True
        sess.exited = True
        sess.exit_code = 0
        reg._running[sess.id] = sess
        monkeypatch.setattr(pr, "process_registry", reg)

        reg._move_to_finished(sess)

        # Drain and check the notification
        results = reg.drain_notifications()
        assert len(results) == 1
        _evt, text = results[0]
        assert "sk-proj-secret123" not in text
        assert "REDACTED" in text or "sk-proj" not in text

    def test_watch_match_notification_redacts_secret(self, monkeypatch):
        """_check_watch_patterns watch_match notification redacts secrets."""
        import agent.redact as _r
        monkeypatch.setattr(_r, "_REDACT_ENABLED", True)
        from tools import process_registry as pr

        reg = ProcessRegistry()
        sess = _make_session(sid="proc_notif2", command="python server.py")
        sess.output_buffer = "Server started\nAPI_TOKEN=ghp_abc123def456\nListening on :8080"
        sess.watch_patterns = ["API_TOKEN"]
        sess._watch_disabled = False
        sess._watch_hits = 0
        sess._watch_suppressed = 0
        sess.watcher_platform = None
        sess.watcher_chat_id = None
        sess.watcher_user_id = None
        sess.watcher_user_name = None
        sess.watcher_thread_id = None
        sess.watcher_message_id = None
        sess.exited = False
        reg._running[sess.id] = sess
        monkeypatch.setattr(pr, "process_registry", reg)

        reg._check_watch_patterns(sess, "API_TOKEN=ghp_abc123def456\n")

        results = reg.drain_notifications()
        assert len(results) == 1
        _evt, text = results[0]
        assert "ghp_abc123def456" not in text
        assert "ghp_" not in text or "REDACTED" in text


# ── Prefix resolution (Factory Droid-inspired task-ID prefixes) ──────────────


class TestGetByPrefix:
    """ProcessRegistry.get() resolves unique ID prefixes like git short hashes."""

    def test_full_id_still_exact(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6")
        registry._running[s.id] = s
        assert registry.get("proc_4dae56ca81f6") is s

    def test_unique_prefix_resolves(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6")
        registry._running[s.id] = s
        assert registry.get("proc_4dae5") is s

    def test_bare_suffix_resolves(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6")
        registry._running[s.id] = s
        assert registry.get("4dae56") is s

    def test_finished_sessions_also_resolve(self, registry):
        s = _make_session(sid="proc_9bee77aa0011", exited=True, exit_code=0)
        registry._finished[s.id] = s
        assert registry.get("proc_9bee") is s

    def test_ambiguous_prefix_returns_none(self, registry):
        a = _make_session(sid="proc_4dae56ca81f6")
        b = _make_session(sid="proc_4dae99999999")
        registry._running[a.id] = a
        registry._running[b.id] = b
        assert registry.get("proc_4dae") is None

    def test_too_short_prefix_returns_none(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6")
        registry._running[s.id] = s
        assert registry.get("proc_4da") is None
        assert registry.get("4da") is None
        assert registry.get("proc_") is None
        assert registry.get("") is None

    def test_exact_id_wins_over_prefix_scan(self, registry):
        # A session whose FULL id happens to be a prefix of another's must
        # resolve to itself, never trigger the ambiguity path.
        short = _make_session(sid="proc_4dae")
        long = _make_session(sid="proc_4dae56ca81f6")
        registry._running[short.id] = short
        registry._running[long.id] = long
        assert registry.get("proc_4dae") is short

    def test_no_match_returns_none(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6")
        registry._running[s.id] = s
        assert registry.get("proc_ffff") is None

    def test_poll_accepts_prefix(self, registry):
        s = _make_session(sid="proc_4dae56ca81f6", output="hello world")
        registry._running[s.id] = s
        result = registry.poll("4dae56ca")
        assert result["session_id"] == "proc_4dae56ca81f6"
        assert result["status"] == "running"


# ---------------------------------------------------------------------------
# Config-level model_not_found notice in delegation batch reports (#97654)
# ---------------------------------------------------------------------------


def _make_delegation_batch_evt(results):
    """A batch async-delegation event carrying a per-task ``results`` list."""
    return {
        "type": "async_delegation",
        "delegation_id": "deleg_97654",
        "is_batch": True,
        "results": results,
        "goals": [r.get("goal") or "" for r in results],
        "session_key": "agent:main:cli:dm:local",
        "status": "completed",
        "model": "upstage/solar-pro-4",
    }


def _patch_delegation_config(
    monkeypatch, model="upstage/solar-pro-4", provider="openrouter", **over
):
    import tools.process_registry_notifications as _prn

    cfg = {"model": model, "provider": provider}
    cfg.update(over)
    monkeypatch.setattr(_prn, "_delegation_config", lambda: cfg)
    return cfg


def _format_async(evt) -> str:
    from tools.process_registry_notifications import format_process_notification

    text = format_process_notification(evt)
    assert text is not None, "format_process_notification returned None"
    return text


def test_model_not_found_notice_single_failure_once(monkeypatch):
    evt = _make_delegation_batch_evt([
        {
            "task_index": 0,
            "status": "failed",
            "exit_reason": "error",
            "goal": "Create bridge module",
            "error": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
            "summary": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
        }
    ])
    _patch_delegation_config(monkeypatch)
    text = _format_async(evt)
    assert text is not None
    assert text.count("SUBAGENT MODEL REJECTED") == 1
    assert "upstage/solar-pro-4" in text
    assert "openrouter" in text
    assert "No fallback chain is configured" in text


def test_model_not_found_notice_mixed_batch_named_model(monkeypatch):
    evt = _make_delegation_batch_evt([
        {
            "task_index": 0,
            "status": "failed",
            "exit_reason": "error",
            "goal": "A",
            "error": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
            "summary": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
        },
        {
            "task_index": 1,
            "status": "completed",
            "goal": "B",
            "summary": "ok",
            "api_calls": 3,
        },
    ])
    _patch_delegation_config(monkeypatch)
    text = _format_async(evt)
    assert text.count("SUBAGENT MODEL REJECTED") == 1
    assert "upstage/solar-pro-4" in text


def test_model_not_found_notice_absent_for_non_model_errors(monkeypatch):
    evt = _make_delegation_batch_evt([
        {
            "task_index": 0,
            "status": "failed",
            "goal": "A",
            "error": "HTTP 429: rate limit exceeded",
        },
        {
            "task_index": 1,
            "status": "failed",
            "goal": "B",
            "error": "Connection timed out",
        },
    ])
    _patch_delegation_config(monkeypatch)
    text = _format_async(evt)
    assert "SUBAGENT MODEL REJECTED" not in text


def test_model_not_found_notice_absent_when_configured_model_not_named(monkeypatch):
    evt = _make_delegation_batch_evt([
        {
            "task_index": 0,
            "status": "failed",
            "goal": "A",
            "error": "HTTP 400: gpt-99 is not a valid model ID",
        }
    ])
    # Configured model is upstage/solar-pro-4; the rejection names gpt-99.
    _patch_delegation_config(monkeypatch)
    text = _format_async(evt)
    assert "SUBAGENT MODEL REJECTED" not in text


def test_model_not_found_notice_single_dispatch(monkeypatch):
    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_single",
        "session_key": "agent:main:cli:dm:local",
        "goal": "task A",
        "model": "upstage/solar-pro-4",
        "status": "failed",
        "error": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
        "summary": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
    }
    _patch_delegation_config(monkeypatch)
    text = _format_async(evt)
    assert text.count("SUBAGENT MODEL REJECTED") == 1
    assert "upstage/solar-pro-4" in text


def test_model_not_found_notice_absent_when_fallback_chain_configured(monkeypatch):
    evt = _make_delegation_batch_evt([
        {
            "task_index": 0,
            "status": "failed",
            "goal": "A",
            "error": "HTTP 400: upstage/solar-pro-4 is not a valid model ID",
        }
    ])
    _patch_delegation_config(
        monkeypatch,
        fallback_providers=[{"provider": "openrouter", "model": "upstage/solar-pro4"}],
    )
    text = _format_async(evt)
    assert text.count("SUBAGENT MODEL REJECTED") == 1
    assert "No fallback chain is configured" not in text
