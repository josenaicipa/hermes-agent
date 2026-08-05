"""Forward-progress plumbing shared by auxiliary providers.

This module is deliberately tiny and stdlib-only so that *both*
``agent.auxiliary_client`` and the CLI-backed adapters
(``agent.agy_cli_client``, ``agent.kimi_code_cli_client``) can import it at
module scope without any import cycle.

Two related concerns live here:

**The forward-progress hook.**  Long auxiliary calls (context compression is
the prime case) are watched by wall-clock deadlines in their hosts (gateway
session hygiene).  A fixed deadline punishes SLOW summary models exactly as
hard as HUNG ones.  This thread-local hook lets the host observe liveness
instead: stream consumers tick it per token/SSE event and the host extends
its deadline while output is moving.  Thread-local matches the call topology
— the aux call and its stream consumption run synchronously on the thread
that installed the hook.

**Liveness for buffered external-process providers.**  A CLI-backed provider
has no stream from Hermes' point of view: the child buffers and emits its
whole answer at the end (measured on the Kimi Code CLI: first stdout event at
106.7s of a 107.4s synthesis).  Such a call produces ZERO progress ticks, so
a caller-side token-inactivity watchdog (~30s) cancels a summary that is in
fact perfectly healthy.

:func:`external_process_liveness` fixes exactly that and nothing more:

* it runs only while a *verified child process* is alive — the caller
  supplies the liveness predicate and it is polled, never assumed;
* it stops the instant the process exits, so nothing can claim liveness for
  a finished (or crashed) call;
* it publishes PROCESS liveness only.  Total-time bounds are untouched: the
  provider's own subprocess timeout and the host's total ceiling (gateway
  ``hygiene_total_ceiling_seconds``, default 600s) still fire, so a genuinely
  hung child is still killed and still cancels;
* streaming providers never call it, so their token-inactivity semantics are
  unchanged.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_LIVENESS_INTERVAL_SECONDS",
    "external_process_liveness",
    "notify_progress",
    "progress_active",
    "progress_hook",
    "run_text_capture",
]

# Interval between liveness publications.  Well below the ~30s inactivity
# window Hermes' hosts use, and cheap: one predicate poll plus one callback.
DEFAULT_LIVENESS_INTERVAL_SECONDS = 5.0

_progress = threading.local()


# ── forward-progress hook ────────────────────────────────────────────────


def notify_progress() -> None:
    """Tick the installed forward-progress hook, if any.  Never raises."""
    hook = getattr(_progress, "hook", None)
    if hook is None:
        return
    try:
        hook()
    except Exception:
        logger.debug("aux progress hook failed", exc_info=True)


def progress_active() -> bool:
    """Whether this thread currently has a forward-progress hook installed."""
    return getattr(_progress, "hook", None) is not None


@contextlib.contextmanager
def progress_hook(hook):
    """Install *hook* as the current thread's aux forward-progress callback.

    ``hook=None`` is a no-op passthrough so callers can wire it
    unconditionally.  Re-entrant-safe: restores the previous hook on exit.
    """
    prev = getattr(_progress, "hook", None)
    _progress.hook = hook if callable(hook) else prev
    try:
        yield
    finally:
        _progress.hook = prev


# ── liveness for a verified child process ────────────────────────────────


@contextlib.contextmanager
def external_process_liveness(
    is_alive: Callable[[], bool],
    *,
    label: str = "external-process",
    interval: float = DEFAULT_LIVENESS_INTERVAL_SECONDS,
):
    """Publish forward progress while a verified external child process runs.

    ``is_alive`` must report the *actual* child state (e.g.
    ``lambda: proc.poll() is None``).  Ticking stops on the first falsy
    result and on context exit, whichever comes first.

    A no-op passthrough when no progress hook is installed on this thread, so
    non-hooked auxiliary tasks are byte-for-byte unchanged.
    """
    hook = getattr(_progress, "hook", None)
    if hook is None or not callable(is_alive):
        yield
        return

    stop = threading.Event()
    ticks = 0

    def _pump() -> None:
        nonlocal ticks
        while not stop.wait(max(0.05, float(interval))):
            try:
                alive = bool(is_alive())
            except Exception:
                alive = False
            if not alive:
                # Never claim liveness for a process that has exited.
                break
            ticks += 1
            try:
                hook()
            except Exception:
                logger.debug(
                    "aux external-process liveness hook failed (%s)",
                    label, exc_info=True,
                )

    thread = threading.Thread(
        target=_pump,
        name=f"hermes-aux-liveness-{label}",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=max(0.5, float(interval)))
        if ticks:
            logger.debug(
                "aux external-process liveness: %d tick(s) published for %s",
                ticks, label,
            )


# ── liveness-aware capturing runner ──────────────────────────────────────


def _terminate(proc: Any) -> None:
    """Kill and reap a child, best effort."""
    with contextlib.suppress(Exception):
        if proc.poll() is None:
            proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=5)


def run_text_capture(
    argv: list,
    *,
    timeout: Optional[float],
    env: Optional[dict],
    cwd: Optional[str],
    label: str = "external-process",
    interval: float = DEFAULT_LIVENESS_INTERVAL_SECONDS,
) -> "subprocess.CompletedProcess":
    """Run *argv*, capturing text stdout/stderr, publishing child liveness.

    Without an active progress hook this is exactly ``subprocess.run(...)`` —
    byte-for-byte the historical path, including how it is monkeypatched in
    tests.  With a hook active it spawns via ``Popen`` so the REAL child pid
    can back the heartbeat, then reproduces ``subprocess.run`` semantics:
    the child is killed and reaped before ``TimeoutExpired`` propagates.

    Raises:
        subprocess.TimeoutExpired: same contract as ``subprocess.run``.
    """
    if not progress_active():
        return subprocess.run(  # noqa: S603 - argv list, shell=False
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            cwd=cwd,
            env=env,
            check=False,
        )

    proc = subprocess.Popen(  # noqa: S603 - argv list, shell=False
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
        cwd=cwd,
        env=env,
    )
    try:
        with external_process_liveness(
            lambda: proc.poll() is None, label=label, interval=interval
        ):
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _terminate(proc)
                with contextlib.suppress(Exception):
                    proc.communicate(timeout=5)
                raise
    except BaseException:
        # Cancellation must not leak the child.
        _terminate(proc)
        raise
    return subprocess.CompletedProcess(
        args=argv, returncode=proc.returncode, stdout=stdout, stderr=stderr
    )
