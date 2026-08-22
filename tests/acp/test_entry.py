"""Tests for acp_adapter.entry startup wiring."""

import json
import os
import subprocess
import sys
from pathlib import Path

import acp
import pytest

from acp_adapter import entry

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_main_enables_unstable_protocol(monkeypatch):
    calls = {}

    async def fake_run_agent(agent, **kwargs):
        calls["kwargs"] = kwargs

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert calls["kwargs"]["use_unstable_protocol"] is True


def test_main_skips_configured_mcp_discovery_when_requested(monkeypatch):
    discovery_calls = []

    async def fake_run_agent(agent, **kwargs):
        pass

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    monkeypatch.setattr(
        "tools.mcp_tool.discover_mcp_tools",
        lambda: discovery_calls.append(True),
    )
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert discovery_calls == []










def test_main_setup_offers_browser_install_when_tty(monkeypatch):
    """When stdin is a TTY and the user answers yes, model setup is followed
    by a browser-tools bootstrap call."""
    monkeypatch.setattr("hermes_cli.main.main", lambda: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "y")

    bootstrap_calls = []
    monkeypatch.setattr(
        entry,
        "_run_setup_browser",
        lambda assume_yes=False: bootstrap_calls.append(assume_yes) or 0,
    )

    entry.main(["--setup"])

    assert bootstrap_calls == [False]










def test_main_setup_browser_propagates_browser_failure(monkeypatch):
    """If browser install fails, exit code is 1."""
    def fake_ensure(dep, interactive=True):
        return dep != "browser"  # browser fails

    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", fake_ensure)

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--setup-browser"])
    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# Techos de espera larga en ACP (Jose 2026-08-21)
# ---------------------------------------------------------------------------
# En ACP nadie puede despertar al agente, así que una misión se espera dentro
# del turno que la despachó. Estos dos techos están calibrados para turnos
# cortos y hay que levantarlos, pero solo en este proceso y sin pisar lo que
# el operador haya fijado.


def test_long_wait_ceilings_cover_a_full_mission(monkeypatch):
    from acp_adapter import entry

    monkeypatch.delenv("TERMINAL_MAX_FOREGROUND_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", raising=False)
    entry._apply_acp_long_wait_ceilings()

    foreground = int(os.environ["TERMINAL_MAX_FOREGROUND_TIMEOUT"])
    tool_call = int(os.environ["HERMES_CONCURRENT_TOOL_TIMEOUT_S"])
    # El vigía espera por defecto 7500 s (alineado con CURSOR_TIMEOUT_MS = 2 h).
    # Ambos techos deben quedar POR ENCIMA o la espera muere antes que la misión.
    assert foreground > 7500, "el terminal rechazaría la espera de una misión larga"
    assert tool_call > foreground, "el ejecutor cortaría la espera antes que el terminal"


def test_long_wait_ceilings_never_override_the_operator(monkeypatch):
    from acp_adapter import entry

    monkeypatch.setenv("TERMINAL_MAX_FOREGROUND_TIMEOUT", "42")
    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "43")
    entry._apply_acp_long_wait_ceilings()

    assert os.environ["TERMINAL_MAX_FOREGROUND_TIMEOUT"] == "42"
    assert os.environ["HERMES_CONCURRENT_TOOL_TIMEOUT_S"] == "43"


def _run_entry_child(script: str, tmp_path: Path) -> dict:
    """Fresh interpreter so prior ACP imports cannot leak into the probe."""
    env = os.environ.copy()
    env.pop("TERMINAL_MAX_FOREGROUND_TIMEOUT", None)
    env.pop("HERMES_CONCURRENT_TOOL_TIMEOUT_S", None)
    env["HERMES_HOME"] = str(tmp_path)
    env["HERMES_ACP_SKIP_CONFIGURED_MCP"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_main_applies_ceilings_before_importing_server(tmp_path):
    """main() must raise the ceilings before ``acp_adapter.server`` is imported.

    ``tools.terminal_tool.FOREGROUND_MAX_TIMEOUT`` is baked at import time. If
    server (or a later terminal_tool import it triggers) loads first, the
    600 s default sticks for the life of the ACP process.
    """
    script = r"""
import importlib.abc
import importlib.machinery
import json
import os
import sys
import types

sys.path.insert(0, os.getcwd())

captured = {}

class _ServerImportProbe(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname != "acp_adapter.server":
            return None
        captured["fg"] = os.environ.get("TERMINAL_MAX_FOREGROUND_TIMEOUT")
        captured["tool"] = os.environ.get("HERMES_CONCURRENT_TOOL_TIMEOUT_S")
        return importlib.machinery.ModuleSpec(fullname, self)

    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        class HermesACPAgent:
            def __init__(self, *args, **kwargs):
                pass
        mod.HermesACPAgent = HermesACPAgent
        return mod

    def exec_module(self, module):
        pass

sys.meta_path.insert(0, _ServerImportProbe())

from acp_adapter import entry
import acp

async def fake_run_agent(agent, **kwargs):
    return None

entry._setup_logging = lambda: None
entry._load_env = lambda: None
acp.run_agent = fake_run_agent
entry.main([])

assert "tools.terminal_tool" not in sys.modules
import tools.terminal_tool as terminal_tool

print(json.dumps({
    "at_server_import": captured,
    "foreground_max": terminal_tool.FOREGROUND_MAX_TIMEOUT,
}))
"""
    payload = _run_entry_child(script, tmp_path)
    at_import = payload["at_server_import"]
    assert at_import.get("fg"), at_import
    assert at_import.get("tool"), at_import
    assert int(at_import["fg"]) > 7500
    assert int(at_import["tool"]) > int(at_import["fg"])
    assert payload["foreground_max"] > 7500


def test_long_wait_ceilings_are_scoped_to_the_acp_process(tmp_path):
    """Importing the adapter must not raise ceilings; only ACP main() does.

    Gateway and CLI are other processes and never call this helper, so their
    600/420 s caps stay intact.
    """
    script = r"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from acp_adapter import entry

print(json.dumps({
    "fg": os.environ.get("TERMINAL_MAX_FOREGROUND_TIMEOUT"),
    "tool": os.environ.get("HERMES_CONCURRENT_TOOL_TIMEOUT_S"),
    "apply": entry._apply_acp_long_wait_ceilings.__name__,
}))
"""
    payload = _run_entry_child(script, tmp_path)
    assert payload["fg"] is None
    assert payload["tool"] is None
    assert payload["apply"] == "_apply_acp_long_wait_ceilings"
