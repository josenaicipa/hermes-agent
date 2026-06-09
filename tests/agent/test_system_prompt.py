"""Tests for agent/system_prompt.py — context-file cwd wiring."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.system_prompt import build_system_prompt_parts


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _implementation_delegation_guidance=True,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _captured_context_cwd(agent):
    """The cwd build_system_prompt_parts hands to build_context_files_prompt."""
    captured = {}

    def fake_context_files(cwd=None, skip_soul=False):
        captured["cwd"] = cwd
        return ""

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", side_effect=fake_context_files),
    ):
        build_system_prompt_parts(agent)
    return captured["cwd"]


class TestContextFileCwd:
    def test_none_when_terminal_cwd_unset(self, monkeypatch):
        # Unset → None, so discovery falls back to the launch dir inside
        # build_context_files_prompt (the local-CLI #19242 contract).
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        assert _captured_context_cwd(_make_agent()) is None

    def test_configured_dir_when_terminal_cwd_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert _captured_context_cwd(_make_agent()) == tmp_path


def _stable_part(agent):
    """The ``stable`` tier string build_system_prompt_parts produces."""
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


class TestImplementationDelegationGuardrail:
    """The delegate-code-to-Claude/Fable guardrail must be globally present."""

    MARKER = "delegate the actual"

    def test_present_for_every_channel_and_platform(self, monkeypatch):
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        # The guardrail lives in the stable tier, which is built identically
        # regardless of platform/channel/thread — prove it appears across a
        # spread of surfaces, not just one.
        for platform in ("discord", "slack", "whatsapp", "telegram", ""):
            agent = _make_agent(
                valid_tool_names=["write_file", "terminal"],
                platform=platform,
            )
            stable = _stable_part(agent)
            assert self.MARKER in stable, platform
            assert "Claude Agent SDK" in stable
            assert "orchestrate, review, and verify" in stable
            # Exception path is spelled out: unavailable OR explicit ask,
            # and must be reported.
            assert "unavailable" in stable and "explicit" in stable

    def test_absent_when_disabled_via_config_flag(self, monkeypatch):
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        agent = _make_agent(
            valid_tool_names=["write_file", "terminal"],
            _implementation_delegation_guidance=False,
        )
        assert self.MARKER not in _stable_part(agent)

    def test_default_on_when_flag_attribute_missing(self, monkeypatch):
        # getattr default is True, so sessions built before the flag existed
        # still get the guardrail.
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        agent = _make_agent(valid_tool_names=["write_file"])
        del agent._implementation_delegation_guidance  # type: ignore[attr-defined]
        assert self.MARKER in _stable_part(agent)

    def test_skipped_when_session_has_no_tools(self, monkeypatch):
        # A tool-less session cannot touch code, so the routing note is noise.
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        agent = _make_agent(valid_tool_names=[])
        assert self.MARKER not in _stable_part(agent)
