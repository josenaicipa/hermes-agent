"""Focused tests for the kimi-code-cli auxiliary adapter (Kimi Code OAuth).

Covers the properties that make this route safe to use for compression:

* it is a **distinct provider** from the ``kimi-coding`` API-key route and
  can never be reached by that route's aliases;
* every call runs under a Hermes-generated no-tools agent definition
  (``tools: []`` / ``subagents: []``), written 0600 into that call's private
  workspace and bound with ``--agent-file`` on the FIRST invocation only —
  the flag cannot be combined with session resume, and the bound agent
  persists across resumes;
* **session identity is pinned explicitly**: the identifier the CLI reports
  on the first turn is captured, validated, and passed back with the official
  Kimi Code 0.29 selector (``-S, --session TEXT``) on every later turn.  The
  adapter never emits ``--continue``, which would resolve against the shared
  profile-HOME session store and could splice two concurrent compressions
  together.  Missing/malformed/conflicting/drifting identifiers fail closed;
* ``KIMI_CODE_EXPERIMENTAL_FLAG=1`` is set explicitly (the installed build
  gates ``--agent-file`` behind it) and never inherited;
* ``extra_args`` cannot weaken any of the above (including via ``-S``);
* it is ARG_MAX-safe: the prompt is chunked, ordered, and acknowledged
  before the summary is requested;
* the stream-json parser demands a real final assistant response, and any
  observed tool activity fails the call as defense in depth;
* timeouts and cancellation kill the child.

All deterministic: a generated fake executable, never the real CLI, never
real credentials.
"""

from __future__ import annotations

import json
import logging
import re
import stat
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.kimi_code_cli_client import (
    AGENT_FILE_NAME,
    AGENT_NAME,
    DEFAULT_MODEL,
    KIMI_CODE_CLI_PROVIDER,
    KIMI_EXPERIMENTAL_ENV_VALUE,
    KIMI_EXPERIMENTAL_ENV_VAR,
    RESUME_SHORTHAND_TOKENS,
    SESSION_FLAG,
    SESSION_FLAG_SHORT,
    KimiCodeCLIClient,
    KimiCodeCLIConfigurationError,
    KimiCodeCLIConnectionError,
    KimiCodeCLITimeout,
    KimiCodeCLIToolActivityError,
    build_subprocess_env,
    format_messages_as_prompt,
    is_valid_session_id,
    observed_session_ids,
    parse_stream_json_final_text,
    parse_stream_json_session_id,
    render_agent_file,
    resolve_kimi_binary,
    validate_extra_args,
    verify_session_continuity,
    write_agent_file,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _event(**kwargs) -> str:
    return json.dumps(kwargs)


def _stream(*events: str) -> str:
    return "\n".join(events) + "\n"


def _assistant(text: str) -> str:
    return _event(
        type="assistant",
        message={"role": "assistant", "content": [{"type": "text", "text": text}]},
    )


def _session(session_id: str, **extra) -> str:
    """A Kimi Code 0.29 ``session.resume_hint`` event carrying an identifier."""
    payload = {
        "type": "session.resume_hint",
        "role": "system",
        "session_id": session_id,
    }
    payload.update(extra)
    return json.dumps(payload)


# Opaque identifiers of the shape Kimi Code 0.29 emits.
FAKE_SESSION_ID = "01JQ8ZC3VN4KDXR2H7YB6TFAWE"
FAKE_SESSION_ID_B = "01JQ8ZC3VN4KDXR2H7YB6TFBBB"


FAKE_CLI_TEMPLATE = '''#!{python}
"""Deterministic stand-in for the Kimi Code CLI 0.29 (tests only).

Session semantics mirror the real contract: the first turn *creates* a session
and reports its id on a ``session.resume_hint`` event; a later turn must hand
that id back with ``--session``. Being asked to resume an id this process did
not create is a hard error, so a cross-contaminated resume can never quietly
look like a success.
"""
import hashlib, json, os, re, sys

LOG = {log!r}
MODE = {mode!r}
SESSION_ID = {session_id!r}
BAD_SESSION_ID = {bad_session_id!r}

argv = sys.argv[1:]
prompt = ""
if "--prompt" in argv:
    prompt = argv[argv.index("--prompt") + 1]

session_arg = argv[argv.index("--session") + 1] if "--session" in argv else None
if "-S" in argv:
    session_arg = argv[argv.index("-S") + 1]
if SESSION_ID == "@cwd":
    # A real id is opaque. Deriving it from this call's private sandbox gives
    # every concurrent call a distinct, stable identifier of its own.
    SESSION_ID = "kimisess" + hashlib.sha1(
        os.getcwd().encode("utf-8")).hexdigest()[:20]

agent_file = argv[argv.index("--agent-file") + 1] if "--agent-file" in argv else None
agent_text = ""
agent_mode = ""
if agent_file and os.path.exists(agent_file):
    agent_text = open(agent_file, encoding="utf-8").read()
    agent_mode = oct(os.stat(agent_file).st_mode & 0o777)

record = {{
    "flags": [a for a in argv if a.startswith("--")],
    "dash_args": [a for a in argv if a.startswith("-")],
    "cwd": os.getcwd(),
    "home": os.environ.get("HOME"),
    "experimental": os.environ.get("KIMI_CODE_EXPERIMENTAL_FLAG"),
    "env_keys": sorted(os.environ.keys()),
    "cwd_entries": sorted(os.listdir(os.getcwd())),
    "continue": "--continue" in argv or "-c" in argv,
    "session_arg": session_arg,
    "session_id": SESSION_ID,
    "agent_file": agent_file,
    "agent_text": agent_text,
    "agent_mode": agent_mode,
    "prompt_bytes": len(prompt.encode("utf-8")),
    "is_final": "FINALIZE_STORED_CHUNKS" in prompt,
    "chunk_header": (prompt.split(chr(10))[0] if prompt else ""),
    "payload": prompt.split("<SOURCE_DATA>" + chr(10))[1].split(
        chr(10) + "</SOURCE_DATA>")[0] if "<SOURCE_DATA>" in prompt else "",
}}
# O_APPEND + one write() is atomic, so concurrent calls can share one log.
fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.write(fd, (json.dumps(record) + chr(10)).encode("utf-8"))
os.close(fd)

if session_arg is not None and session_arg != SESSION_ID:
    sys.stderr.write("unknown session " + str(session_arg) + chr(10))
    sys.exit(4)

if MODE == "leak_session" and session_arg is not None:
    sys.stderr.write("session " + str(session_arg) + " expired" + chr(10))
    sys.exit(5)

if MODE == "hang":
    import time
    time.sleep(600)


def emit(obj):
    print(json.dumps(obj))


def session_event(value):
    # Real resume hints carry prose; it must never become assistant text.
    emit({{"type": "session.resume_hint", "role": "system",
          "session_id": value,
          "content": "resume this conversation with --session " + str(value)}})


emit({{"type": "system", "subtype": "init"}})

if session_arg is None:
    if MODE == "no_session":
        pass
    elif MODE == "bad_session":
        session_event(BAD_SESSION_ID)
    elif MODE == "session_conflict":
        session_event(SESSION_ID)
        session_event(SESSION_ID + "B")
    else:
        session_event(SESSION_ID)
elif MODE == "session_drift":
    session_event(SESSION_ID + "DRIFT")
else:
    session_event(SESSION_ID)

if record["is_final"]:
    begin = re.search(r"HERMESFINALBEGIN[0-9A-F]+", prompt).group(0)
    end = re.search(r"HERMESFINALEND[0-9A-F]+", prompt).group(0)
    text = begin + chr(10) + "FINAL SUMMARY OK" + chr(10) + end
    if MODE == "no_final":
        emit({{"type": "thinking", "thinking": "pondering"}})
        sys.exit(0)
    if MODE == "tool_event":
        emit({{"type": "tool_use", "name": "Bash", "input": {{"command": "ls"}}}})
        sys.exit(0)
else:
    text = re.search(r"HERMESACK[0-9A-F]+\\d+", prompt).group(0)

emit({{"type": "thinking", "thinking": "internal reasoning"}})
emit({{
    "type": "assistant",
    "message": {{"role": "assistant",
                "content": [{{"type": "text", "text": text}}]}},
}})
emit({{"type": "result", "subtype": "success", "result": text}})
'''


def _install_fake_cli(
    tmp_path: Path,
    mode: str = "ok",
    *,
    name: str = "kimi",
    session_id: str = FAKE_SESSION_ID,
    bad_session_id: str = "",
) -> tuple[Path, Path]:
    """Write a fake ``kimi`` executable; return (binary, invocation log)."""
    log = tmp_path / f"invocations-{name}.jsonl"
    binary = tmp_path / "bin" / name
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(
        FAKE_CLI_TEMPLATE.format(
            python=sys.executable,
            log=str(log),
            mode=mode,
            session_id=session_id,
            bad_session_id=bad_session_id,
        ),
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary, log


def _profile(tmp_path: Path) -> Path:
    home = tmp_path / "profile" / "home"
    home.mkdir(parents=True, exist_ok=True)
    return home


def _read_log(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def _by_call(calls: list[dict]) -> dict[str, list[dict]]:
    """Group invocations by their private sandbox cwd (one group per call)."""
    grouped: dict[str, list[dict]] = {}
    for call in calls:
        grouped.setdefault(call["cwd"], []).append(call)
    return grouped


# ── provider identity: no collision with the API-key route ───────────────


class TestProviderNormalization:
    def test_canonical_id(self):
        assert KIMI_CODE_CLI_PROVIDER == "kimi-code-cli"

    def test_aliases_normalize_to_kimi_code_cli(self):
        from agent.auxiliary_client import _normalize_aux_provider

        assert _normalize_aux_provider("kimi-code-cli") == "kimi-code-cli"
        assert _normalize_aux_provider("kimi-cli") == "kimi-code-cli"
        assert _normalize_aux_provider("KIMI-CODE-CLI") == "kimi-code-cli"

    def test_api_key_provider_aliases_are_untouched(self):
        from agent.auxiliary_client import _normalize_aux_provider

        # These must keep resolving to the paid API-key route.
        assert _normalize_aux_provider("kimi") == "kimi-coding"
        assert _normalize_aux_provider("moonshot") == "kimi-coding"
        assert _normalize_aux_provider("kimi-coding") == "kimi-coding"
        assert _normalize_aux_provider("kimi-cn") == "kimi-coding-cn"
        assert _normalize_aux_provider("moonshot-cn") == "kimi-coding-cn"

    def test_no_alias_maps_across_the_two_routes(self):
        from agent.auxiliary_client import _PROVIDER_ALIASES

        for alias, target in _PROVIDER_ALIASES.items():
            if target == "kimi-code-cli":
                assert alias not in {"kimi", "moonshot", "kimi-coding"}
            if target in {"kimi-coding", "kimi-coding-cn"}:
                assert "cli" not in alias

    def test_exact_model_survives_normalization(self):
        from agent.auxiliary_client import _normalize_resolved_model

        assert (
            _normalize_resolved_model("kimi-code/k3", "kimi-code-cli")
            == "kimi-code/k3"
        )
        assert DEFAULT_MODEL == "kimi-code/k3"


class TestResolverBranch:
    def test_unconfigured_returns_unavailable_not_api_key_route(self):
        """A misconfigured OAuth route must never silently become an API key."""
        from agent import auxiliary_client

        with patch(
            "agent.kimi_code_cli_client.KimiCodeCLIClient",
            side_effect=KimiCodeCLIConfigurationError("cli not found"),
        ):
            client, model = auxiliary_client.resolve_provider_client(
                "kimi-code-cli", model="kimi-code/k3"
            )
        assert client is None
        assert model is None

    def test_async_mode_is_unavailable(self):
        from agent import auxiliary_client

        client, model = auxiliary_client.resolve_provider_client(
            "kimi-code-cli", model="kimi-code/k3", async_mode=True
        )
        assert (client, model) == (None, None)


# ── mandatory no-tools agent definition ──────────────────────────────────


class TestGeneratedAgentFile:
    def test_frontmatter_disables_tools_and_subagents(self):
        text = render_agent_file()
        assert text.startswith("---\n")
        frontmatter = text.split("---", 2)[1]
        assert re.search(r"^tools:\s*\[\]\s*$", frontmatter, re.M)
        assert re.search(r"^subagents:\s*\[\]\s*$", frontmatter, re.M)
        assert re.search(rf"^name:\s*{AGENT_NAME}\s*$", frontmatter, re.M)

    def test_frontmatter_parses_as_yaml_with_empty_lists(self):
        yaml = pytest.importorskip("yaml")
        frontmatter = render_agent_file().split("---", 2)[1]
        data = yaml.safe_load(frontmatter)
        assert data["tools"] == []
        assert data["subagents"] == []
        assert data["name"] == AGENT_NAME

    def test_body_is_self_contained_text_only_compression_prompt(self):
        body = render_agent_file().split("---", 2)[2]
        lowered = body.lower()
        # Deterministic, text-only summarization.
        assert "summarization" in lowered
        # Untrusted-data framing for conversation content.
        assert "untrusted" in lowered
        assert "source_data" in lowered
        # Explicit refusal of every tool-ish capability.
        for forbidden in ("run commands", "browse", "delegate", "no tools"):
            assert forbidden in lowered
        # Ordering + marker contract restated so the agent is self-contained.
        assert "numeric order" in lowered
        assert "marker" in lowered

    def test_written_file_is_mode_0600(self, tmp_path):
        path = Path(write_agent_file(str(tmp_path)))
        assert path.name == AGENT_FILE_NAME
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600, oct(mode)
        assert path.read_text(encoding="utf-8") == render_agent_file()

    def test_written_file_is_never_group_or_world_readable(self, tmp_path):
        path = Path(write_agent_file(str(tmp_path)))
        mode = stat.S_IMODE(path.stat().st_mode)
        assert not mode & stat.S_IRGRP
        assert not mode & stat.S_IROTH
        assert not mode & stat.S_IXUSR

    def test_refuses_to_clobber_an_existing_file(self, tmp_path):
        write_agent_file(str(tmp_path))
        with pytest.raises(FileExistsError):
            write_agent_file(str(tmp_path))


# ── extra_args cannot weaken the bind ────────────────────────────────────


class TestExtraArgsValidation:
    def test_empty_is_fine(self):
        assert validate_extra_args(None) == ()
        assert validate_extra_args([]) == ()

    def test_benign_flag_with_value_is_allowed(self):
        assert validate_extra_args(["--timeout", "30"]) == ("--timeout", "30")

    def test_benign_short_flag_is_still_allowed(self):
        """The short-option guard is targeted, not a blanket ban."""
        assert validate_extra_args(["-v"]) == ("-v",)

    def test_non_list_is_rejected(self):
        with pytest.raises(KimiCodeCLIConfigurationError, match="list of strings"):
            validate_extra_args("--yolo")
        with pytest.raises(KimiCodeCLIConfigurationError, match="list of strings"):
            validate_extra_args([1, 2])

    @pytest.mark.parametrize(
        "arg",
        [
            "--agent-file",
            "--agent",
            "--agents",
            "--agent-file=/tmp/evil.md",
            "--tools",
            "--allowed-tools",
            "--allowedTools",
            "--disallowed-tools",
            "--subagents",
            "--continue",
            "--resume",
            "--session",
            "--session-id",
            "--session=01JQ8ZC3VN4KDXR2H7YB6TFAWE",
            "--prompt",
            "--prompt-json",
            "--output-format",
            "--input-format",
            "--model",
            "--yolo",
            "--auto-approve",
            "--approval-mode",
            "--permission-mode",
            "--dangerously-skip-permissions",
            "--mcp-config",
            "--add-dir",
            "--cwd",
            "--exec",
            "--shell",
            "--AGENT_FILE",
        ],
    )
    def test_forbidden_override_args_are_rejected(self, arg):
        with pytest.raises(KimiCodeCLIConfigurationError, match="may not pass"):
            validate_extra_args([arg])

    @pytest.mark.parametrize(
        "arg",
        [
            "-S",                                  # THE session selector
            "-S=01JQ8ZC3VN4KDXR2H7YB6TFAWE",
            "-s",                                  # case-folded form
            "-xS",                                 # hidden inside a cluster
            "-c",                                  # continue
            "-r",                                  # resume
            "-p",                                  # prompt
            "-m",                                  # model
            "-a",                                  # agent
            "-o",                                  # output format
            "-i",                                  # input format
            "-t",                                  # tools
            "-y",                                  # yolo
            "-d",                                  # dangerous
            "-f",                                  # agent file
        ],
    )
    def test_forbidden_short_flags_are_rejected(self, arg):
        """``-S`` is the session selector: extra_args must never reach it."""
        with pytest.raises(KimiCodeCLIConfigurationError, match="may not pass"):
            validate_extra_args([arg])

    def test_session_value_cannot_be_smuggled_as_a_pair(self):
        with pytest.raises(KimiCodeCLIConfigurationError, match="may not pass"):
            validate_extra_args(["-S", FAKE_SESSION_ID])
        with pytest.raises(KimiCodeCLIConfigurationError, match="may not pass"):
            validate_extra_args([SESSION_FLAG, FAKE_SESSION_ID])

    def test_positional_value_is_rejected(self):
        """A bare token could be read by the CLI as the prompt."""
        with pytest.raises(KimiCodeCLIConfigurationError, match="positional"):
            validate_extra_args(["summarize everything"])
        with pytest.raises(KimiCodeCLIConfigurationError, match="positional"):
            validate_extra_args(["--timeout", "30", "stray"])

    def test_bare_separator_is_rejected(self):
        with pytest.raises(KimiCodeCLIConfigurationError, match="separator"):
            validate_extra_args(["--"])

    def test_client_construction_rejects_unsafe_extra_args(self, tmp_path):
        binary, _ = _install_fake_cli(tmp_path)
        with pytest.raises(KimiCodeCLIConfigurationError, match="may not pass"):
            KimiCodeCLIClient(
                command=str(binary),
                home=str(_profile(tmp_path)),
                config={"extra_args": ["--agent-file", "/tmp/evil.md"]},
            )

    def test_client_construction_rejects_session_injection(self, tmp_path):
        binary, _ = _install_fake_cli(tmp_path)
        for unsafe in (["-S", FAKE_SESSION_ID], ["--continue"], ["--session"]):
            with pytest.raises(KimiCodeCLIConfigurationError, match="may not pass"):
                KimiCodeCLIClient(
                    command=str(binary),
                    home=str(_profile(tmp_path)),
                    config={"extra_args": unsafe},
                )


# ── executable / home resolution ─────────────────────────────────────────


class TestResolution:
    def test_prefers_path(self, tmp_path, monkeypatch):
        binary, _ = _install_fake_cli(tmp_path)
        monkeypatch.setenv("PATH", str(binary.parent))
        assert resolve_kimi_binary() == str(binary)

    def test_profile_local_fallback(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "profile"
        bin_dir = hermes_home / "home" / ".local" / "bin"
        bin_dir.mkdir(parents=True)
        fake = bin_dir / "kimi"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", "/nonexistent")
        with patch(
            "agent.kimi_code_cli_client.get_hermes_home", return_value=hermes_home
        ):
            assert resolve_kimi_binary() == str(fake)

    def test_missing_binary_fails_clearly_and_names_no_api_route(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("PATH", "/nonexistent")
        with patch(
            "agent.kimi_code_cli_client.get_hermes_home",
            return_value=tmp_path / "profile",
        ):
            with pytest.raises(KimiCodeCLIConfigurationError) as exc:
                KimiCodeCLIClient(config={})
        assert "kimi-coding" in str(exc.value)  # explicitly refuses that route

    def test_absolute_command_must_be_executable(self, tmp_path):
        missing = tmp_path / "nope" / "kimi"
        assert resolve_kimi_binary(str(missing)) is None

    def test_missing_profile_home_fails_closed(self, tmp_path):
        """No profile HOME means no OAuth session — refuse, never guess."""
        binary, _ = _install_fake_cli(tmp_path)
        with patch(
            "agent.kimi_code_cli_client.resolve_kimi_home", return_value=None
        ):
            with pytest.raises(KimiCodeCLIConfigurationError, match="profile HOME"):
                KimiCodeCLIClient(command=str(binary), config={})


# ── environment: experimental flag on, credentials off ───────────────────


class TestSubprocessEnv:
    def test_profile_home_is_preserved_for_oauth(self, tmp_path):
        home = str(_profile(tmp_path))
        env = build_subprocess_env(home)
        assert env["HOME"] == home

    def test_experimental_flag_is_set_explicitly(self, tmp_path):
        env = build_subprocess_env(str(_profile(tmp_path)))
        assert env[KIMI_EXPERIMENTAL_ENV_VAR] == KIMI_EXPERIMENTAL_ENV_VALUE
        assert KIMI_EXPERIMENTAL_ENV_VAR == "KIMI_CODE_EXPERIMENTAL_FLAG"

    def test_experimental_flag_is_not_inherited(self, tmp_path, monkeypatch):
        """Ambient value must not decide whether the no-tools bind works."""
        monkeypatch.setenv(KIMI_EXPERIMENTAL_ENV_VAR, "0")
        env = build_subprocess_env(str(_profile(tmp_path)))
        assert env[KIMI_EXPERIMENTAL_ENV_VAR] == "1"

    def test_credential_env_vars_are_dropped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-must-not-leak")
        monkeypatch.setenv("KIMI_CODING_API_KEY", "sk-kimi-must-not-leak")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
        monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
        monkeypatch.setenv("GITHUB_TOKEN", "must-not-leak")
        monkeypatch.setenv("SOME_SECRET", "must-not-leak")
        monkeypatch.setenv("RANDOM_UNRELATED", "also-dropped")

        env = build_subprocess_env(str(_profile(tmp_path)))

        for key in (
            "KIMI_API_KEY",
            "KIMI_CODING_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GITHUB_TOKEN",
            "SOME_SECRET",
            "RANDOM_UNRELATED",
        ):
            assert key not in env
        assert not any(v == "must-not-leak" for v in env.values())

    def test_env_import_failure_still_allowlists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SECRET_TOKEN", "must-not-leak")
        with patch.dict(sys.modules, {"tools.environments.local": None}):
            env = build_subprocess_env(str(_profile(tmp_path)))
        assert "SECRET_TOKEN" not in env
        assert env[KIMI_EXPERIMENTAL_ENV_VAR] == "1"


# ── stream-json parsing ──────────────────────────────────────────────────


class TestStreamJsonParser:
    def test_result_event_wins(self):
        raw = _stream(
            _event(type="thinking", thinking="ignore me"),
            _assistant("partial"),
            _event(type="result", subtype="success", result="FINAL TEXT"),
        )
        assert parse_stream_json_final_text(raw) == "FINAL TEXT"

    def test_assistant_message_when_no_result_event(self):
        raw = _stream(
            _event(type="system", subtype="init"),
            _assistant("summary body"),
        )
        assert parse_stream_json_final_text(raw) == "summary body"

    def test_kimi_029_bare_assistant_message_is_understood(self):
        raw = _stream(
            _event(type="system.version", role="system", version="0.29.0"),
            _event(role="assistant", content="HERMESACK123"),
            _event(type="session.resume_hint", role="system", content="resume"),
        )
        assert parse_stream_json_final_text(raw) == "HERMESACK123"

    def test_session_metadata_is_never_assistant_text(self):
        """A resume hint carries prose; it must not become a summary."""
        raw = _stream(
            _event(type="system", subtype="init"),
            _session(
                FAKE_SESSION_ID,
                content="resume this conversation with --session " + FAKE_SESSION_ID,
            ),
            _assistant("real summary"),
        )
        assert parse_stream_json_final_text(raw) == "real summary"

    def test_session_metadata_without_a_role_is_still_not_text(self):
        raw = _stream(
            _event(
                type="session.created",
                id=FAKE_SESSION_ID,
                text="session started; resume with -S " + FAKE_SESSION_ID,
            ),
            _assistant("real summary"),
        )
        assert parse_stream_json_final_text(raw) == "real summary"

    def test_session_metadata_alone_is_not_a_response(self):
        with pytest.raises(KimiCodeCLIConnectionError):
            parse_stream_json_final_text(_stream(_session(FAKE_SESSION_ID)))

    def test_last_assistant_message_wins(self):
        raw = _stream(_assistant("first"), _assistant("second"))
        assert parse_stream_json_final_text(raw) == "second"

    def test_deltas_accumulate_when_nothing_complete_arrives(self):
        raw = _stream(
            _event(type="text_delta", delta={"type": "text", "text": "he"}),
            _event(type="text_delta", delta={"type": "text", "text": "llo"}),
        )
        assert parse_stream_json_final_text(raw) == "hello"

    def test_non_json_noise_lines_are_skipped(self):
        raw = "Starting Kimi Code...\n" + _assistant("clean output") + "\nbye\n"
        assert parse_stream_json_final_text(raw) == "clean output"

    def test_openai_shaped_choices_are_understood(self):
        raw = _stream(
            _event(choices=[{"message": {"role": "assistant", "content": "oai"}}])
        )
        assert parse_stream_json_final_text(raw) == "oai"

    def test_thinking_blocks_are_excluded_from_text(self):
        raw = _stream(
            _event(
                type="assistant",
                message={
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "secret chain of thought"},
                        {"type": "text", "text": "visible"},
                    ],
                },
            )
        )
        assert parse_stream_json_final_text(raw) == "visible"

    # -- failure modes --------------------------------------------------

    def test_reasoning_only_stream_is_a_failure(self):
        raw = _stream(
            _event(type="thinking", thinking="a"),
            _event(type="reasoning", reasoning="b"),
            _event(type="thought", thought="c"),
        )
        with pytest.raises(KimiCodeCLIConnectionError, match="reasoning"):
            parse_stream_json_final_text(raw)

    def test_empty_output_is_a_failure(self):
        with pytest.raises(KimiCodeCLIConnectionError, match="no parsable"):
            parse_stream_json_final_text("")

    def test_whitespace_only_assistant_text_is_a_failure(self):
        with pytest.raises(KimiCodeCLIConnectionError, match="no final assistant"):
            parse_stream_json_final_text(_stream(_assistant("   \n  ")))

    def test_system_events_only_is_a_failure(self):
        raw = _stream(
            _event(type="system", subtype="init"),
            _event(type="system", subtype="done"),
        )
        with pytest.raises(KimiCodeCLIConnectionError):
            parse_stream_json_final_text(raw)

    def test_error_result_raises(self):
        raw = _stream(
            _event(type="result", subtype="error_max_turns", is_error=True,
                   result="ran out")
        )
        with pytest.raises(KimiCodeCLIConnectionError, match="error result"):
            parse_stream_json_final_text(raw)

    # -- tool activity: defense in depth --------------------------------

    def test_tool_use_event_is_rejected(self):
        raw = _stream(
            _event(type="tool_use", name="Bash", input={"command": "ls"}),
            _assistant("summary"),
        )
        with pytest.raises(KimiCodeCLIToolActivityError, match="tool"):
            parse_stream_json_final_text(raw)

    def test_tool_use_content_block_is_rejected(self):
        raw = _stream(
            _event(
                type="assistant",
                message={
                    "role": "assistant",
                    "content": [{"type": "tool_use", "name": "Read", "input": {}}],
                },
            )
        )
        with pytest.raises(KimiCodeCLIToolActivityError):
            parse_stream_json_final_text(raw)

    def test_tool_result_event_is_rejected(self):
        raw = _stream(_event(type="tool_result", content="file contents"))
        with pytest.raises(KimiCodeCLIToolActivityError):
            parse_stream_json_final_text(raw)

    def test_error_text_is_redacted(self):
        raw = _stream(
            _event(
                type="result",
                is_error=True,
                error="auth token=abcdefghijklmnopqrstuvwxyz0123456789 rejected",
            )
        )
        with pytest.raises(KimiCodeCLIConnectionError) as exc:
            parse_stream_json_final_text(raw)
        assert "abcdefghijklmnopqrstuvwxyz0123456789" not in str(exc.value)


# ── session identifiers: capture + validation ────────────────────────────


class TestSessionIdValidation:
    @pytest.mark.parametrize(
        "value",
        [
            FAKE_SESSION_ID,                          # ULID-ish (real shape)
            "9f0c6d2a-3b41-4e5f-8a71-2c9d0e4b6f13",   # uuid4
            "9f0c6d2a3b414e5f8a712c9d0e4b6f13",        # bare hex
            "kimisess_01JQ8ZC3VN4KDXR2",
            "sess.01JQ8ZC3VN4KDXR2",
            "abc",                                     # shortest accepted
        ],
    )
    def test_real_kimi_shapes_are_accepted(self, value):
        assert is_valid_session_id(value)

    @pytest.mark.parametrize(
        "value",
        [
            None,
            123,
            True,
            [],
            {},
            "",                                  # empty
            "  ",                                # whitespace only
            "ab",                                # implausibly short
            "x" * 129,                           # implausibly long
            " 01JQ8ZC3VN4KDXR2",                 # leading space
            "01JQ8ZC3VN4KDXR2 ",                 # trailing space
            "01JQ8Z C3VN4KDXR2",                 # inner space
            "01JQ8Z\tC3VN4KDXR2",                # tab
            "01JQ8Z\nC3VN4KDXR2",                # newline
            "01JQ8Z\x00C3VN4KDXR2",              # NUL
            "01JQ8Z\x01C3VN4KDXR2",              # control char
            "-S",                                # option-like
            "--continue",                        # option-like
            "-01JQ8ZC3VN4KDXR2",                 # option-like
            "../../etc/passwd",                  # path-like
            "/tmp/evil",                         # path-like
            "sessions/01JQ8ZC3VN4KDXR2",         # path-like
            "C:\\sessions\\01JQ8Z",              # path-like
            "01JQ8Z;rm -rf /",                   # shell-ish
            "01JQ8Z$(whoami)",                   # shell-ish
            "01JQ8Z|tee",                        # shell-ish
            "session id=01JQ8Z",
        ],
    )
    def test_unsafe_identifiers_are_rejected(self, value):
        assert not is_valid_session_id(value)

    def test_captured_from_resume_hint(self):
        raw = _stream(
            _event(type="system", subtype="init"),
            _session(FAKE_SESSION_ID),
            _assistant("HERMESACK1"),
        )
        assert parse_stream_json_session_id(raw) == FAKE_SESSION_ID

    @pytest.mark.parametrize(
        "event",
        [
            _event(type="session.created", session_id=FAKE_SESSION_ID),
            _event(type="session.created", id=FAKE_SESSION_ID),
            _event(type="system", subtype="init", sessionId=FAKE_SESSION_ID),
            _event(type="system", session={"id": FAKE_SESSION_ID}),
            _event(type="system", session=FAKE_SESSION_ID),
            _event(type="result", subtype="success", result="ok",
                   session_id=FAKE_SESSION_ID),
            _event(type="result", subtype="success",
                   result={"session_id": FAKE_SESSION_ID}),
        ],
    )
    def test_capture_is_robust_across_event_shapes(self, event):
        raw = _stream(event, _assistant("HERMESACK1"))
        assert parse_stream_json_session_id(raw) == FAKE_SESSION_ID

    def test_repeated_identical_id_is_not_a_conflict(self):
        raw = _stream(
            _session(FAKE_SESSION_ID),
            _event(type="result", subtype="success", result="ok",
                   session_id=FAKE_SESSION_ID),
        )
        assert parse_stream_json_session_id(raw) == FAKE_SESSION_ID

    def test_unrelated_message_ids_are_not_mistaken_for_sessions(self):
        """A bare ``id`` on a non-session event must not be pinned."""
        raw = _stream(
            _event(type="assistant", id="msg_0001",
                   message={"role": "assistant", "content": "hi"}),
        )
        with pytest.raises(KimiCodeCLIConnectionError, match="session"):
            parse_stream_json_session_id(raw)

    def test_nested_message_ids_do_not_manufacture_a_conflict(self):
        """Message/result ids must not shadow or contradict the real one."""
        raw = _stream(
            _session(FAKE_SESSION_ID),
            _event(type="assistant", id="msg_0001",
                   message={"id": "msg_0001", "role": "assistant",
                            "content": "HERMESACK1"}),
            _event(type="result", subtype="success",
                   result={"id": "res_0002", "content": "HERMESACK1"}),
        )
        assert parse_stream_json_session_id(raw) == FAKE_SESSION_ID

    def test_missing_session_id_raises_retryable(self):
        raw = _stream(_event(type="system", subtype="init"), _assistant("ok"))
        with pytest.raises(KimiCodeCLIConnectionError, match="did not report"):
            parse_stream_json_session_id(raw)

    @pytest.mark.parametrize(
        "bad",
        ["", "   ", "--continue", "-S", "../../etc/passwd", "a b", "x" * 200],
    )
    def test_malformed_session_id_raises_retryable(self, bad):
        raw = _stream(_session(bad), _assistant("ok"))
        with pytest.raises(KimiCodeCLIConnectionError, match="unusable session"):
            parse_stream_json_session_id(raw)

    def test_conflicting_session_ids_raise_retryable(self):
        raw = _stream(
            _session(FAKE_SESSION_ID),
            _session(FAKE_SESSION_ID_B),
            _assistant("ok"),
        )
        with pytest.raises(KimiCodeCLIConnectionError, match="conflicting"):
            parse_stream_json_session_id(raw)

    def test_errors_never_disclose_the_identifier(self):
        raw = _stream(_session(FAKE_SESSION_ID), _session(FAKE_SESSION_ID_B))
        with pytest.raises(KimiCodeCLIConnectionError) as exc:
            parse_stream_json_session_id(raw)
        assert FAKE_SESSION_ID not in str(exc.value)
        assert FAKE_SESSION_ID_B not in str(exc.value)

    def test_observed_ids_reports_valid_and_malformed(self):
        ids, malformed = observed_session_ids(_stream(_session(FAKE_SESSION_ID)))
        assert ids == frozenset({FAKE_SESSION_ID})
        assert malformed is False

        ids, malformed = observed_session_ids(_stream(_session("-S")))
        assert ids == frozenset()
        assert malformed is True

    def test_continuity_accepts_a_silent_resume(self):
        """A resumed turn need not repeat the id."""
        verify_session_continuity(_stream(_assistant("ok")), FAKE_SESSION_ID)

    def test_continuity_accepts_the_pinned_id(self):
        verify_session_continuity(
            _stream(_session(FAKE_SESSION_ID), _assistant("ok")), FAKE_SESSION_ID
        )

    def test_continuity_rejects_a_different_id(self):
        with pytest.raises(KimiCodeCLIConnectionError, match="different session"):
            verify_session_continuity(
                _stream(_session(FAKE_SESSION_ID_B), _assistant("ok")),
                FAKE_SESSION_ID,
            )

    def test_continuity_rejects_a_malformed_id(self):
        with pytest.raises(KimiCodeCLIConnectionError, match="different session"):
            verify_session_continuity(
                _stream(_session("-S"), _assistant("ok")), FAKE_SESSION_ID
            )


# ── message formatting ───────────────────────────────────────────────────


class TestMessageFormatting:
    def test_role_labelled_and_ordered(self):
        prompt = format_messages_as_prompt(
            [
                {"role": "system", "content": "You compress context."},
                {"role": "user", "content": "CHECKPOINT INSTRUCTION"},
            ]
        )
        assert prompt.index("SYSTEM: You compress context.") < prompt.index(
            "USER: CHECKPOINT INSTRUCTION"
        )

    def test_rejects_images(self):
        with pytest.raises(ValueError, match="image|multimodal"):
            format_messages_as_prompt(
                [{"role": "user", "content": [{"type": "image_url",
                                               "image_url": {"url": "x"}}]}]
            )

    def test_rejects_streaming(self, tmp_path):
        binary, _ = _install_fake_cli(tmp_path)
        client = KimiCodeCLIClient(
            command=str(binary), home=str(_profile(tmp_path)), config={}
        )
        with pytest.raises(ValueError, match="stream"):
            client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )

    def test_rejects_tools(self, tmp_path):
        binary, _ = _install_fake_cli(tmp_path)
        client = KimiCodeCLIClient(
            command=str(binary), home=str(_profile(tmp_path)), config={}
        )
        with pytest.raises(ValueError, match="tool"):
            client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "x"}}],
            )


# ── end-to-end chunked session against the fake executable ───────────────


@pytest.fixture
def kimi_env(tmp_path, monkeypatch):
    """Fake CLI + isolated profile + workspace root under tmp_path."""
    binary, log = _install_fake_cli(tmp_path)
    home = _profile(tmp_path)
    monkeypatch.setattr(
        "agent.kimi_code_cli_client.get_hermes_home",
        lambda: tmp_path / "profile",
    )
    return {"binary": binary, "log": log, "home": home, "tmp": tmp_path}


class TestChunkedSession:
    def _client(self, kimi_env, **cfg):
        return KimiCodeCLIClient(command=str(kimi_env["binary"]), config=dict(cfg))

    def test_large_prompt_is_chunked_ordered_and_finalized(self, kimi_env):
        client = self._client(kimi_env)
        body = "A" * 200_000  # > 2 chunks
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": body}],
            timeout=120,
        )
        assert resp.choices[0].message.content == "FINAL SUMMARY OK"
        assert resp.model == "kimi-code/k3"

        calls = _read_log(kimi_env["log"])
        chunk_calls = [c for c in calls if not c["is_final"]]
        final_calls = [c for c in calls if c["is_final"]]

        # Ordered chunk delivery, then exactly one synthesis request.
        assert len(chunk_calls) >= 3
        assert len(final_calls) == 1
        assert calls[-1]["is_final"] is True
        total = len(chunk_calls)
        for i, call in enumerate(chunk_calls, start=1):
            assert call["chunk_header"] == f"SOURCE_CHUNK_{i}_OF_{total}"

        # Fresh session: the first turn creates it, every later turn resumes it
        # by explicit id — never by a bare --continue.
        assert chunk_calls[0]["session_arg"] is None
        assert all(c["session_arg"] == FAKE_SESSION_ID for c in chunk_calls[1:])
        assert final_calls[0]["session_arg"] == FAKE_SESSION_ID
        assert not any(c["continue"] for c in calls)

    # -- the no-tools bind, observed from the child ---------------------

    def test_agent_file_bound_on_first_invocation_only(self, kimi_env):
        client = self._client(kimi_env)
        client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "A" * 200_000}],
            timeout=120,
        )
        calls = _read_log(kimi_env["log"])
        assert len(calls) >= 4

        # First invocation binds the agent and does NOT resume.
        assert calls[0]["agent_file"] is not None
        assert "--agent-file" in calls[0]["flags"]
        assert calls[0]["session_arg"] is None
        assert SESSION_FLAG not in calls[0]["flags"]

        # No later invocation repeats it — docs: cannot combine with a resume.
        for call in calls[1:]:
            assert call["agent_file"] is None
            assert "--agent-file" not in call["flags"]
            assert call["session_arg"] == FAKE_SESSION_ID

    def test_no_invocation_combines_agent_file_with_session_resume(self, kimi_env):
        client = self._client(kimi_env)
        client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "A" * 200_000}],
            timeout=120,
        )
        for call in _read_log(kimi_env["log"]):
            assert not (
                call["session_arg"] is not None and "--agent-file" in call["flags"]
            )

    def test_resumed_turns_rebind_nothing_but_the_session(self, kimi_env):
        """A resumed turn carries only model/output-format/session/prompt."""
        client = self._client(kimi_env)
        client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "A" * 200_000}],
            timeout=120,
        )
        for call in _read_log(kimi_env["log"])[1:]:
            assert set(call["flags"]) == {
                "--model", "--output-format", SESSION_FLAG, "--prompt",
            }

    def test_child_sees_no_tools_agent_definition_at_mode_0600(self, kimi_env):
        client = self._client(kimi_env)
        client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            timeout=60,
        )
        first = _read_log(kimi_env["log"])[0]
        assert first["agent_mode"] == "0o600"
        frontmatter = first["agent_text"].split("---", 2)[1]
        assert re.search(r"^tools:\s*\[\]\s*$", frontmatter, re.M)
        assert re.search(r"^subagents:\s*\[\]\s*$", frontmatter, re.M)
        assert Path(first["agent_file"]).name == AGENT_FILE_NAME
        # The agent file lives inside this call's private workspace.
        assert str(Path(first["agent_file"]).parent) == first["cwd"]

    def test_child_sees_the_experimental_flag(self, kimi_env):
        client = self._client(kimi_env)
        client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            timeout=60,
        )
        for call in _read_log(kimi_env["log"]):
            assert call["experimental"] == "1"

    def test_agent_file_is_removed_with_the_workspace(self, kimi_env):
        client = self._client(kimi_env)
        client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            timeout=60,
        )
        first = _read_log(kimi_env["log"])[0]
        assert not Path(first["agent_file"]).exists()
        assert not Path(first["cwd"]).exists()

    def test_tool_event_from_the_child_fails_the_call(self, tmp_path, monkeypatch):
        """Defense in depth: even bound no-tools, observed tools abort."""
        binary, _ = _install_fake_cli(tmp_path, mode="tool_event")
        monkeypatch.setattr(
            "agent.kimi_code_cli_client.get_hermes_home",
            lambda: tmp_path / "profile",
        )
        _profile(tmp_path)
        client = KimiCodeCLIClient(command=str(binary), config={})
        with pytest.raises(KimiCodeCLIToolActivityError):
            client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                timeout=60,
            )

    # -- ARG_MAX / isolation --------------------------------------------

    def test_every_argument_stays_under_the_execve_limit(self, kimi_env):
        client = self._client(kimi_env)
        client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "B" * 300_000}],
            timeout=120,
        )
        for call in _read_log(kimi_env["log"]):
            assert call["prompt_bytes"] < 128 * 1024

    def test_payload_is_reassembled_losslessly(self, kimi_env):
        client = self._client(kimi_env)
        body = "漢字🌍é" * 20_000
        client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": body}],
            timeout=120,
        )
        chunks = [c["payload"] for c in _read_log(kimi_env["log"]) if not c["is_final"]]
        assert "".join(chunks) == format_messages_as_prompt(
            [{"role": "user", "content": body}]
        )

    def test_utf8_payload_survives_a_pinned_multi_turn_session(self, kimi_env):
        """Reassembly and session pinning must both hold on the same call."""
        client = self._client(kimi_env)
        body = "漢字🌍é" * 20_000
        client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": body}],
            timeout=120,
        )
        calls = _read_log(kimi_env["log"])
        assert len(calls) > 2  # genuinely multi-turn
        assert calls[0]["session_arg"] is None
        assert all(c["session_arg"] == FAKE_SESSION_ID for c in calls[1:])
        assert not any(c["continue"] for c in calls)

    def test_session_is_isolated_and_cleaned_up(self, kimi_env):
        client = self._client(kimi_env)
        client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "small"}],
            timeout=60,
        )
        calls = _read_log(kimi_env["log"])
        cwds = {c["cwd"] for c in calls}
        # One fresh sandbox cwd shared by this call's turns...
        assert len(cwds) == 1
        sandbox = Path(cwds.pop())
        # ...containing only Hermes' own scaffolding, no user files...
        assert set(calls[0]["cwd_entries"]) <= {"tmp", AGENT_FILE_NAME}
        # ...and removed deterministically afterwards.
        assert not sandbox.exists()

    def test_consecutive_calls_get_distinct_sessions(self, kimi_env):
        client = self._client(kimi_env)
        for _ in range(2):
            client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": "hello"}],
                timeout=60,
            )
        calls = _read_log(kimi_env["log"])
        assert len({c["cwd"] for c in calls}) == 2  # fresh session per call
        # Each call re-binds its own freshly generated agent file.
        first_of_each = [c for c in calls if c["agent_file"]]
        assert len(first_of_each) == 2
        assert first_of_each[0]["agent_file"] != first_of_each[1]["agent_file"]
        # Each call re-establishes its own session before resuming anything.
        for turns in _by_call(calls).values():
            assert turns[0]["session_arg"] is None
            assert all(t["session_arg"] is not None for t in turns[1:])

    def test_child_sees_profile_home_but_no_credentials(self, kimi_env, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-must-not-leak")
        client = self._client(kimi_env)
        client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            timeout=60,
        )
        call = _read_log(kimi_env["log"])[0]
        assert call["home"] == str(kimi_env["home"])
        assert "KIMI_API_KEY" not in call["env_keys"]

    def test_argv_uses_only_verified_native_cli_flags(self, kimi_env):
        client = self._client(kimi_env)
        client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            timeout=60,
        )
        calls = _read_log(kimi_env["log"])
        flags = calls[0]["flags"]
        assert "--model" in flags
        assert "--output-format" in flags
        assert "--effort" not in flags
        assert "--agent-file" in flags
        assert "--prompt" in flags
        # The resume selector is the official long form of ``-S, --session``.
        assert SESSION_FLAG == "--session"
        assert SESSION_FLAG_SHORT == "-S"
        assert SESSION_FLAG in calls[-1]["flags"]

    def test_exit_zero_without_final_response_fails(self, tmp_path, monkeypatch):
        binary, _ = _install_fake_cli(tmp_path, mode="no_final")
        monkeypatch.setattr(
            "agent.kimi_code_cli_client.get_hermes_home",
            lambda: tmp_path / "profile",
        )
        _profile(tmp_path)
        client = KimiCodeCLIClient(command=str(binary), config={})
        with pytest.raises(KimiCodeCLIConnectionError):
            client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                timeout=60,
            )


# ── session identity pinned end to end ───────────────────────────────────


class TestSessionPinningEndToEnd:
    def _run(self, tmp_path, monkeypatch, *, mode="ok", body="hi", **kw):
        binary, log = _install_fake_cli(tmp_path, mode=mode, **kw)
        monkeypatch.setattr(
            "agent.kimi_code_cli_client.get_hermes_home",
            lambda: tmp_path / "profile",
        )
        _profile(tmp_path)
        client = KimiCodeCLIClient(command=str(binary), config={})
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": body}],
            timeout=120,
        )
        return resp, _read_log(log)

    def test_every_later_invocation_receives_the_first_turn_id(
        self, tmp_path, monkeypatch
    ):
        resp, calls = self._run(tmp_path, monkeypatch, body="A" * 200_000)
        assert resp.choices[0].message.content == "FINAL SUMMARY OK"
        assert len(calls) >= 4
        assert calls[0]["session_arg"] is None
        for call in calls[1:]:
            assert call["session_arg"] == FAKE_SESSION_ID
            assert call["session_arg"] == calls[0]["session_id"]
            assert SESSION_FLAG in call["flags"]

    def test_continue_is_absent_from_every_invocation(self, tmp_path, monkeypatch):
        _, calls = self._run(tmp_path, monkeypatch, body="A" * 200_000)
        assert len(calls) >= 4
        for call in calls:
            assert call["continue"] is False
            for token in RESUME_SHORTHAND_TOKENS:
                assert token not in call["dash_args"]
            assert "--continue" not in call["flags"]

    def test_short_selector_is_not_emitted(self, tmp_path, monkeypatch):
        """We pin with the long form; ``-S`` must not appear in argv."""
        _, calls = self._run(tmp_path, monkeypatch, body="A" * 200_000)
        for call in calls:
            assert SESSION_FLAG_SHORT not in call["dash_args"]

    def test_missing_session_id_fails_closed(self, tmp_path, monkeypatch):
        with pytest.raises(KimiCodeCLIConnectionError, match="did not report"):
            self._run(tmp_path, monkeypatch, mode="no_session")

    @pytest.mark.parametrize(
        "bad",
        ["", "  ", "--continue", "-S", "../../etc/passwd", "a b", "x" * 200],
    )
    def test_malformed_session_id_fails_closed(self, tmp_path, monkeypatch, bad):
        with pytest.raises(KimiCodeCLIConnectionError, match="unusable session"):
            self._run(tmp_path, monkeypatch, mode="bad_session", bad_session_id=bad)

    def test_conflicting_session_ids_fail_closed(self, tmp_path, monkeypatch):
        with pytest.raises(KimiCodeCLIConnectionError, match="conflicting"):
            self._run(tmp_path, monkeypatch, mode="session_conflict")

    def test_session_drift_on_a_resumed_turn_fails_closed(
        self, tmp_path, monkeypatch
    ):
        with pytest.raises(KimiCodeCLIConnectionError, match="different session"):
            self._run(tmp_path, monkeypatch, mode="session_drift")

    def test_fail_closed_errors_are_retryable_for_the_chain(
        self, tmp_path, monkeypatch
    ):
        from agent.auxiliary_client import _is_connection_error

        for mode in ("no_session", "bad_session", "session_conflict"):
            with pytest.raises(KimiCodeCLIConnectionError) as exc:
                self._run(
                    tmp_path / mode, monkeypatch, mode=mode, bad_session_id="-S"
                )
            assert _is_connection_error(exc.value)

    def test_no_fallback_to_continue_when_the_id_is_unusable(
        self, tmp_path, monkeypatch
    ):
        """Fail closed means *stop*, not resume by another route."""
        binary, log = _install_fake_cli(tmp_path, mode="no_session")
        monkeypatch.setattr(
            "agent.kimi_code_cli_client.get_hermes_home",
            lambda: tmp_path / "profile",
        )
        _profile(tmp_path)
        client = KimiCodeCLIClient(command=str(binary), config={})
        with pytest.raises(KimiCodeCLIConnectionError):
            client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": "A" * 200_000}],
                timeout=120,
            )
        calls = _read_log(log)
        # Exactly one turn ran; nothing was resumed by any means.
        assert len(calls) == 1
        assert not any(c["continue"] for c in calls)
        assert all(c["session_arg"] is None for c in calls)

    def test_session_id_is_never_logged(self, tmp_path, monkeypatch):
        logger = logging.getLogger("agent.kimi_code_cli_client")
        messages: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record):
                messages.append(record.getMessage())

        handler = _Capture()
        previous_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            self._run(tmp_path, monkeypatch, body="A" * 200_000)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

        assert messages, "the adapter should still emit its debug breadcrumb"
        assert any("session pinned" in m for m in messages)
        for message in messages:
            assert FAKE_SESSION_ID not in message

    def test_session_id_is_redacted_from_cli_errors(self, tmp_path, monkeypatch):
        with pytest.raises(KimiCodeCLIConnectionError) as exc:
            self._run(tmp_path, monkeypatch, mode="leak_session")
        message = str(exc.value)
        assert "exit status 5" in message
        assert FAKE_SESSION_ID not in message


class TestConcurrentSessionIdentity:
    def test_two_calls_sharing_home_never_cross_sessions(
        self, tmp_path, monkeypatch
    ):
        """The regression this P1 exists for.

        Two compressions run concurrently under the SAME profile HOME (so the
        same CLI session store) and the same executable. Each first turn
        establishes its own id; every continuation must carry that call's own
        id and never the other's. The fake CLI also exits non-zero if handed
        an id it did not create, so a cross would fail loudly either way.
        """
        # "@cwd": each call's private sandbox yields its own stable identifier.
        binary, log = _install_fake_cli(tmp_path, session_id="@cwd")
        monkeypatch.setattr(
            "agent.kimi_code_cli_client.get_hermes_home",
            lambda: tmp_path / "profile",
        )
        _profile(tmp_path)
        client = KimiCodeCLIClient(command=str(binary), config={})

        results: dict[int, str] = {}
        errors: list[BaseException] = []

        def _worker(index: int) -> None:
            try:
                resp = client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    # Multi-chunk so each call has several continuations.
                    messages=[{"role": "user", "content": chr(65 + index) * 200_000}],
                    timeout=180,
                )
                results[index] = resp.choices[0].message.content
            except BaseException as exc:  # pragma: no cover - failure detail
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=300)
        assert not any(t.is_alive() for t in threads)
        assert not errors, f"concurrent calls failed: {errors!r}"
        assert results == {0: "FINAL SUMMARY OK", 1: "FINAL SUMMARY OK"}

        calls = _read_log(log)
        per_call = _by_call(calls)
        assert len(per_call) == 2, "two calls must use two private sandboxes"

        pinned: set[str] = set()
        for turns in per_call.values():
            first, rest = turns[0], turns[1:]
            own = first["session_id"]
            assert first["session_arg"] is None      # created here
            assert "--agent-file" in first["flags"]  # bound here
            assert rest, "each call must have continuations to pin"
            # Every continuation pins THIS call's identity, not the other's.
            assert all(t["session_arg"] == own for t in rest)
            assert all(t["session_id"] == own for t in turns)
            assert all(SESSION_FLAG in t["flags"] for t in rest)
            pinned.add(own)

        # Distinct identities, and no invocation anywhere saw the other's id
        # or a bare --continue.
        assert len(pinned) == 2
        for turns in per_call.values():
            others = pinned - {turns[0]["session_id"]}
            assert all(t["session_arg"] not in others for t in turns)
        assert not any(c["continue"] for c in calls)

    def test_distinct_first_turn_ids_stay_bound_to_their_own_call(
        self, tmp_path, monkeypatch
    ):
        """Same shared HOME, two CLIs with fixed distinct ids."""
        monkeypatch.setattr(
            "agent.kimi_code_cli_client.get_hermes_home",
            lambda: tmp_path / "profile",
        )
        _profile(tmp_path)
        first, first_log = _install_fake_cli(
            tmp_path, name="kimi-a", session_id=FAKE_SESSION_ID
        )
        second, second_log = _install_fake_cli(
            tmp_path, name="kimi-b", session_id=FAKE_SESSION_ID_B
        )
        clients = [
            KimiCodeCLIClient(command=str(first), config={}),
            KimiCodeCLIClient(command=str(second), config={}),
        ]
        errors: list[BaseException] = []

        def _worker(client) -> None:
            try:
                client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=[{"role": "user", "content": "Z" * 200_000}],
                    timeout=180,
                )
            except BaseException as exc:  # pragma: no cover - failure detail
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(c,)) for c in clients]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=300)
        assert not errors, f"concurrent calls failed: {errors!r}"

        for log, own, other in (
            (first_log, FAKE_SESSION_ID, FAKE_SESSION_ID_B),
            (second_log, FAKE_SESSION_ID_B, FAKE_SESSION_ID),
        ):
            calls = _read_log(log)
            assert len(calls) >= 4
            assert calls[0]["session_arg"] is None
            assert all(c["session_arg"] == own for c in calls[1:])
            assert all(c["session_arg"] != other for c in calls)
            assert not any(c["continue"] for c in calls)
        # Both calls really did share one profile HOME.
        homes = {c["home"] for c in _read_log(first_log) + _read_log(second_log)}
        assert homes == {str(tmp_path / "profile" / "home")}


# ── process lifecycle: timeout / cancellation ────────────────────────────


class TestProcessLifecycle:
    def test_timeout_kills_child_and_raises(self, tmp_path, monkeypatch):
        binary, log = _install_fake_cli(tmp_path, mode="hang")
        monkeypatch.setattr(
            "agent.kimi_code_cli_client.get_hermes_home",
            lambda: tmp_path / "profile",
        )
        _profile(tmp_path)
        client = KimiCodeCLIClient(command=str(binary), config={})

        with pytest.raises(KimiCodeCLITimeout, match="timed out"):
            client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                timeout=1,
            )
        # The child recorded its sandbox before hanging; the sandbox is gone,
        # so cleanup ran even on the timeout path.
        recorded = _read_log(log)
        assert recorded, "fake CLI should have been started"
        assert not Path(recorded[0]["cwd"]).exists()

    def test_nonzero_exit_is_sanitized(self, tmp_path, monkeypatch):
        """Provider stderr must never leak credential material into logs."""
        monkeypatch.setattr(
            "agent.kimi_code_cli_client.get_hermes_home",
            lambda: tmp_path / "profile",
        )
        _profile(tmp_path)
        binary = tmp_path / "bin" / "kimi"
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text(
            "#!/bin/sh\n"
            "echo 'auth token=abcdefghijklmnopqrstuvwxyz0123456789 failed' 1>&2\n"
            "exit 3\n"
        )
        binary.chmod(0o755)
        client = KimiCodeCLIClient(command=str(binary), config={})

        with pytest.raises(KimiCodeCLIConnectionError) as exc:
            client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                timeout=30,
            )
        msg = str(exc.value)
        assert "abcdefghijklmnopqrstuvwxyz0123456789" not in msg
        assert "exit status 3" in msg

    def test_timeout_error_is_classified_as_retryable(self):
        from agent.auxiliary_client import _is_connection_error, _is_timeout_error

        exc = KimiCodeCLITimeout("kimi-code-cli timed out after 300.0s")
        assert _is_timeout_error(exc) or _is_connection_error(exc)

    def test_transport_error_is_classified_as_retryable(self):
        from agent.auxiliary_client import _is_connection_error

        assert _is_connection_error(
            KimiCodeCLIConnectionError("kimi-code-cli produced no final response")
        )

    def test_configuration_error_is_not_a_transport_error(self):
        from agent.auxiliary_client import _is_connection_error

        # A misconfiguration must not look like a transient blip.
        assert not _is_connection_error(
            KimiCodeCLIConfigurationError("extra_args may not pass --agent-file")
        )

    def test_cancellation_terminates_the_child(self, tmp_path, monkeypatch):
        """A KeyboardInterrupt/cancel raised into the worker kills the CLI."""
        from agent import kimi_code_cli_client as mod

        binary, _ = _install_fake_cli(tmp_path, mode="hang")
        monkeypatch.setattr(mod, "get_hermes_home", lambda: tmp_path / "profile")
        _profile(tmp_path)
        client = KimiCodeCLIClient(command=str(binary), config={})

        started: list = []
        real_popen = subprocess.Popen

        class _TrackingPopen(real_popen):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                started.append(self)

            def communicate(self, *a, **kw):
                raise KeyboardInterrupt("cancelled")

        monkeypatch.setattr(mod.subprocess, "Popen", _TrackingPopen)
        with pytest.raises(KeyboardInterrupt):
            client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                timeout=30,
            )
        assert started, "a child process should have been spawned"
        for proc in started:
            proc.wait(timeout=10)
            assert proc.poll() is not None  # reaped, not orphaned
