"""Focused tests for the google-gemini-cli / agy auxiliary adapter.

Covers non-streaming text completions via the installed ``agy`` CLI for
profile-level auxiliary.compression (display model names like
``Gemini 3.5 Flash (Medium)`` over existing OAuth).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── helpers ──────────────────────────────────────────────────────────────


def _ok_proc(stdout: str = "compressed summary", returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = ""
    return proc


# ── message → prompt conversion ─────────────────────────────────────────


class TestFormatMessagesAsPrompt:
    def test_role_labelled_text_messages(self):
        from agent.agy_cli_client import format_messages_as_prompt

        prompt = format_messages_as_prompt(
            [
                {"role": "system", "content": "You compress context."},
                {"role": "user", "content": "Summarize this."},
                {"role": "assistant", "content": "Prior note."},
            ]
        )
        assert "SYSTEM: You compress context." in prompt
        assert "USER: Summarize this." in prompt
        assert "ASSISTANT: Prior note." in prompt

    def test_rejects_image_content_parts(self):
        from agent.agy_cli_client import format_messages_as_prompt

        with pytest.raises(ValueError, match="image|multimodal|unsupported"):
            format_messages_as_prompt(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "look"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,abc"},
                            },
                        ],
                    }
                ]
            )

    def test_rejects_tools_kwarg_via_create(self):
        from agent.agy_cli_client import AgyCLIClient

        client = AgyCLIClient(command="/usr/bin/agy")
        with pytest.raises(ValueError, match="tools|unsupported"):
            client.chat.completions.create(
                model="Gemini 3.5 Flash (Medium)",
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "x"}}],
            )

    def test_rejects_streaming(self):
        from agent.agy_cli_client import AgyCLIClient

        client = AgyCLIClient(command="/usr/bin/agy")
        with pytest.raises(ValueError, match="stream|unsupported"):
            client.chat.completions.create(
                model="Gemini 3.5 Flash (Medium)",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )


# ── binary resolution ───────────────────────────────────────────────────


class TestResolveAgyBinary:
    def test_prefers_path_which(self, tmp_path, monkeypatch):
        from agent.agy_cli_client import resolve_agy_binary

        fake = tmp_path / "agy"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        assert resolve_agy_binary() == str(fake)

    def test_profile_home_fallback(self, tmp_path, monkeypatch):
        from agent.agy_cli_client import resolve_agy_binary

        hermes_home = tmp_path / "profile"
        profile_home = hermes_home / "home"
        bin_dir = profile_home / ".local" / "bin"
        bin_dir.mkdir(parents=True)
        fake = bin_dir / "agy"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)

        monkeypatch.setenv("PATH", "/nonexistent")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        with patch("agent.agy_cli_client.get_hermes_home", return_value=hermes_home):
            assert resolve_agy_binary() == str(fake)


# ── subprocess invocation ───────────────────────────────────────────────


class TestAgyCLIClientCreate:
    def test_successful_response_and_model(self):
        from agent.agy_cli_client import AgyCLIClient

        client = AgyCLIClient(command="/usr/bin/agy")
        with patch("agent.agy_cli_client.subprocess.run", return_value=_ok_proc("hello world")) as run:
            resp = client.chat.completions.create(
                model="Gemini 3.5 Flash (Medium)",
                messages=[{"role": "user", "content": "Say hi"}],
                timeout=30,
            )

        assert resp.choices[0].message.content == "hello world"
        assert resp.model == "Gemini 3.5 Flash (Medium)"
        run.assert_called_once()
        args, kwargs = run.call_args
        argv = args[0]
        assert argv[0] == "/usr/bin/agy"
        assert "--print" in argv
        assert "Say hi" in argv or any("USER:" in a for a in argv)
        assert "--model" in argv
        mi = argv.index("--model")
        assert argv[mi + 1] == "Gemini 3.5 Flash (Medium)"
        assert "--print-timeout" in argv
        ti = argv.index("--print-timeout")
        assert argv[ti + 1].endswith("s")
        assert kwargs.get("shell") in (None, False)
        assert kwargs.get("timeout") == 30

    def test_argv_list_no_shell(self):
        from agent.agy_cli_client import AgyCLIClient

        client = AgyCLIClient(command="/usr/bin/agy")
        with patch("agent.agy_cli_client.subprocess.run", return_value=_ok_proc("ok")) as run:
            client.chat.completions.create(
                model="m",
                messages=[{"role": "user", "content": "x"}],
                timeout=12,
            )
        kwargs = run.call_args.kwargs
        assert kwargs.get("shell") is not True
        assert isinstance(run.call_args.args[0], list)

    def test_profile_home_is_not_exposed_to_agy(self, tmp_path, monkeypatch):
        from agent.agy_cli_client import AgyCLIClient

        hermes_home = tmp_path / "profile"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        (profile_home / "private.txt").write_text("secret")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("SHOULD_NOT_LEAK", "secret")

        client = AgyCLIClient(command="/usr/bin/agy")
        with patch("agent.agy_cli_client.get_hermes_home", return_value=hermes_home), \
             patch("agent.agy_cli_client.subprocess.run", return_value=_ok_proc("ok")) as run:
            client.chat.completions.create(
                model="m",
                messages=[{"role": "user", "content": "x"}],
                timeout=10,
            )
        env = run.call_args.kwargs["env"]
        isolated_home = Path(env["HOME"])
        assert isolated_home != profile_home
        assert "SHOULD_NOT_LEAK" not in env
        assert not isolated_home.exists()

    def test_env_import_failure_remains_allowlisted(self, tmp_path, monkeypatch):
        import sys
        from agent.agy_cli_client import _build_subprocess_env

        monkeypatch.setenv("SECRET_TOKEN", "must-not-leak")
        with patch.dict(sys.modules, {"tools.environments.local": None}):
            env = _build_subprocess_env(str(tmp_path))
        assert env["HOME"] == str(tmp_path)
        assert "SECRET_TOKEN" not in env

    def test_timeout_kills_and_raises(self):
        from agent.agy_cli_client import AgyCLIClient

        client = AgyCLIClient(command="/usr/bin/agy")
        with patch(
            "agent.agy_cli_client.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["agy"], timeout=5),
        ):
            with pytest.raises(RuntimeError, match="timeout|timed out"):
                client.chat.completions.create(
                    model="m",
                    messages=[{"role": "user", "content": "x"}],
                    timeout=5,
                )

    def test_nonzero_exit_sanitized(self):
        from agent.agy_cli_client import AgyCLIClient

        client = AgyCLIClient(command="/usr/bin/agy")
        bad = _ok_proc(stdout="", returncode=2)
        bad.stderr = "auth token=super-secret-token-xyz failed"
        with patch("agent.agy_cli_client.subprocess.run", return_value=bad):
            with pytest.raises(RuntimeError) as exc_info:
                client.chat.completions.create(
                    model="m",
                    messages=[{"role": "user", "content": "x"}],
                    timeout=10,
                )
        msg = str(exc_info.value)
        assert "super-secret-token-xyz" not in msg
        assert "exit" in msg.lower() or "failed" in msg.lower() or "non-zero" in msg.lower()

    def test_empty_output_rejected(self):
        from agent.agy_cli_client import AgyCLIClient

        client = AgyCLIClient(command="/usr/bin/agy")
        with patch("agent.agy_cli_client.subprocess.run", return_value=_ok_proc(stdout="   \n")):
            with pytest.raises(RuntimeError, match="empty"):
                client.chat.completions.create(
                    model="m",
                    messages=[{"role": "user", "content": "x"}],
                    timeout=10,
                )


# ── auxiliary resolver aliases ──────────────────────────────────────────


class TestAgyAuxiliaryAliases:
    def test_aliases_normalize_to_google_gemini_cli(self):
        from agent.auxiliary_client import _normalize_aux_provider

        assert _normalize_aux_provider("google-gemini-cli") == "google-gemini-cli"
        assert _normalize_aux_provider("gemini-cli") == "google-gemini-cli"
        assert _normalize_aux_provider("agy") == "google-gemini-cli"

    def test_native_gemini_alias_unchanged(self):
        from agent.auxiliary_client import _normalize_aux_provider

        assert _normalize_aux_provider("gemini") == "gemini"
        assert _normalize_aux_provider("google") == "gemini"
        assert _normalize_aux_provider("google-gemini") == "gemini"

    def test_resolve_provider_client_returns_agy_adapter(self):
        from agent.auxiliary_client import resolve_provider_client

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock())),
            base_url="agy://cli",
            api_key="agy",
        )
        with patch("agent.agy_cli_client.resolve_agy_binary", return_value="/usr/bin/agy"), \
             patch("agent.agy_cli_client.AgyCLIClient", return_value=fake_client) as ctor:
            client, model = resolve_provider_client(
                "gemini-cli",
                model="Gemini 3.5 Flash (Medium)",
            )
        assert client is fake_client
        assert model == "Gemini 3.5 Flash (Medium)"
        ctor.assert_called_once()

    def test_missing_agy_returns_none_for_fallback(self):
        from agent.auxiliary_client import resolve_provider_client

        with patch("agent.agy_cli_client.resolve_agy_binary", return_value=None):
            client, model = resolve_provider_client(
                "google-gemini-cli",
                model="Gemini 3.5 Flash (Medium)",
            )
        assert client is None
        assert model is None

    def test_async_resolution_fails_closed_for_sync_only_cli(self):
        from agent.auxiliary_client import resolve_provider_client

        with patch("agent.agy_cli_client.resolve_agy_binary", return_value="/usr/bin/agy"):
            client, model = resolve_provider_client(
                "google-gemini-cli",
                model="Gemini 3.5 Flash (Medium)",
                async_mode=True,
            )
        assert client is None
        assert model is None

    def test_call_llm_routes_through_agy_client(self):
        from agent.auxiliary_client import call_llm

        create = MagicMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))],
                model="Gemini 3.5 Flash (Medium)",
            )
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            base_url="agy://cli",
            api_key="agy",
        )
        with patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("google-gemini-cli", "Gemini 3.5 Flash (Medium)", None, None, None),
        ), patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(fake_client, "Gemini 3.5 Flash (Medium)"),
        ), patch(
            "agent.auxiliary_client._validate_llm_response",
            side_effect=lambda r, *a, **kw: r,
        ), patch(
            "agent.auxiliary_client._get_task_timeout",
            return_value=60.0,
        ):
            resp = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "compress me"}],
            )
        assert resp.choices[0].message.content == "summary"
        create.assert_called_once()


class TestAgyLargePromptChunking:
    def test_large_prompt_uses_private_chunked_conversation(self):
        from agent.agy_cli_client import AgyCLIClient

        client = AgyCLIClient(command="/usr/bin/agy")
        huge = ("context-line-🙂\n" * 20_000) + "OMEGA992"
        seen = []
        conversation_id = "623aa500-494d-4436-a6b3-4f75d08e0039"

        def fake_run(argv, **kwargs):
            import re
            from pathlib import Path

            seen.append((list(argv), dict(kwargs)))
            prompt = argv[argv.index("--print") + 1]
            log_path = Path(argv[argv.index("--log-file") + 1])
            assert log_path.exists()
            assert log_path.stat().st_mode & 0o777 == 0o600
            if "--conversation" not in argv:
                log_path.write_text(
                    f"Created conversation {conversation_id}\n",
                    encoding="utf-8",
                )
            if "FINALIZE_STORED_CHUNKS" in prompt:
                begin = re.search(r"HERMESFINALBEGIN[A-F0-9]+", prompt)
                end = re.search(r"HERMESFINALEND[A-F0-9]+", prompt)
                assert begin is not None and end is not None
                return _ok_proc(
                    f"{begin.group(0)}\nSUMMARY_OK\n{end.group(0)}\n"
                )
            chunk_no = re.search(r"SOURCE_CHUNK_(\d+)_OF_\d+", prompt)
            assert chunk_no is not None
            return _ok_proc(f"HERMESACK{chunk_no.group(1)}")

        with patch(
            "agent.agy_cli_client.subprocess.run",
            side_effect=fake_run,
        ):
            response = client.chat.completions.create(
                model="Gemini 3.5 Flash (Medium)",
                messages=[{"role": "user", "content": huge}],
                timeout=90,
            )

        assert response.choices[0].message.content == "SUMMARY_OK"
        assert len(seen) >= 3
        for argv, kwargs in seen:
            assert huge not in argv
            assert "--dangerously-skip-permissions" not in argv
            assert "--sandbox" in argv
            assert kwargs.get("input") is None
            assert kwargs.get("shell") is False
            assert max(len(arg.encode("utf-8")) for arg in argv) < 128 * 1024
        assert "--conversation" not in seen[0][0]
        assert all("--conversation" in argv for argv, _ in seen[1:])
        workspace = Path(seen[0][1]["cwd"])
        assert workspace.name.startswith("call-")
        assert "agy-compression" in workspace.parent.name
        assert not workspace.exists()
        assert workspace.parent.stat().st_mode & 0o777 == 0o700

    def test_tool_attempt_in_log_fails_closed(self):
        from agent.agy_cli_client import AgyCLIClient, AgyCLITransportError

        client = AgyCLIClient(command="/usr/bin/agy")
        huge = "untrusted source\n" * 10_000

        def fake_run(argv, **kwargs):
            log_path = Path(argv[argv.index("--log-file") + 1])
            log_path.write_text(
                "Created conversation 623aa500-494d-4436-a6b3-4f75d08e0039\n"
                "Tool confirmation for conversation: Step_RunCommand\n",
                encoding="utf-8",
            )
            return _ok_proc("HERMESACK1")

        with patch("agent.agy_cli_client.subprocess.run", side_effect=fake_run):
            with pytest.raises(AgyCLITransportError, match="tool"):
                client.chat.completions.create(
                    model="Gemini 3.5 Flash (Medium)",
                    messages=[{"role": "user", "content": huge}],
                    timeout=90,
                )

    def test_dead_pid_workspace_is_cleaned(self, tmp_path, monkeypatch):
        from agent.agy_cli_client import resolve_agy_workspace

        # Hermetic: pin the workspace to an isolated profile home so the test
        # never touches the real profile's agy-compression cache. get_hermes_home
        # is consulted before the HERMES_HOME env fallback, so it must be patched.
        hermes_home = tmp_path / "profile"
        (hermes_home / "home").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        with patch("agent.agy_cli_client.get_hermes_home", return_value=hermes_home):
            root = Path(resolve_agy_workspace())
            # A dead PID (pid 99999999 does not exist) plus a legacy unlabelled
            # workspace must both be reclaimed on the next resolve.
            stale = root / "call-99999999-stale"
            stale.mkdir(mode=0o700)
            (stale / "orphan.txt").write_text("stale")
            legacy = root / "call-deadbeef"  # no PID label (pre-hardening layout)
            legacy.mkdir(mode=0o700)
            resolve_agy_workspace()
            assert not stale.exists()
            assert not legacy.exists()

    def test_live_pid_workspace_is_preserved(self, tmp_path, monkeypatch):
        import sys

        from agent.agy_cli_client import resolve_agy_workspace

        hermes_home = tmp_path / "profile"
        (hermes_home / "home").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        # A real, live concurrent process owned by another PID must never be
        # reclaimed (requirement: never delete live concurrent calls).
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            assert child.pid != os.getpid()
            with patch(
                "agent.agy_cli_client.get_hermes_home", return_value=hermes_home
            ):
                root = Path(resolve_agy_workspace())
                live = root / f"call-{child.pid}-live"
                live.mkdir(mode=0o700)
                (live / "in_flight.txt").write_text("keep")
                resolve_agy_workspace()
            assert live.exists(), "live concurrent call workspace was deleted"
            assert (live / "in_flight.txt").exists()
        finally:
            child.terminate()
            child.wait(timeout=5)

    def test_runtime_failure_is_classified_for_hermes_fallback(self):
        from agent.agy_cli_client import AgyCLITransportError
        from agent.auxiliary_client import _is_connection_error

        error = AgyCLITransportError("agy CLI failed with non-zero exit status 2")
        assert _is_connection_error(error) is True

    def test_timeout_is_a_real_timeout_error(self):
        from agent.agy_cli_client import AgyCLITimeout
        from agent.auxiliary_client import _is_timeout_error

        assert _is_timeout_error(AgyCLITimeout("agy timed out")) is True


# ── private log, sandbox & cleanup (direct/small-prompt path) ────────────


class TestAgyPrivacyAndCleanup:
    def test_small_prompt_log_is_private_0600(self):
        from agent.agy_cli_client import AgyCLIClient

        client = AgyCLIClient(command="/usr/bin/agy")
        observed = {}

        def fake_run(argv, **kwargs):
            log_path = Path(argv[argv.index("--log-file") + 1])
            # The exclusive --log-file is precreated 0600 before agy runs, and
            # its parent workspace stays 0700.
            observed["log_mode"] = log_path.stat().st_mode & 0o777
            observed["parent_mode"] = log_path.parent.stat().st_mode & 0o777
            observed["cwd"] = kwargs["cwd"]
            assert kwargs.get("shell") is False
            assert "--sandbox" in argv
            assert "--dangerously-skip-permissions" not in argv
            return _ok_proc("compressed")

        with patch("agent.agy_cli_client.subprocess.run", side_effect=fake_run):
            client.chat.completions.create(
                model="m",
                messages=[{"role": "user", "content": "small prompt"}],
                timeout=10,
            )

        assert observed["log_mode"] == 0o600
        assert observed["parent_mode"] == 0o700
        # Workspace is removed after a normal completion.
        assert not Path(observed["cwd"]).exists()

    def test_small_prompt_tool_attempt_fails_closed(self):
        from agent.agy_cli_client import AgyCLIClient, AgyCLITransportError

        client = AgyCLIClient(command="/usr/bin/agy")

        def fake_run(argv, **kwargs):
            log_path = Path(argv[argv.index("--log-file") + 1])
            log_path.write_text(
                "CORTEX_STEP_TYPE_TOOL_CALL requested\n",
                encoding="utf-8",
            )
            return _ok_proc("this output must be rejected")

        with patch("agent.agy_cli_client.subprocess.run", side_effect=fake_run):
            with pytest.raises(AgyCLITransportError, match="tool"):
                client.chat.completions.create(
                    model="m",
                    messages=[{"role": "user", "content": "small prompt"}],
                    timeout=10,
                )

    def test_failure_path_removes_workspace(self):
        from agent.agy_cli_client import AgyCLIClient, AgyCLITransportError

        client = AgyCLIClient(command="/usr/bin/agy")
        observed = {}

        def fake_run(argv, **kwargs):
            observed["cwd"] = kwargs["cwd"]
            observed["log"] = argv[argv.index("--log-file") + 1]
            bad = _ok_proc(stdout="", returncode=2)
            bad.stderr = "boom"
            return bad

        with patch("agent.agy_cli_client.subprocess.run", side_effect=fake_run):
            with pytest.raises(AgyCLITransportError):
                client.chat.completions.create(
                    model="m",
                    messages=[{"role": "user", "content": "x"}],
                    timeout=10,
                )

        # Temporary HOME/cwd and its private log are removed even on failure.
        assert not Path(observed["cwd"]).exists()
        assert not Path(observed["log"]).exists()

    def test_split_utf8_preserves_multibyte_and_bounds(self):
        from agent.agy_cli_client import _CHUNK_PAYLOAD_BYTES, _split_text_utf8

        # 4-byte code points straddle every candidate boundary; the splitter
        # must never cut inside a code point and must respect the byte cap.
        text = "🙂" * 5000  # 20_000 bytes, well over the 1_000-byte test cap
        chunks = _split_text_utf8(text, max_bytes=1000)

        assert len(chunks) > 1
        assert "".join(chunks) == text
        for chunk in chunks:
            encoded = chunk.encode("utf-8")
            assert len(encoded) <= 1000
            # Each chunk is independently valid UTF-8 (round-trips cleanly).
            assert encoded.decode("utf-8") == chunk

        # Empty input yields a single empty chunk; small input stays whole.
        assert _split_text_utf8("") == [""]
        assert _split_text_utf8("tiny", max_bytes=_CHUNK_PAYLOAD_BYTES) == ["tiny"]
