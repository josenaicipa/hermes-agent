import subprocess

import pytest

from agent.claude_execution_profiles import (
    ClaudeExecutionProfileError,
    normalize_claude_execution_envelope,
)
from agent.transports.chat_completions import ChatCompletionsTransport


def _messages():
    return [{"role": "user", "content": "smoke"}]


def test_transport_fails_locally_for_unprofiled_primary_and_backup_bridge():
    transport = ChatCompletionsTransport()
    for port in (4318, 4319):
        with pytest.raises(ClaudeExecutionProfileError, match="execution_profile is required"):
            transport.build_kwargs(
                model="claude-sonnet-5",
                messages=_messages(),
                base_url=f"http://127.0.0.1:{port}/v1",
            )


def test_transport_forwards_explicit_probe_profile_top_level():
    kwargs = ChatCompletionsTransport().build_kwargs(
        model="claude-sonnet-5",
        messages=_messages(),
        base_url="http://127.0.0.1:4318/v1",
        request_overrides={"execution_profile": "probe"},
    )
    # OpenAI Python accepts custom HTTP fields through extra_body; it merges
    # them into the top-level JSON body sent to the bridge.
    assert kwargs["extra_body"]["profile"] == "probe"
    assert "execution_profile" not in kwargs
    assert "profile" not in kwargs


def test_non_bridge_transport_is_unchanged_without_profile():
    kwargs = ChatCompletionsTransport().build_kwargs(
        model="test-model",
        messages=_messages(),
        base_url="https://example.invalid/v1",
    )
    assert "profile" not in kwargs


def test_code_profile_requires_candidate_sha_to_equal_worktree_head(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    (tmp_path / "README.md").write_text("fixture\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "fixture"], check=True)

    with pytest.raises(ClaudeExecutionProfileError, match="must equal HEAD"):
        normalize_claude_execution_envelope(
            {
                "execution_profile": "code",
                "project": "delivery-2",
                "candidate_sha": "a" * 40,
                "cwd": str(tmp_path),
            }
        )
