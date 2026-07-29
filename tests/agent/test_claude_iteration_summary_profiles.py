import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.chat_completion_helpers import (
    _apply_claude_execution_envelope_to_direct_kwargs,
    handle_max_iterations,
)
from agent.claude_execution_profiles import ClaudeExecutionProfileError


def _repo_context():
    root = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    return root, sha


@pytest.mark.parametrize("profile", ["probe", "review_bundle", "code"])
def test_direct_summary_helper_canonicalizes_all_profiles_top_level(profile):
    root, sha = _repo_context()
    overrides = {"execution_profile": profile}
    if profile in {"review_bundle", "code"}:
        overrides.update(project="delivery-2", candidate_sha=sha)
    if profile == "code":
        overrides.update(cwd=str(root), skills=["test-driven-development"], rules=["No deploy"])

    agent = SimpleNamespace(
        provider="claude-sdk-local",
        base_url="http://127.0.0.1:4318/v1",
        request_overrides=overrides,
    )
    kwargs = _apply_claude_execution_envelope_to_direct_kwargs(agent, {"model": "claude"})

    assert kwargs["extra_body"]["profile"] == profile
    assert "profile" not in kwargs
    assert "execution_profile" not in kwargs
    if profile != "probe":
        assert kwargs["extra_body"]["candidate_sha"] == sha
    if profile == "code":
        assert kwargs["extra_body"]["cwd"] == str(root)


def _summary_agent(request_overrides):
    transport = MagicMock()
    transport.normalize_response.side_effect = lambda response: SimpleNamespace(content=response)
    client = MagicMock()
    client.chat.completions.create.side_effect = ["", "retry complete"]
    return SimpleNamespace(
        max_iterations=1,
        _should_sanitize_tool_calls=lambda: False,
        _copy_reasoning_content_for_api=lambda source, target: None,
        _sanitize_api_messages=lambda messages: messages,
        _drop_thinking_only_and_merge_users=lambda messages: messages,
        _cached_system_prompt="",
        ephemeral_system_prompt="",
        prefill_messages=[],
        model="claude-sonnet-5",
        base_url="http://127.0.0.1:4318/v1",
        _base_url_lower="http://127.0.0.1:4318/v1",
        provider="claude-sdk-local",
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        provider_require_parameters=False,
        provider_data_collection=None,
        request_overrides=request_overrides,
        reasoning_config=None,
        api_mode="chat_completions",
        max_tokens=100,
        _max_tokens_param=lambda value: {"max_tokens": value},
        _supports_reasoning_extra_body=lambda: False,
        _is_openrouter_url=lambda: False,
        openrouter_min_coding_score=None,
        session_id="summary-smoke",
        _get_transport=lambda: transport,
        _ensure_primary_openai_client=MagicMock(return_value=client),
        _client=client,
    )


def test_iteration_summary_missing_profile_never_obtains_http_client():
    agent = _summary_agent({})
    result = handle_max_iterations(agent, [{"role": "user", "content": "done?"}], 1)

    assert "execution_profile is required" in result
    agent._ensure_primary_openai_client.assert_not_called()


def test_iteration_summary_and_retry_send_probe_profile_top_level():
    agent = _summary_agent({"execution_profile": "probe"})
    result = handle_max_iterations(agent, [{"role": "user", "content": "done?"}], 1)

    assert result == "retry complete"
    calls = agent._client.chat.completions.create.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert call.kwargs["extra_body"]["profile"] == "probe"
        assert "profile" not in call.kwargs
    assert all("execution_profile" not in call.kwargs for call in calls)
