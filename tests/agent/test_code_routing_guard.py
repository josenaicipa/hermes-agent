"""Tests for the hard code-routing guard (Phase 2 tool-level enforcement).

The guard blocks direct Hermes use of code-mutation/execution tools when the
request is code work and no Claude/Fable delegation was attempted, unless an
explicit exception applies.  See agent/code_routing_guard.py.
"""

from agent.code_routing_guard import (
    CodeRoutingGuardConfig,
    evaluate_code_routing_guard,
    has_delegation_evidence,
    is_code_request,
    is_code_terminal_command,
    is_explicit_direct_edit,
    latest_user_request,
)


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _delegate_call() -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "tc1", "type": "function", "function": {"name": "delegate_task", "arguments": "{}"}}
        ],
    }


# ── request classification ────────────────────────────────────────────────


class TestIsCodeRequest:
    def test_strong_verbs_match(self):
        assert is_code_request("Please implement the auth flow")
        assert is_code_request("refactor this module")
        assert is_code_request("debug the failing run")
        assert is_code_request("deploy the service to prod")
        assert is_code_request("do a technical audit of the repo")

    def test_weak_verb_needs_code_noun(self):
        assert is_code_request("fix the build")
        assert is_code_request("add an endpoint to the API")
        assert is_code_request("update the migration script")

    def test_non_code_requests_do_not_match(self):
        assert not is_code_request("add a calendar invite for Tuesday")
        assert not is_code_request("build a marketing campaign plan")
        assert not is_code_request("summarize this article and save my notes")
        assert not is_code_request("what's the weather today?")
        assert not is_code_request("")
        assert not is_code_request(None)


class TestIsCodeTerminalCommand:
    def test_build_and_deploy_commands_match(self):
        assert is_code_terminal_command("npm run build")
        assert is_code_terminal_command("pytest tests/")
        assert is_code_terminal_command("cargo build --release")
        assert is_code_terminal_command("docker build -t app .")
        assert is_code_terminal_command("git commit -m 'x'")
        assert is_code_terminal_command("python -m build")
        assert is_code_terminal_command("ls && make all")

    def test_read_only_commands_do_not_match(self):
        assert not is_code_terminal_command("ls -la")
        assert not is_code_terminal_command("cat README.md")
        assert not is_code_terminal_command("git status")
        assert not is_code_terminal_command("cd /tmp")
        assert not is_code_terminal_command("")


class TestExceptionDetectors:
    def test_explicit_direct_edit(self):
        assert is_explicit_direct_edit("just edit it yourself")
        assert is_explicit_direct_edit("skip Fable and do it directly")
        assert is_explicit_direct_edit("this is an emergency, patch this now")
        assert is_explicit_direct_edit("don't delegate, you fix it")
        assert not is_explicit_direct_edit("please implement the feature")

    def test_addressing_hermes_by_name_is_not_an_override(self):
        # Merely naming Hermes is an ordinary request, not a direct-edit override.
        assert not is_explicit_direct_edit("Hermes, implement the login endpoint")
        assert not is_explicit_direct_edit("Hermes, fix the failing test")
        # An actual override still works even when addressed to Hermes.
        assert is_explicit_direct_edit("Hermes, edit it yourself")

    def test_bare_emergency_word_is_not_an_override(self):
        # A code request that merely contains "emergency" must still route.
        assert not is_explicit_direct_edit("implement the emergency-stop endpoint")
        assert not is_explicit_direct_edit("add an emergency shutdown route")
        assert not is_explicit_direct_edit("we had an emergency, fix the failing deploy")
        # Explicit emergency-override phrasing does lift the guard.
        assert is_explicit_direct_edit("this is an emergency, skip Fable")
        assert is_explicit_direct_edit("it's an emergency, patch it now")
        assert is_explicit_direct_edit("emergency override: edit it directly")

    def test_delegation_evidence_from_tool_call(self):
        assert has_delegation_evidence([_user("implement x"), _delegate_call()])

    def test_delegation_evidence_from_failure_text(self):
        msgs = [
            _user("implement x"),
            {"role": "assistant", "content": "Claude (Fable) is unavailable right now, falling back."},
        ]
        assert has_delegation_evidence(msgs)

    def test_no_delegation_evidence(self):
        assert not has_delegation_evidence([_user("implement x")])
        assert not has_delegation_evidence([])
        assert not has_delegation_evidence(None)

    def test_stale_delegation_before_newer_request_does_not_count(self):
        # Delegation happened for an earlier request; a newer user request has
        # arrived since. The stale evidence must not unlock the new request.
        msgs = [
            _user("implement x"),
            _delegate_call(),
            _user("implement y"),
        ]
        assert not has_delegation_evidence(msgs)

    def test_delegation_after_latest_request_counts(self):
        # Delegation dispatched in response to the latest user request counts.
        msgs = [
            _user("implement x"),
            _user("implement y"),
            _delegate_call(),
        ]
        assert has_delegation_evidence(msgs)

    def test_latest_user_request_picks_most_recent(self):
        msgs = [_user("first"), {"role": "assistant", "content": "ok"}, _user("second")]
        assert latest_user_request(msgs) == "second"


# ── end-to-end guard decisions ─────────────────────────────────────────────


class TestEvaluateGuard:
    def test_code_prompt_blocks_write_file_without_delegation(self):
        msgs = [_user("implement the login endpoint")]
        decision = evaluate_code_routing_guard(
            tool_name="write_file",
            args={"path": "auth.py", "content": "..."},
            messages=msgs,
        )
        assert decision.blocked is True
        assert decision.code == "code_routing_block"
        assert "delegate" in decision.message.lower()
        assert "Claude (Fable)" in decision.message

    def test_addressing_hermes_by_name_still_blocks(self):
        # "Hermes, implement ..." is an ordinary code request — addressing Hermes
        # by name must not bypass the guard.
        decision = evaluate_code_routing_guard(
            tool_name="write_file",
            args={"path": "auth.py", "content": "..."},
            messages=[_user("Hermes, implement the login endpoint")],
        )
        assert decision.blocked is True
        assert decision.code == "code_routing_block"

    def test_bare_emergency_word_still_blocks(self):
        # "implement emergency-stop endpoint" contains "emergency" but is a plain
        # code request — it must still route to Fable.
        decision = evaluate_code_routing_guard(
            tool_name="write_file",
            args={"path": "shutdown.py", "content": "..."},
            messages=[_user("implement the emergency-stop endpoint")],
        )
        assert decision.blocked is True
        assert decision.code == "code_routing_block"

    def test_explicit_emergency_override_allows(self):
        decision = evaluate_code_routing_guard(
            tool_name="write_file",
            args={"path": "shutdown.py", "content": "..."},
            messages=[_user("this is an emergency, skip Fable and fix it")],
        )
        assert decision.blocked is False
        assert decision.code == "explicit_direct_edit_exception"

    def test_stale_delegation_before_newer_request_still_blocks(self):
        # Delegation for an earlier request does not unlock a newer code request.
        msgs = [
            _user("implement the login endpoint"),
            _delegate_call(),
            _user("now implement the logout endpoint"),
        ]
        decision = evaluate_code_routing_guard(
            tool_name="write_file",
            args={"path": "auth.py", "content": "..."},
            messages=msgs,
        )
        assert decision.blocked is True

    def test_delegation_after_latest_request_allows(self):
        msgs = [
            _user("implement the login endpoint"),
            _user("now implement the logout endpoint"),
            _delegate_call(),
        ]
        decision = evaluate_code_routing_guard(
            tool_name="write_file",
            args={"path": "auth.py", "content": "..."},
            messages=msgs,
        )
        assert decision.blocked is False
        assert decision.code == "delegation_evidence_exception"

    def test_code_prompt_blocks_patch_without_delegation(self):
        decision = evaluate_code_routing_guard(
            tool_name="patch",
            args={"path": "main.go"},
            messages=[_user("fix the bug in the handler")],
        )
        assert decision.blocked is True

    def test_code_prompt_blocks_build_terminal_command(self):
        decision = evaluate_code_routing_guard(
            tool_name="terminal",
            args={"command": "npm run build"},
            messages=[_user("implement the dashboard feature")],
        )
        assert decision.blocked is True

    def test_terminal_readonly_command_allowed_even_for_code_prompt(self):
        decision = evaluate_code_routing_guard(
            tool_name="terminal",
            args={"command": "git status"},
            messages=[_user("implement the dashboard feature")],
        )
        assert decision.blocked is False

    def test_non_code_prompt_allows_write_file(self):
        decision = evaluate_code_routing_guard(
            tool_name="write_file",
            args={"path": "notes.md", "content": "meeting notes"},
            messages=[_user("save my meeting notes")],
        )
        assert decision.blocked is False

    def test_explicit_direct_edit_exception_allows(self):
        decision = evaluate_code_routing_guard(
            tool_name="write_file",
            args={"path": "auth.py"},
            messages=[_user("implement the login endpoint yourself, skip Fable")],
        )
        assert decision.blocked is False
        assert decision.code == "explicit_direct_edit_exception"

    def test_delegation_evidence_allows(self):
        msgs = [_user("implement the login endpoint"), _delegate_call()]
        decision = evaluate_code_routing_guard(
            tool_name="write_file",
            args={"path": "auth.py"},
            messages=msgs,
        )
        assert decision.blocked is False
        assert decision.code == "delegation_evidence_exception"

    def test_disabled_config_allows_everything(self):
        decision = evaluate_code_routing_guard(
            tool_name="write_file",
            args={"path": "auth.py"},
            messages=[_user("implement the login endpoint")],
            config=CodeRoutingGuardConfig(enabled=False),
        )
        assert decision.blocked is False

    def test_subagent_is_never_guarded(self):
        decision = evaluate_code_routing_guard(
            tool_name="write_file",
            args={"path": "auth.py"},
            messages=[_user("implement the login endpoint")],
            is_subagent=True,
        )
        assert decision.blocked is False

    def test_unguarded_tool_allowed(self):
        decision = evaluate_code_routing_guard(
            tool_name="read_file",
            args={"path": "auth.py"},
            messages=[_user("implement the login endpoint")],
        )
        assert decision.blocked is False


class TestConfigFromValue:
    def test_bool_true(self):
        assert CodeRoutingGuardConfig.from_value(True).enabled is True

    def test_bool_false(self):
        assert CodeRoutingGuardConfig.from_value(False).enabled is False

    def test_none_defaults_on(self):
        assert CodeRoutingGuardConfig.from_value(None).enabled is True

    def test_mapping_with_custom_tools(self):
        cfg = CodeRoutingGuardConfig.from_value(
            {"enabled": True, "guarded_tools": ["write_file", "patch"]}
        )
        assert cfg.enabled is True
        assert cfg.guarded_tools == frozenset({"write_file", "patch"})

    def test_mapping_disabled(self):
        assert CodeRoutingGuardConfig.from_value({"enabled": False}).enabled is False
