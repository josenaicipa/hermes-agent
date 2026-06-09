"""Wiring tests for AIAgent._code_routing_block_message.

Exercises the runtime adapter (the method tool_executor calls) rather than the
pure guard logic — subagent exemption, config gating, and fail-open behavior.
The pure decision logic is covered in tests/agent/test_code_routing_guard.py.
"""

from types import SimpleNamespace

from run_agent import AIAgent
from agent.code_routing_guard import CodeRoutingGuardConfig


def _fake_agent(*, enabled=True, delegate_depth=0, config="default"):
    if config == "default":
        config = CodeRoutingGuardConfig(enabled=enabled)
    return SimpleNamespace(
        _code_routing_guard_config=config,
        _delegate_depth=delegate_depth,
    )


_CODE_MSGS = [{"role": "user", "content": "implement the login endpoint"}]


def _call(agent, name="write_file", args=None, messages=None):
    return AIAgent._code_routing_block_message(
        agent,
        name,
        args if args is not None else {"path": "auth.py", "content": "x"},
        messages if messages is not None else _CODE_MSGS,
    )


def test_blocks_direct_code_edit_when_enabled():
    msg = _call(_fake_agent())
    assert msg is not None
    assert "hard code-routing guard" in msg


def test_config_disables_guard():
    assert _call(_fake_agent(enabled=False)) is None


def test_subagent_is_exempt():
    assert _call(_fake_agent(delegate_depth=1)) is None


def test_non_code_prompt_allowed():
    msgs = [{"role": "user", "content": "save my meeting notes"}]
    assert _call(_fake_agent(), args={"path": "notes.md", "content": "x"}, messages=msgs) is None


def test_delegation_evidence_allows():
    msgs = [
        {"role": "user", "content": "implement the login endpoint"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t", "type": "function", "function": {"name": "delegate_task", "arguments": "{}"}}
            ],
        },
    ]
    assert _call(_fake_agent(), messages=msgs) is None


def test_fail_open_when_config_missing_attr():
    # No _code_routing_guard_config attribute → falls back to default-on config.
    agent = SimpleNamespace(_delegate_depth=0)
    assert _call(agent) is not None


def test_fail_open_on_bad_messages():
    # Malformed messages must never raise out of the guard.
    agent = _fake_agent()
    assert AIAgent._code_routing_block_message(agent, "write_file", {"path": "a.py"}, "not-a-list") is None
