"""Tests for optional Context Fabric filtering of Memory Fabric prefetch context."""
from __future__ import annotations

import json
import sys

from agent.context_fabric_memory import (
    extract_memory_fabric_payload,
    infer_project_from_payload,
    maybe_filter_memory_context,
)


def test_extract_memory_fabric_payload_from_fenced_provider_text() -> None:
    raw = (
        "prefix\n[Memory Fabric scoped context]\n"
        '{"has_data": true, "rendered": "# Context", "graph_context": {"rendered": "graph"}}\n'
        "suffix"
    )

    payload = extract_memory_fabric_payload(raw)

    assert payload == {"has_data": True, "rendered": "# Context", "graph_context": {"rendered": "graph"}}


def test_context_fabric_filter_invokes_configured_preflight_command(tmp_path, monkeypatch) -> None:
    command = tmp_path / "fake_preflight.py"
    command.write_text(
        "import json, pathlib, sys\n"
        "args = sys.argv\n"
        "payload = json.loads(pathlib.Path(args[args.index('--memory-fabric-json') + 1]).read_text())\n"
        "assert payload['memory_ids'] == ['mem-1']\n"
        "assert 'graph_context' in payload\n"
        "print('# AGENT_CONTEXT')\n"
        "print('Project: context-fabric')\n"
        "print('Channel: #context-fabric')\n"
        "print('filtered memory only')\n"
    )
    raw = "[Memory Fabric scoped context]\n" + json.dumps(
        {
            "has_data": True,
            "rendered": "# Context\n- useful memory",
            "memory_ids": ["mem-1"],
            "graph_context": {"rendered": "graph should not leak"},
        }
    )
    config = {
        "enabled": True,
        "command": [sys.executable, str(command)],
        "default_project": "context-fabric",
        "default_channel": "#context-fabric",
        "timeout_seconds": 5,
    }

    filtered = maybe_filter_memory_context(raw, "build it", config=config)

    assert filtered.startswith("# AGENT_CONTEXT")
    assert "filtered memory only" in filtered
    assert "graph should not leak" not in filtered


def test_context_fabric_filter_fails_open_when_disabled() -> None:
    raw = "[Memory Fabric scoped context]\n{\"has_data\": true}"

    assert maybe_filter_memory_context(raw, "query", config={"enabled": False}) == raw


def test_infer_project_from_memory_fabric_rendered_scope() -> None:
    payload = {"rendered": "# Context for: x\n\n## scope: jarvis-metrics\n- fact"}

    assert infer_project_from_payload(payload) == "jarvis-metrics"


def test_infer_project_from_current_channel_when_no_route_or_payload_scope(monkeypatch) -> None:
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(platform="discord", chat_id="new-channel-id", chat_name="Hermes / #new-client-channel")
    try:
        raw = "[Memory Fabric scoped context]\n" + json.dumps({"has_data": True, "rendered": "# Context\n- fact"})
        seen: dict[str, list[str]] = {}

        def fake_run(args, **kwargs):
            seen["args"] = args

            class Completed:
                returncode = 0
                stdout = "# AGENT_CONTEXT\nProject: new-client-channel\nChannel: #new-client-channel\n"
                stderr = ""

            return Completed()

        monkeypatch.setattr("agent.context_fabric_memory.subprocess.run", fake_run)
        filtered = maybe_filter_memory_context(
            raw,
            "query",
            config={"enabled": True, "command": [sys.executable, "fake.py"], "timeout_seconds": 5},
        )
    finally:
        clear_session_vars(tokens)

    assert "Project: new-client-channel" in filtered
    assert seen["args"][seen["args"].index("--project") + 1] == "new-client-channel"
    assert seen["args"][seen["args"].index("--channel") + 1] == "#new-client-channel"
