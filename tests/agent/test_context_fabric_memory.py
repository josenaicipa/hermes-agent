"""Tests for optional Context Fabric filtering of Memory Fabric prefetch context."""
from __future__ import annotations

import json
import subprocess
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


def test_context_fabric_filter_fails_open_without_command() -> None:
    raw = "[Memory Fabric scoped context]\n{\"has_data\": true, \"project\": \"x\"}"

    assert maybe_filter_memory_context(raw, "query", config={"enabled": True, "default_channel": "#x"}) == raw


def test_context_fabric_filter_fails_open_when_restricted_route_missing() -> None:
    raw = "[Memory Fabric scoped context]\n{\"has_data\": true, \"project\": \"x\"}"

    assert maybe_filter_memory_context(
        raw,
        "query",
        config={"enabled": True, "command": [sys.executable, "fake.py"], "restrict_to_routes": True, "default_channel": "#x"},
    ) == raw


def test_context_fabric_filter_fails_open_on_subprocess_errors(monkeypatch) -> None:
    raw = "[Memory Fabric scoped context]\n" + json.dumps({"has_data": True, "project": "x", "rendered": "# Context"})
    config = {"enabled": True, "command": [sys.executable, "fake.py"], "default_channel": "#x"}

    class Failed:
        returncode = 1
        stdout = ""
        stderr = "secret-looking stderr should not leak"

    monkeypatch.setattr("agent.context_fabric_memory.subprocess.run", lambda *args, **kwargs: Failed())
    assert maybe_filter_memory_context(raw, "query", config=config) == raw

    class Empty:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr("agent.context_fabric_memory.subprocess.run", lambda *args, **kwargs: Empty())
    assert maybe_filter_memory_context(raw, "query", config=config) == raw

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="fake", timeout=3)

    monkeypatch.setattr("agent.context_fabric_memory.subprocess.run", raise_timeout)
    assert maybe_filter_memory_context(raw, "query", config=config) == raw


def test_infer_project_from_memory_fabric_rendered_scope() -> None:
    payload = {"rendered": "# Context for: x\n\n## scope: jarvis-metrics\n- fact"}

    assert infer_project_from_payload(payload) == "jarvis-metrics"


def test_infer_project_from_memory_fabric_graph_route_for_channel() -> None:
    payload = {
        "rendered": "# Context\n(no relevant memories found)",
        "graph_context": {
            "semantic_relations": [
                {"from": "hermes-updates", "relation": "routes_to", "to": "hermes"},
                {"from": "context-fabric", "relation": "routes_to", "to": "context-fabric"},
            ]
        },
    }

    assert infer_project_from_payload(payload, channel="#hermes-updates") == "hermes"


def test_context_fabric_filter_passes_route_skills_and_toolsets(monkeypatch) -> None:
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(platform="discord", chat_id="n8n-channel-id", chat_name="Hermes / #n8n", guild_id="guild-1")
    try:
        raw = "[Memory Fabric scoped context]\n" + json.dumps({"has_data": True, "rendered": "# Context\n- workflow facts"})
        seen: dict[str, list[str]] = {}

        def fake_run(args, **kwargs):
            seen["args"] = args

            class Completed:
                returncode = 0
                stdout = "# AGENT_CONTEXT\nProject: n8n\nChannel: #n8n\n"
                stderr = ""

            return Completed()

        monkeypatch.setattr("agent.context_fabric_memory.subprocess.run", fake_run)
        maybe_filter_memory_context(
            raw,
            "debug workflow failure",
            config={
                "enabled": True,
                "command": [sys.executable, "fake.py"],
                "timeout_seconds": 5,
                "channel_routes": {
                    "#n8n": {
                        "project": "n8n",
                        "channel": "#n8n",
                        "required_skills": ["n8n-operations"],
                        "fallback_skills": ["terminal-ops", "github-ops"],
                        "enabled_toolsets": ["terminal", "n8n"],
                        "task_type": "automation-debug",
                        "budget_profile": "deep",
                    }
                },
            },
            compaction_count=2,
        )
    finally:
        clear_session_vars(tokens)

    assert seen["args"][seen["args"].index("--required-skills") + 1] == "n8n-operations"
    assert seen["args"][seen["args"].index("--fallback-skills") + 1] == "terminal-ops,github-ops"
    assert seen["args"][seen["args"].index("--enabled-toolsets") + 1] == "terminal,n8n"
    assert seen["args"][seen["args"].index("--task-type") + 1] == "automation-debug"
    assert seen["args"][seen["args"].index("--budget-profile") + 1] == "deep"
    assert seen["args"][seen["args"].index("--guild-id") + 1] == "guild-1"
    assert seen["args"][seen["args"].index("--channel-id") + 1] == "n8n-channel-id"
    assert seen["args"][seen["args"].index("--channel-name") + 1] == "Hermes / #n8n"
    assert seen["args"][seen["args"].index("--compaction-count") + 1] == "2"


def test_context_fabric_filter_uses_graph_route_before_channel_slug(monkeypatch) -> None:
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(platform="discord", chat_id="1505614493932326983", chat_name="Hermes / #hermes-updates")
    try:
        raw = "[Memory Fabric scoped context]\n" + json.dumps(
            {
                "has_data": False,
                "rendered": "# Context\n(no relevant memories found)",
                "graph_context": {
                    "semantic_relations": [
                        {"from": "hermes-updates", "relation": "routes_to", "to": "hermes"},
                    ]
                },
            }
        )
        seen: dict[str, list[str]] = {}

        def fake_run(args, **kwargs):
            seen["args"] = args

            class Completed:
                returncode = 0
                stdout = "# AGENT_CONTEXT\nProject: hermes\nChannel: #hermes-updates\n"
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

    assert "Project: hermes" in filtered
    assert seen["args"][seen["args"].index("--project") + 1] == "hermes"
    assert seen["args"][seen["args"].index("--channel") + 1] == "#hermes-updates"


def test_context_fabric_filter_routes_explicit_discord_channel_mention(monkeypatch) -> None:
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(
        platform="discord",
        chat_id="1513706178843115660",
        chat_name="Hermes / #context-fabric",
        guild_id="guild-1",
    )
    try:
        target_id = "1521226961756749825"
        raw = "[Memory Fabric scoped context]\n" + json.dumps(
            {
                "query": f"audita Context Fabric para <#{target_id}>",
                "has_data": False,
                "rendered": "# Context\n(no relevant memories found)",
                "memory_ids": [],
            }
        )
        seen: dict[str, list[str]] = {}

        def fake_run(args, **kwargs):
            seen["args"] = args

            class Completed:
                returncode = 0
                stdout = "# AGENT_CONTEXT\nProject: vexa\nChannel: #vexa\n"
                stderr = ""

            return Completed()

        monkeypatch.setattr("agent.context_fabric_memory.subprocess.run", fake_run)
        filtered = maybe_filter_memory_context(
            raw,
            f"audita Context Fabric para <#{target_id}>",
            config={
                "enabled": True,
                "command": [sys.executable, "fake.py"],
                "timeout_seconds": 5,
                "channel_routes": {
                    "#context-fabric": {"project": "context-fabric", "channel": "#context-fabric"},
                    "1513706178843115660": {"project": "context-fabric", "channel": "#context-fabric"},
                    "#vexa": {"project": "vexa", "channel": "#vexa"},
                    target_id: {"project": "vexa", "channel": "#vexa"},
                },
            },
        )
    finally:
        clear_session_vars(tokens)

    assert "Project: vexa" in filtered
    assert seen["args"][seen["args"].index("--project") + 1] == "vexa"
    assert seen["args"][seen["args"].index("--channel") + 1] == "#vexa"
    assert seen["args"][seen["args"].index("--channel-id") + 1] == target_id
    assert seen["args"][seen["args"].index("--channel-name") + 1] == "#vexa"


def test_incidental_bare_channel_name_in_prose_does_not_reroute(monkeypatch) -> None:
    """Blocker #2 guard: a bare ``#name`` in ordinary conversation must NOT hijack
    routing away from a valid session channel. Only the explicit Discord ``<#id>``
    mention (or an opt-in flag) may override the current chat's scope. Otherwise a
    sentence like "reminds me of #control" would silently inject another project's
    memory into the turn — the exact cross-project contamination Jose forbids.
    """
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(
        platform="discord",
        chat_id="ecom-session-id",
        chat_name="Hermes / #ecommerce360",
        guild_id="guild-1",
    )
    try:
        raw = "[Memory Fabric scoped context]\n" + json.dumps(
            {
                "query": "esto me recuerda al issue que tuvimos en #control la semana pasada",
                "has_data": False,
                "rendered": "# Context\n(no relevant memories found)",
                "memory_ids": [],
            }
        )
        seen: dict[str, list[str]] = {}

        def fake_run(args, **kwargs):
            seen["args"] = args

            class Completed:
                returncode = 0
                stdout = "# AGENT_CONTEXT\nProject: ecommerce360\nChannel: #ecommerce360\n"
                stderr = ""

            return Completed()

        monkeypatch.setattr("agent.context_fabric_memory.subprocess.run", fake_run)
        filtered = maybe_filter_memory_context(
            raw,
            "esto me recuerda al issue que tuvimos en #control la semana pasada",
            config={
                "enabled": True,
                "command": [sys.executable, "fake.py"],
                "timeout_seconds": 5,
                # NOTE: allow_bare_query_channel_route intentionally NOT set (default off).
                "channel_routes": {
                    "#ecommerce360": {"project": "ecommerce360", "channel": "#ecommerce360"},
                    "#control": {"project": "torre-de-control", "channel": "#control"},
                },
            },
        )
    finally:
        clear_session_vars(tokens)

    # Routing must stay on the session channel, not the channel named in prose.
    assert seen["args"][seen["args"].index("--project") + 1] == "ecommerce360"
    assert seen["args"][seen["args"].index("--channel") + 1] == "#ecommerce360"
    assert "torre-de-control" not in seen["args"]
    assert "#control" not in seen["args"]


def test_bare_channel_name_reroutes_only_when_explicitly_opted_in(monkeypatch) -> None:
    """Counterpart to the guard above: when an operator explicitly opts in via
    ``allow_bare_query_channel_route``, a bare ``#name`` that keys into a configured
    route MAY scope the preflight. This keeps the audit ergonomics available without
    making bare-name rerouting the unsafe default.
    """
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(
        platform="discord",
        chat_id="cf-session-id",
        chat_name="Hermes / #context-fabric",
        guild_id="guild-1",
    )
    try:
        raw = "[Memory Fabric scoped context]\n" + json.dumps(
            {
                "query": "audita #vexa",
                "has_data": False,
                "rendered": "# Context\n(no relevant memories found)",
                "memory_ids": [],
            }
        )
        seen: dict[str, list[str]] = {}

        def fake_run(args, **kwargs):
            seen["args"] = args

            class Completed:
                returncode = 0
                stdout = "# AGENT_CONTEXT\nProject: vexa\nChannel: #vexa\n"
                stderr = ""

            return Completed()

        monkeypatch.setattr("agent.context_fabric_memory.subprocess.run", fake_run)
        filtered = maybe_filter_memory_context(
            raw,
            "audita #vexa",
            config={
                "enabled": True,
                "command": [sys.executable, "fake.py"],
                "timeout_seconds": 5,
                "allow_bare_query_channel_route": True,
                "channel_routes": {
                    "#context-fabric": {"project": "context-fabric", "channel": "#context-fabric"},
                    "#vexa": {"project": "vexa", "channel": "#vexa"},
                },
            },
        )
    finally:
        clear_session_vars(tokens)

    assert "Project: vexa" in filtered
    assert seen["args"][seen["args"].index("--project") + 1] == "vexa"
    assert seen["args"][seen["args"].index("--channel") + 1] == "#vexa"


def test_context_fabric_filter_does_not_route_incidental_discord_channel_mention(monkeypatch) -> None:
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(
        platform="discord",
        chat_id="1513706178843115660",
        chat_name="Hermes / #context-fabric",
        guild_id="guild-1",
    )
    try:
        target_id = "1521226961756749825"
        raw = "[Memory Fabric scoped context]\n" + json.dumps(
            {
                "query": f"esto se parece a <#{target_id}> pero no es una petición de contexto",
                "has_data": False,
                "rendered": "# Context\n(no relevant memories found)",
                "memory_ids": [],
            }
        )
        seen: dict[str, list[str]] = {}

        def fake_run(args, **kwargs):
            seen["args"] = args

            class Completed:
                returncode = 0
                stdout = "# AGENT_CONTEXT\nProject: context-fabric\nChannel: #context-fabric\n"
                stderr = ""

            return Completed()

        monkeypatch.setattr("agent.context_fabric_memory.subprocess.run", fake_run)
        filtered = maybe_filter_memory_context(
            raw,
            f"esto se parece a <#{target_id}> pero no es una petición de contexto",
            config={
                "enabled": True,
                "command": [sys.executable, "fake.py"],
                "timeout_seconds": 5,
                "channel_routes": {
                    "#context-fabric": {"project": "context-fabric", "channel": "#context-fabric"},
                    "1513706178843115660": {"project": "context-fabric", "channel": "#context-fabric"},
                    target_id: {"project": "vexa", "channel": "#vexa"},
                },
            },
        )
    finally:
        clear_session_vars(tokens)

    assert "Project: context-fabric" in filtered
    assert seen["args"][seen["args"].index("--project") + 1] == "context-fabric"
    assert seen["args"][seen["args"].index("--channel") + 1] == "#context-fabric"



def test_context_fabric_filter_does_not_route_incidental_bare_channel_name(monkeypatch) -> None:
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(
        platform="discord",
        chat_id="1513706178843115660",
        chat_name="Hermes / #context-fabric",
        guild_id="guild-1",
    )
    try:
        raw = "[Memory Fabric scoped context]\n" + json.dumps(
            {
                "query": "esto se parece a #vexa pero no cambies de canal",
                "has_data": False,
                "rendered": "# Context\n(no relevant memories found)",
                "memory_ids": [],
            }
        )
        seen: dict[str, list[str]] = {}

        def fake_run(args, **kwargs):
            seen["args"] = args

            class Completed:
                returncode = 0
                stdout = "# AGENT_CONTEXT\nProject: context-fabric\nChannel: #context-fabric\n"
                stderr = ""

            return Completed()

        monkeypatch.setattr("agent.context_fabric_memory.subprocess.run", fake_run)
        filtered = maybe_filter_memory_context(
            raw,
            "esto se parece a #vexa pero no cambies de canal",
            config={
                "enabled": True,
                "command": [sys.executable, "fake.py"],
                "timeout_seconds": 5,
                "channel_routes": {
                    "#context-fabric": {"project": "context-fabric", "channel": "#context-fabric"},
                    "1513706178843115660": {"project": "context-fabric", "channel": "#context-fabric"},
                    "#vexa": {"project": "vexa", "channel": "#vexa"},
                },
            },
        )
    finally:
        clear_session_vars(tokens)

    assert "Project: context-fabric" in filtered
    assert seen["args"][seen["args"].index("--project") + 1] == "context-fabric"
    assert seen["args"][seen["args"].index("--channel") + 1] == "#context-fabric"



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
