"""
Regression tests for the shared-container task_id mapping.

The top-level agent and all delegate_task subagents share a single
terminal sandbox keyed by ``"default"``.  ``_resolve_container_task_id``
is the sole gatekeeper for which tool-call task_ids go to the shared
container vs. get their own isolated sandbox.  RL / benchmark
environments opt in to isolation by calling
``register_task_env_overrides(task_id, {...})`` before the agent loop;
every other task_id collapses back to ``"default"``.

If you change the collapse logic, update both the helper and these
tests -- see `hermes-agent-dev` skill, "Why do subagents get their own
containers?" section, and the Container lifecycle paragraph under
Docker Backend in ``website/docs/user-guide/configuration.md``.
"""

import pytest

from tools import terminal_tool


@pytest.fixture(autouse=True)
def _clean_overrides():
    """Ensure no stray overrides from other tests leak in."""
    before = dict(terminal_tool._task_env_overrides)
    terminal_tool._task_env_overrides.clear()
    yield
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(before)


def test_none_task_id_maps_to_default():
    assert terminal_tool._resolve_container_task_id(None) == "default"


def test_empty_task_id_maps_to_default():
    assert terminal_tool._resolve_container_task_id("") == "default"


def test_literal_default_stays_default():
    assert terminal_tool._resolve_container_task_id("default") == "default"


def test_subagent_task_id_collapses_to_default():
    # delegate_task constructs IDs like "subagent-<N>-<uuid_hex>"; these
    # should share the parent's container, not spin up their own.
    assert terminal_tool._resolve_container_task_id("subagent-0-deadbeef") == "default"
    assert terminal_tool._resolve_container_task_id("subagent-42-cafef00d") == "default"


def test_arbitrary_session_id_collapses_to_default():
    # Session UUIDs or anything else without an override still collapse.
    assert terminal_tool._resolve_container_task_id("sess-123e4567-e89b-12d3") == "default"


def test_rl_task_with_override_keeps_its_own_id():
    # RL / benchmark pattern: register a per-task image, then the task_id
    # must survive ``_resolve_container_task_id`` so the rollout lands in
    # its own sandbox.
    terminal_tool.register_task_env_overrides(
        "tb2-task-fix-git", {"docker_image": "tb2:fix-git", "cwd": "/app"}
    )
    try:
        assert (
            terminal_tool._resolve_container_task_id("tb2-task-fix-git")
            == "tb2-task-fix-git"
        )
    finally:
        terminal_tool.clear_task_env_overrides("tb2-task-fix-git")


def test_cleared_override_collapses_again():
    terminal_tool.register_task_env_overrides("tb2-x", {"docker_image": "x:y"})
    assert terminal_tool._resolve_container_task_id("tb2-x") == "tb2-x"
    terminal_tool.clear_task_env_overrides("tb2-x")
    assert terminal_tool._resolve_container_task_id("tb2-x") == "default"


def test_get_active_env_reads_shared_container_from_subagent_id():
    """``get_active_env`` must see the shared ``"default"`` sandbox when
    called with a subagent's task_id, so the agent loop's turn-budget
    enforcement reads the real env (not None) during delegation."""
    sentinel = object()
    terminal_tool._active_environments["default"] = sentinel
    try:
        assert terminal_tool.get_active_env("subagent-7-cafe") is sentinel
        assert terminal_tool.get_active_env(None) is sentinel
        assert terminal_tool.get_active_env("default") is sentinel
    finally:
        terminal_tool._active_environments.pop("default", None)


def test_get_active_env_honours_rl_override():
    rl_env = object()
    default_env = object()
    terminal_tool._active_environments["default"] = default_env
    terminal_tool._active_environments["rl-42"] = rl_env
    terminal_tool.register_task_env_overrides("rl-42", {"docker_image": "x"})
    try:
        # With an override registered, lookup returns the task's own env,
        # not the shared "default" one.
        assert terminal_tool.get_active_env("rl-42") is rl_env
    finally:
        terminal_tool.clear_task_env_overrides("rl-42")
        terminal_tool._active_environments.pop("default", None)
        terminal_tool._active_environments.pop("rl-42", None)


def test_cwd_only_override_collapses_to_default():
    """CWD-only overrides (ACP adapter workspace tracking) must NOT trigger
    container isolation — they should collapse to the shared 'default'
    container so all surfaces (TUI, gateway, dashboard) share one sandbox.
    Regression for #37361."""
    terminal_tool.register_task_env_overrides(
        "acp-session-abc", {"cwd": "/home/user/project"}
    )
    try:
        assert (
            terminal_tool._resolve_container_task_id("acp-session-abc")
            == "default"
        )
    finally:
        terminal_tool.clear_task_env_overrides("acp-session-abc")


def test_cwd_plus_isolate_env_keeps_own_id():
    """A CWD-only override with the explicit ``isolate_env`` opt-in must get
    its OWN environment (not collapse to 'default').  The cron scheduler uses
    this so concurrent per-workdir jobs don't share one live shell and clobber
    each other's env.cwd."""
    terminal_tool.register_task_env_overrides(
        "cron_job_0600", {"cwd": "/home/user/project-a", "isolate_env": True}
    )
    try:
        assert (
            terminal_tool._resolve_container_task_id("cron_job_0600")
            == "cron_job_0600"
        )
    finally:
        terminal_tool.clear_task_env_overrides("cron_job_0600")


def test_isolate_env_falsy_still_collapses_to_default():
    """A falsy ``isolate_env`` is not an isolation signal — a plain CWD-only
    override still collapses to the shared 'default' env."""
    terminal_tool.register_task_env_overrides(
        "sess-x", {"cwd": "/p", "isolate_env": False}
    )
    try:
        assert terminal_tool._resolve_container_task_id("sess-x") == "default"
    finally:
        terminal_tool.clear_task_env_overrides("sess-x")


def test_child_inherits_parent_isolated_environment_and_cwd():
    """A delegated child must resolve to its cron parent's isolated environment
    and inherit the parent's cwd instead of falling back to ``default``."""
    terminal_tool.register_task_env_overrides(
        "cron-parent", {"cwd": "/home/user/project-a", "isolate_env": True}
    )
    try:
        inherited = terminal_tool.inherit_task_env_overrides(
            "cron-parent", "cron-child"
        )
        assert inherited is True
        assert terminal_tool._resolve_container_task_id("cron-child") == "cron-child"
        child_overrides = terminal_tool.resolve_task_overrides("cron-child")
        assert child_overrides["cwd"] == "/home/user/project-a"
        assert child_overrides["isolate_env"] is True
        assert "inherit_env_from" not in child_overrides
    finally:
        terminal_tool.clear_task_env_overrides("cron-child")
        terminal_tool.clear_task_env_overrides("cron-parent")


def test_child_preserves_shared_default_for_cwd_only_parent():
    """Ordinary ACP/TUI cwd tracking must not become sandbox isolation."""
    terminal_tool.register_task_env_overrides(
        "acp-parent", {"cwd": "/workspace/project"}
    )
    try:
        assert terminal_tool.inherit_task_env_overrides(
            "acp-parent", "acp-child"
        )
        child_overrides = terminal_tool.resolve_task_overrides("acp-child")
        assert child_overrides["cwd"] == "/workspace/project"
        assert not child_overrides.get("isolate_env")
        assert terminal_tool._resolve_container_task_id("acp-child") == "default"
    finally:
        terminal_tool.clear_task_env_overrides("acp-child")
        terminal_tool.clear_task_env_overrides("acp-parent")


def test_concurrent_children_stay_bound_to_their_own_parent_workdirs():
    """Children of two concurrent cron parents must never share cwd/container."""
    terminal_tool.register_task_env_overrides(
        "parent-a", {"cwd": "/projects/a", "isolate_env": True}
    )
    terminal_tool.register_task_env_overrides(
        "parent-b", {"cwd": "/projects/b", "isolate_env": True}
    )
    try:
        assert terminal_tool.inherit_task_env_overrides("parent-a", "child-a")
        assert terminal_tool.inherit_task_env_overrides("parent-b", "child-b")
        assert terminal_tool._resolve_container_task_id("child-a") == "child-a"
        assert terminal_tool._resolve_container_task_id("child-b") == "child-b"
        assert terminal_tool.resolve_task_overrides("child-a")["cwd"] == "/projects/a"
        assert terminal_tool.resolve_task_overrides("child-b")["cwd"] == "/projects/b"
    finally:
        for task_id in ("child-a", "child-b", "parent-a", "parent-b"):
            terminal_tool.clear_task_env_overrides(task_id)


def test_child_does_not_register_when_parent_has_no_override():
    """Ordinary delegation without a parent override keeps legacy behavior."""
    terminal_tool.clear_task_env_overrides("plain-parent")
    terminal_tool.clear_task_env_overrides("plain-child")
    assert terminal_tool.inherit_task_env_overrides("plain-parent", "plain-child") is False
    assert terminal_tool.resolve_task_overrides("plain-child") == {}
    assert terminal_tool._resolve_container_task_id("plain-child") == "default"


def test_cwd_plus_docker_image_keeps_own_id():
    """When overrides include both cwd AND docker_image, isolation must
    still be honoured (RL/benchmark pattern with explicit cwd)."""
    terminal_tool.register_task_env_overrides(
        "rl-with-cwd", {"docker_image": "myimg:latest", "cwd": "/workspace"}
    )
    try:
        assert (
            terminal_tool._resolve_container_task_id("rl-with-cwd")
            == "rl-with-cwd"
        )
    finally:
        terminal_tool.clear_task_env_overrides("rl-with-cwd")


def test_env_type_override_keeps_own_id():
    """env_type is an isolation key — must trigger per-task container."""
    terminal_tool.register_task_env_overrides(
        "bench-env", {"env_type": "sandbox", "cwd": "/work"}
    )
    try:
        assert (
            terminal_tool._resolve_container_task_id("bench-env")
            == "bench-env"
        )
    finally:
        terminal_tool.clear_task_env_overrides("bench-env")
