"""Regression tests for host-path cwd sanitization on container backends.

Two code paths in ``tools/terminal_tool.py`` must reject a host (or relative)
working directory before it reaches ``docker run -w``:

  1. ``_get_env_config()`` sanitizes the ``TERMINAL_CWD``-derived ``config["cwd"]``.
  2. ``terminal_tool()`` resolves a *per-task cwd override* that WINS over
     ``config["cwd"]`` (registered by the gateway/TUI for workspace tracking,
     and by RL/benchmark envs). That override was applied RAW — never sanitized
     — so a host cwd (e.g. a Windows desktop session's ``C:\\Users\\<user>``)
     leaked straight to ``docker run -w C:\\Users\\<user>``, which fails to start
     the container (exit 125). The sanitizer at path #1 lists ``C:\\``/``C:/`` as
     host prefixes but only ever ran against ``config["cwd"]``, so the override
     bypassed the one guard that would have caught it.

Both paths now share ``_is_unusable_container_cwd()``; these tests pin its
behaviour so neither path can regress.
"""

import pytest
import tools.terminal_tool as tt


class TestIsUnusableContainerCwd:
    def test_windows_backslash_host_path_rejected(self):
        # The exact shape from the bug report: a Windows host cwd reaching a
        # Linux container's -w flag.
        assert tt._is_unusable_container_cwd(r"C:\Users\someuser") is True

    def test_windows_forwardslash_host_path_rejected(self):
        assert tt._is_unusable_container_cwd("C:/Users/someuser") is True

    def test_posix_home_host_path_rejected(self):
        assert tt._is_unusable_container_cwd("/home/ben/projects") is True

    def test_macos_users_host_path_rejected(self):
        assert tt._is_unusable_container_cwd("/Users/ben/projects") is True

    def test_relative_path_rejected(self):
        assert tt._is_unusable_container_cwd(".") is True
        assert tt._is_unusable_container_cwd("src/app") is True

    def test_valid_container_workspace_accepted(self):
        # In-container paths that RL/benchmark overrides legitimately set must
        # pass through untouched.
        assert tt._is_unusable_container_cwd("/workspace") is False
        assert tt._is_unusable_container_cwd("/root") is False
        assert tt._is_unusable_container_cwd("/app") is False
        assert tt._is_unusable_container_cwd("/opt/project") is False

    def test_empty_is_not_flagged(self):
        # Empty/None-ish cwd is handled by the caller's `or config["cwd"]`
        # fallback, not by flagging it here.
        assert tt._is_unusable_container_cwd("") is False

    def test_host_prefixes_include_windows_and_posix(self):
        # Guard the constant itself — the Windows entries are the ones that
        # were load-bearing for the reported desktop bug.
        assert r"C:\\"[:2] in tt._HOST_CWD_PREFIXES or "C:\\" in tt._HOST_CWD_PREFIXES
        assert "C:/" in tt._HOST_CWD_PREFIXES
        assert "/home/" in tt._HOST_CWD_PREFIXES
        assert "/Users/" in tt._HOST_CWD_PREFIXES

    def test_container_backends_set(self):
        assert tt._CONTAINER_BACKENDS == frozenset(
            {"docker", "singularity", "modal", "daytona"}
        )


class TestOverrideCwdSanitizedAtCallSite:
    """E2E pin: a per-task cwd OVERRIDE that is a host path must NOT reach the
    container builder. This is the actual reported bug — the gateway/TUI
    registers the host launch dir as a cwd override, which previously won over
    the (sanitized) config["cwd"] and flowed raw into `docker run -w`.
    """

    def _run_and_capture_cwd(self, monkeypatch, override_cwd, config_cwd="/root"):
        """Drive terminal_tool() on the docker backend with a host-path cwd
        override registered, and return the cwd that reached _create_environment
        (i.e. the cwd that would be passed to `docker run -w`).
        """
        captured = {}

        config = {
            "env_type": "docker",
            "docker_image": "pytorch/pytorch:latest",
            "cwd": config_cwd,
            "host_cwd": None,
            "timeout": 180,
            "lifetime_seconds": 300,
            "container_cpu": 1,
            "container_memory": 5120,
            "container_disk": 51200,
            "container_persistent": True,
            "docker_volumes": [],
            "docker_env": {},
            "docker_extra_args": [],
            "docker_mount_cwd_to_workspace": False,
            "docker_run_as_host_user": False,
            "docker_forward_env": [],
            "modal_mode": "auto",
        }

        class _DummyEnv:
            cwd = config_cwd

            def execute(self, *a, **k):
                return {"output": "", "exit_code": 0}

        def fake_create_environment(env_type, image, cwd, timeout, **kwargs):
            captured["cwd"] = cwd
            return _DummyEnv()

        monkeypatch.setattr(tt, "_get_env_config", lambda: config)
        monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(tt, "_check_all_guards", lambda *a, **k: {"approved": True})
        monkeypatch.setattr(tt, "_create_environment", fake_create_environment)
        # Force a fresh environment build so _create_environment is invoked.
        monkeypatch.setattr(tt, "_active_environments", {})
        monkeypatch.setattr(tt, "_last_activity", {})

        task_id = "sess-host-cwd"
        tt.register_task_env_overrides(task_id, {"cwd": override_cwd})
        try:
            tt.terminal_tool(command="pwd", task_id=task_id)
        finally:
            tt.clear_task_env_overrides(task_id)
            tt._active_environments.pop(task_id, None)
            tt._active_environments.pop("default", None)
        return captured.get("cwd")

    def test_windows_host_override_does_not_reach_container(self, monkeypatch):
        # The bug: C:\Users\<user> registered as override → docker run -w C:\Users\<user> → exit 125.
        cwd = self._run_and_capture_cwd(monkeypatch, r"C:\Users\someuser")
        assert cwd == "/root", (
            f"Host-path cwd override leaked to the container builder: {cwd!r}. "
            "It must be sanitized back to config['cwd']."
        )

    def test_posix_host_override_does_not_reach_container(self, monkeypatch):
        cwd = self._run_and_capture_cwd(monkeypatch, "/home/someuser/project")
        assert cwd == "/root"

    def test_valid_container_override_is_preserved(self, monkeypatch):
        # RL/benchmark envs set an in-container path; it must pass through.
        cwd = self._run_and_capture_cwd(monkeypatch, "/workspace/task42")
        assert cwd == "/workspace/task42"


class TestFileOpsCwdSanitizedAtCallSite:
    """E2E pin: file tools (_get_file_ops) must sanitize a host/relative cwd
    override before it reaches _create_environment on a container backend —
    the same guard the terminal tool got in #50636.  Without it, a Desktop/TUI
    host cwd (e.g. ``/Users/me/workspace``) leaks straight into
    ``docker run -w`` and ``search_files`` returns an empty workspace (#54447).
    """

    def _run_and_capture_cwd(self, monkeypatch, override_cwd, env_type="docker",
                             config_cwd="/workspace"):
        """Drive ``_get_file_ops()`` on a container backend with a host-path cwd
        override registered, and return the cwd that reached
        ``_create_environment`` (i.e. the cwd passed to ``docker run -w``).
        """
        import tools.terminal_tool as tt
        import tools.file_tools as ft

        captured = {}

        config = {
            "env_type": env_type,
            "docker_image": "pytorch/pytorch:latest",
            "singularity_image": "docker://pytorch/pytorch:latest",
            "modal_image": "pytorch/pytorch:latest",
            "daytona_image": "pytorch/pytorch:latest",
            "cwd": config_cwd,
            "host_cwd": None,
            "timeout": 180,
            "lifetime_seconds": 300,
            "container_cpu": 1,
            "container_memory": 5120,
            "container_disk": 51200,
            "container_persistent": True,
            "docker_volumes": [],
            "docker_env": {},
            "docker_extra_args": [],
            "docker_mount_cwd_to_workspace": False,
            "docker_run_as_host_user": False,
            "docker_forward_env": [],
            "modal_mode": "auto",
            "ssh_host": "",
            "ssh_user": "",
            "ssh_port": 22,
            "ssh_key": "",
            "ssh_persistent": False,
            "local_persistent": False,
        }

        class _DummyEnv:
            cwd = config_cwd

            def execute(self, *a, **k):
                return {"output": "", "exit_code": 0}

        def fake_create_environment(env_type, image, cwd, timeout, **kwargs):
            captured["cwd"] = cwd
            return _DummyEnv()

        monkeypatch.setattr(tt, "_get_env_config", lambda: config)
        monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(tt, "_create_environment", fake_create_environment)
        # Force a fresh environment build.
        monkeypatch.setattr(tt, "_active_environments", {})
        monkeypatch.setattr(tt, "_last_activity", {})
        monkeypatch.setattr(ft, "_file_ops_cache", {})
        monkeypatch.setattr(tt, "_session_cwd", {})

        task_id = "sess-fileops-host-cwd"
        tt.register_task_env_overrides(task_id, {"cwd": override_cwd})
        try:
            ft._get_file_ops(task_id)
        finally:
            tt.clear_task_env_overrides(task_id)
        return captured.get("cwd")

    def test_macos_host_override_does_not_reach_container(self, monkeypatch):
        # Desktop/TUI registers /Users/<me>/workspace as the session cwd.
        cwd = self._run_and_capture_cwd(monkeypatch, "/Users/me/workspace")
        assert cwd == "/workspace", (
            f"Host-path cwd override leaked to the container builder: {cwd!r}. "
            "It must be sanitized back to config['cwd']."
        )

    def test_posix_home_host_override_does_not_reach_container(self, monkeypatch):
        cwd = self._run_and_capture_cwd(monkeypatch, "/home/someuser/project")
        assert cwd == "/workspace"

    def test_windows_host_override_does_not_reach_container(self, monkeypatch):
        cwd = self._run_and_capture_cwd(monkeypatch, r"C:\Users\someuser")
        assert cwd == "/workspace"

    def test_relative_cwd_override_does_not_reach_container(self, monkeypatch):
        cwd = self._run_and_capture_cwd(monkeypatch, "src/app")
        assert cwd == "/workspace"

    def test_valid_container_override_is_preserved(self, monkeypatch):
        # RL/benchmark envs set an in-container path; it must pass through.
        cwd = self._run_and_capture_cwd(monkeypatch, "/workspace/task42")
        assert cwd == "/workspace/task42"

    def test_host_override_sanitized_on_singularity(self, monkeypatch):
        cwd = self._run_and_capture_cwd(
            monkeypatch, "/Users/me/workspace", env_type="singularity")
        assert cwd == "/workspace"

    def test_host_override_sanitized_on_modal(self, monkeypatch):
        cwd = self._run_and_capture_cwd(
            monkeypatch, "/Users/me/workspace", env_type="modal")
        assert cwd == "/workspace"


@pytest.mark.parametrize("surface", ["terminal", "file", "execute_code"])
def test_docker_auto_mount_uses_each_tasks_own_host_cwd(
    monkeypatch, tmp_path, surface
):
    """Whichever tool creates Docker first must mount this task's workdir."""
    import tools.code_execution_tool as cet
    import tools.file_tools as ft

    task_cwd = tmp_path / surface
    task_cwd.mkdir()
    decoy = tmp_path / "global-decoy"
    decoy.mkdir(exist_ok=True)
    task_id = f"cron-docker-{surface}"
    captured = {}
    config = {
        "env_type": "docker",
        "docker_image": "example:latest",
        "cwd": "/workspace",
        "host_cwd": str(decoy),
        "docker_mount_cwd_to_workspace": True,
        "timeout": 60,
        "lifetime_seconds": 300,
        "container_cpu": 1,
        "container_memory": 5120,
        "container_disk": 51200,
        "container_persistent": True,
        "docker_volumes": [],
        "docker_env": {},
        "docker_extra_args": [],
        "docker_forward_env": [],
        "docker_run_as_host_user": False,
        "docker_network": True,
        "modal_mode": "auto",
        "local_persistent": False,
    }

    class DummyEnv:
        cwd = "/workspace"
        env = {}

        def execute(self, *_args, **_kwargs):
            return {"output": "", "returncode": 0, "exit_code": 0}

    def fake_create_environment(*, cwd, host_cwd=None, **_kwargs):
        captured.update(cwd=cwd, host_cwd=host_cwd)
        return DummyEnv()

    monkeypatch.setattr(tt, "_get_env_config", lambda: config)
    monkeypatch.setattr(tt, "_create_environment", fake_create_environment)
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(tt, "_check_all_guards", lambda *a, **k: {"approved": True})
    monkeypatch.setattr(tt, "_active_environments", {})
    monkeypatch.setattr(tt, "_last_activity", {})
    monkeypatch.setattr(tt, "_creation_locks", {})
    monkeypatch.setattr(ft, "_file_ops_cache", {})
    # Upstream deleted legacy env-side ``_last_known_cwd`` tracking
    # (session cwd record + task overrides replace it). Do not reintroduce
    # that module attribute just for the test fixture.

    tt.register_task_env_overrides(
        task_id, {"cwd": str(task_cwd), "isolate_env": True}
    )
    try:
        if surface == "terminal":
            tt.terminal_tool(command="pwd", task_id=task_id)
        elif surface == "file":
            ft._get_file_ops(task_id)
        else:
            cet._get_or_create_env(task_id)
        assert captured == {"cwd": "/workspace", "host_cwd": str(task_cwd)}
    finally:
        tt.clear_task_env_overrides(task_id)


def test_explicit_docker_host_cwd_under_root_maps_to_workspace(monkeypatch):
    task_id = "cron-docker-root-host"
    config = {
        "env_type": "docker",
        "cwd": "/workspace",
        "host_cwd": None,
        "docker_mount_cwd_to_workspace": True,
    }
    real_isdir = tt.os.path.isdir
    monkeypatch.setattr(
        tt.os.path,
        "isdir",
        lambda path: True if path == "/root/project" else real_isdir(path),
    )
    tt.register_task_env_overrides(
        task_id,
        {"cwd": "/root/project", "host_cwd": "/root/project", "isolate_env": True},
    )
    try:
        assert tt.resolve_task_environment_paths(task_id, config) == (
            "/workspace",
            "/root/project",
        )
    finally:
        tt.clear_task_env_overrides(task_id)


def test_docker_auto_mount_file_operations_use_workspace_paths(monkeypatch, tmp_path):
    """Relative read/write/patch targets stay inside the mounted checkout."""
    import json
    import posixpath

    import tools.file_tools as ft

    task_cwd = tmp_path / "cron-project"
    task_cwd.mkdir()
    decoy = tmp_path / "global-decoy"
    decoy.mkdir()
    task_id = "cron-docker-file-functional"
    config = {
        "env_type": "docker",
        "docker_image": "example:latest",
        "cwd": "/workspace",
        "host_cwd": str(decoy),
        "docker_mount_cwd_to_workspace": True,
    }
    calls = {}

    class Result:
        content = "hello"
        error = None

        def to_dict(self):
            return {
                "content": self.content,
                "file_size": 5,
                "total_lines": 1,
                "truncated": False,
            }

    class FakeOps:
        cwd = "/workspace"
        env = None

        def read_file(self, path, _offset, _limit):
            calls["read"] = posixpath.join(self.cwd, path)
            return Result()

        def write_file(self, path, _content):
            calls["write"] = path
            return Result()

        def patch_replace(self, path, _old, _new, _replace_all):
            calls["patch"] = path
            return Result()

    monkeypatch.setattr(tt, "_get_env_config", lambda: config)
    monkeypatch.setattr(ft, "_get_file_ops", lambda _task_id: FakeOps())
    monkeypatch.setattr(ft, "_mark_verification_stale", lambda *a, **k: None)
    tt.register_task_env_overrides(
        task_id,
        {"cwd": str(task_cwd), "host_cwd": str(task_cwd), "isolate_env": True},
    )
    try:
        assert str(ft._resolve_path_for_task("src/a.py", task_id)) == "/workspace/src/a.py"
        assert not json.loads(ft.read_file_tool("src/a.py", task_id=task_id)).get("error")
        assert not json.loads(
            ft.write_file_tool("src/a.py", "hello", task_id=task_id)
        ).get("error")
        assert not json.loads(
            ft.patch_tool(
                mode="replace",
                path="src/a.py",
                old_string="hello",
                new_string="bye",
                task_id=task_id,
            )
        ).get("error")
        assert calls == {
            "read": "/workspace/src/a.py",
            "write": "/workspace/src/a.py",
            "patch": "/workspace/src/a.py",
        }
    finally:
        tt.clear_task_env_overrides(task_id)
