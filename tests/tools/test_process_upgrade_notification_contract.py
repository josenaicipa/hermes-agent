"""New retained-result ownership and old reliable wake activation coexist."""
import platform
from unittest.mock import MagicMock, patch

import pytest

from gateway.session_context import scoped_current_session_id
from tools.process_registry import ProcessNotificationConfig, ProcessRegistry


@pytest.mark.parametrize('notification', [None, ProcessNotificationConfig(notify_on_complete=True)])
def test_empty_notification_route_keeps_session_owner(notification):
    with scoped_current_session_id('durable-cli-owner'):
        session = ProcessRegistry._new_session('echo done', 'task', 'task', '', None, notification)
    assert session.parent_session_id == 'durable-cli-owner'
    assert session.owner_task_id == 'task'


@pytest.mark.skipif(platform.system() == 'Windows', reason='POSIX PTY implementation')
def test_started_pty_is_reaped_not_reexecuted_when_activation_fails(tmp_path):
    ptyprocess = pytest.importorskip('ptyprocess')
    registry = ProcessRegistry()
    child = MagicMock(pid=4321)
    with patch.object(ptyprocess.PtyProcess, 'spawn', return_value=child), \
         patch.object(registry, '_scope_argv', return_value=['/bin/sh', '-c', 'true']), \
         patch.object(registry, '_safe_host_start_time', return_value=None), \
         patch.object(registry, '_track_started', side_effect=RuntimeError('checkpoint unavailable')), \
         patch('tools.process_registry.subprocess.Popen') as fallback:
        with pytest.raises(RuntimeError, match='checkpoint unavailable'):
            registry.spawn_local('true', cwd=str(tmp_path), use_pty=True)
    child.terminate.assert_called_once_with(force=True)
    child.close.assert_called_once()
    fallback.assert_not_called()
