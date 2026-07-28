import json
from unittest.mock import patch

import pytest

import tools.terminal_tool as terminal
import tools.code_execution_tool as code_execution


TASK_ID = "sandbox-task-" + ("b" * 64)
TASK_KEY = "sandbox-v1-" + ("b" * 43)


def test_factory_requires_the_process_local_task_override(monkeypatch):
    monkeypatch.setenv("HERMES_SANDBOX_TASK_KEY", TASK_KEY)

    with pytest.raises(RuntimeError, match="task identity is unavailable"):
        terminal._create_environment(
            env_type="sandbox_runner",
            image="",
            cwd="/host/path",
            timeout=30,
            task_id=TASK_ID,
        )


def test_factory_passes_only_the_trusted_override_and_forces_workspace(monkeypatch):
    captured = {}

    class FakeEnvironment:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    terminal.register_task_env_overrides(
        TASK_ID,
        {"env_type": "sandbox_runner", "sandbox_task_key": TASK_KEY},
    )
    monkeypatch.setenv("HERMES_SANDBOX_RUNNER_TOKEN_FD", "8")
    monkeypatch.setenv(
        "HERMES_SANDBOX_RUNNER_SOCKET_PATH",
        "/run/agent-saas-sandbox-runner/runner.sock",
    )
    try:
        with patch(
            "tools.environments.sandbox_runner.SandboxRunnerEnvironment",
            FakeEnvironment,
        ):
            environment = terminal._create_environment(
                env_type="sandbox_runner",
                image="ignored",
                cwd="/host/path",
                timeout=30,
                task_id=TASK_ID,
            )
    finally:
        terminal.clear_task_env_overrides(TASK_ID)

    assert isinstance(environment, FakeEnvironment)
    assert captured == {
        "task_key": TASK_KEY,
        "socket_path": "/run/agent-saas-sandbox-runner/runner.sock",
        "token_fd": 8,
        "cwd": "/workspace",
        "timeout": 30,
    }


def test_sandbox_runner_is_a_container_backend_with_workspace_default(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "sandbox_runner")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    config = terminal._get_env_config()

    assert config["env_type"] == "sandbox_runner"
    assert config["cwd"] == "/workspace"


def test_terminal_rejects_background_pty_and_oversized_timeout_before_env_creation():
    terminal.register_task_env_overrides(
        TASK_ID,
        {"env_type": "sandbox_runner", "sandbox_task_key": TASK_KEY},
    )
    try:
        with patch.object(terminal, "_create_environment") as create:
            background = json.loads(
                terminal.terminal_tool("sleep 10", background=True, task_id=TASK_ID)
            )
            pty = json.loads(terminal.terminal_tool("bash", pty=True, task_id=TASK_ID))
            timeout = json.loads(
                terminal.terminal_tool("sleep 301", timeout=301, task_id=TASK_ID)
            )
    finally:
        terminal.clear_task_env_overrides(TASK_ID)

    assert background["status"] == "blocked"
    assert pty["status"] == "blocked"
    assert timeout["status"] == "blocked"
    create.assert_not_called()


def test_execute_code_dispatches_the_request_override_to_remote_runner(monkeypatch):
    terminal.register_task_env_overrides(
        TASK_ID,
        {"env_type": "sandbox_runner", "sandbox_task_key": TASK_KEY},
    )
    monkeypatch.setattr(code_execution, "SANDBOX_AVAILABLE", True)
    try:
        with (
            patch(
                "tools.approval.check_execute_code_guard",
                return_value={"approved": True},
            ),
            patch.object(
                code_execution,
                "_execute_sandbox_runner_code",
                return_value='{"status":"remote"}',
            ) as execute_remote,
        ):
            result = code_execution.execute_code("print('safe')", task_id=TASK_ID)
    finally:
        terminal.clear_task_env_overrides(TASK_ID)

    assert json.loads(result) == {"status": "remote"}
    execute_remote.assert_called_once_with("print('safe')", TASK_ID, None)


def test_execute_code_uses_one_runner_request_without_same_key_rpc_polling(monkeypatch):
    calls = []

    class FakeEnvironment:
        def execute(self, command, **kwargs):
            calls.append((command, kwargs))
            return {"output": "safe\n", "returncode": 0}

    monkeypatch.setattr(
        code_execution,
        "_get_or_create_env",
        lambda _task_id: (FakeEnvironment(), "sandbox_runner"),
    )
    monkeypatch.setattr(code_execution, "_load_config", lambda: {"timeout": 30})

    result = json.loads(
        code_execution._execute_sandbox_runner_code(
            "print('safe')",
            TASK_ID,
            ["terminal"],
        )
    )

    assert result["status"] == "success"
    assert result["output"] == "safe\n"
    assert result["tool_calls_made"] == 0
    assert calls == [
        (
            "PYTHONDONTWRITEBYTECODE=1 python3 -",
            {
                "cwd": "/workspace",
                "timeout": 30,
                "stdin_data": "print('safe')",
            },
        )
    ]
