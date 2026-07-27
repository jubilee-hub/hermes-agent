from __future__ import annotations

import json
import os
import socket
import socketserver
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from tools.environments.sandbox_runner import (
    SandboxRunnerEnvironment,
    sandbox_runner_ready_from_environment,
)


TASK_KEY = "sandbox-v1-" + ("a" * 43)
TOKEN = "runner-test-token-with-at-least-thirty-two-bytes"


class _ThreadedUnixHTTPServer(
    socketserver.ThreadingMixIn, socketserver.UnixStreamServer
):
    daemon_threads = True

    def __init__(self, socket_path: str):
        self.requests: list[dict[str, object]] = []
        self.response: dict[str, object] = {
            "schemaVersion": 1,
            "ok": True,
            "exitCode": 0,
            "stdout": "stdout",
            "stderr": "",
            "timedOut": False,
        }
        self.status = 200
        self.request_started = threading.Event()
        self.release_response = threading.Event()
        self.disconnect_observed = threading.Event()
        self.block_response = False
        super().__init__(socket_path, _RunnerHandler)


class _RunnerHandler(BaseHTTPRequestHandler):
    server: _ThreadedUnixHTTPServer

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length))
        self.server.requests.append({
            "path": self.path,
            "authorization": self.headers.get("authorization"),
            "contentType": self.headers.get("content-type"),
            "body": body,
        })
        self.server.request_started.set()
        if self.server.block_response:
            self.server.release_response.wait(timeout=5)
        payload = json.dumps(self.server.response).encode()
        try:
            self.send_response(self.server.status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            self.server.disconnect_observed.set()

    def do_GET(self):
        if self.path == "/health":
            payload = {
                "schemaVersion": 1,
                "status": "ready",
                "checks": {"apptainer": "passed", "auth": "passed"},
            }
            status_code = 200
        elif (
            self.path == "/v1/capabilities"
            and self.headers.get("authorization") == f"Bearer {TOKEN}"
        ):
            payload = {
                "schemaVersion": 1,
                "isolation": "per_task_overlay",
                "network": "disabled",
                "imageFingerprint": "sha256:" + ("a" * 64),
                "limits": {"maxTimeoutMs": 300_000},
            }
            status_code = 200
        else:
            payload = {"schemaVersion": 1, "error": {"code": "unauthorized"}}
            status_code = 401
        encoded = json.dumps(payload).encode()
        self.send_response(status_code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format, *_args):
        return


@pytest.fixture
def runner_fixture(tmp_path: Path):
    socket_path = tmp_path / "runner.sock"
    token_path = tmp_path / "runner.token"
    token_path.write_text(TOKEN, encoding="utf-8")
    token_path.chmod(0o600)
    token_fd = os.open(token_path, os.O_RDONLY)
    server = _ThreadedUnixHTTPServer(str(socket_path))
    os.chown(socket_path, -1, os.getegid())
    socket_path.chmod(0o660)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, socket_path, token_fd
    finally:
        server.release_response.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        os.close(token_fd)


def _environment(socket_path: Path, token_fd: int) -> SandboxRunnerEnvironment:
    return SandboxRunnerEnvironment(
        task_key=TASK_KEY,
        socket_path=str(socket_path),
        token_fd=token_fd,
        token_owner_must_differ=False,
        initialize_session=False,
        timeout=3,
    )


def test_exec_uses_authenticated_uds_and_maps_bounded_result(runner_fixture):
    server, socket_path, token_fd = runner_fixture
    server.response = {
        "schemaVersion": 1,
        "ok": False,
        "exitCode": 7,
        "stdout": "out",
        "stderr": "err",
        "timedOut": False,
    }
    environment = _environment(socket_path, token_fd)

    result = environment.execute(
        "printf user-command",
        cwd="/workspace/project",
        timeout=2,
        stdin_data="input",
    )

    assert result == {"output": "out\nerr", "returncode": 7}
    assert len(server.requests) == 1
    request = server.requests[0]
    assert request["path"] == "/v1/exec"
    assert request["authorization"] == f"Bearer {TOKEN}"
    assert request["contentType"] == "application/json"
    body = request["body"]
    assert isinstance(body, dict)
    assert body["schemaVersion"] == 1
    assert body["taskKey"] == TASK_KEY
    assert body["stdin"] == "input"
    assert body["timeoutMs"] == 2000
    assert "printf user-command" in str(body["command"])
    assert "builtin cd -- /workspace/project" in str(body["command"])


def test_timeout_result_maps_to_shell_timeout_without_fallback(runner_fixture):
    server, socket_path, token_fd = runner_fixture
    server.response = {
        "schemaVersion": 1,
        "ok": False,
        "exitCode": None,
        "stdout": "",
        "stderr": "",
        "timedOut": True,
    }
    environment = _environment(socket_path, token_fd)

    result = environment.execute("sleep forever", timeout=2)

    assert result["returncode"] == 124


@pytest.mark.parametrize(
    "response",
    [
        {
            "schemaVersion": 2,
            "ok": True,
            "exitCode": 0,
            "stdout": "",
            "stderr": "",
            "timedOut": False,
        },
        {
            "schemaVersion": 1,
            "ok": True,
            "exitCode": 0,
            "stdout": "",
            "stderr": "",
            "timedOut": True,
        },
        {
            "schemaVersion": 1,
            "ok": True,
            "exitCode": True,
            "stdout": "",
            "stderr": "",
            "timedOut": False,
        },
        {
            "schemaVersion": 1,
            "ok": True,
            "exitCode": 0,
            "stdout": "",
            "stderr": "",
            "timedOut": False,
            "extra": "field",
        },
    ],
)
def test_malformed_runner_response_fails_closed_without_leaking_secrets(
    runner_fixture, response, caplog
):
    server, socket_path, token_fd = runner_fixture
    server.response = response
    environment = _environment(socket_path, token_fd)

    result = environment.execute("true")

    assert result == {"output": "", "returncode": 1}
    combined_logs = caplog.text
    assert TOKEN not in combined_logs
    assert TASK_KEY not in combined_logs


def test_bad_auth_or_unavailable_socket_never_falls_back_to_local(tmp_path: Path):
    token_path = tmp_path / "runner.token"
    token_path.write_text(TOKEN, encoding="utf-8")
    token_path.chmod(0o600)
    token_fd = os.open(token_path, os.O_RDONLY)
    socket_path = tmp_path / "missing.sock"
    try:
        with pytest.raises(
            RuntimeError, match="Sandbox runner transport is unavailable"
        ):
            SandboxRunnerEnvironment(
                task_key=TASK_KEY,
                socket_path=str(socket_path),
                token_fd=token_fd,
                token_owner_must_differ=False,
                initialize_session=False,
            )
    finally:
        os.close(token_fd)


def test_kill_closes_the_uds_request_so_runner_observes_disconnect(runner_fixture):
    server, socket_path, token_fd = runner_fixture
    server.block_response = True
    environment = _environment(socket_path, token_fd)
    process = environment._run_bash("sleep forever", timeout=3)
    assert server.request_started.wait(timeout=2)

    process.kill()
    assert process.wait(timeout=2) == 1
    server.release_response.set()
    assert server.disconnect_observed.wait(timeout=2)


def test_cleanup_does_not_delete_the_durable_remote_overlay(runner_fixture):
    server, socket_path, token_fd = runner_fixture
    environment = _environment(socket_path, token_fd)

    environment.cleanup()

    assert server.requests == []


def test_existing_environment_reconnects_after_runner_restart(tmp_path: Path):
    socket_path = tmp_path / "runner.sock"
    token_path = tmp_path / "runner.token"
    token_path.write_text(TOKEN, encoding="utf-8")
    token_path.chmod(0o600)
    token_fd = os.open(token_path, os.O_RDONLY)

    def start_server():
        server = _ThreadedUnixHTTPServer(str(socket_path))
        os.chown(socket_path, -1, os.getegid())
        socket_path.chmod(0o660)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    server, thread = start_server()
    try:
        environment = _environment(socket_path, token_fd)
        assert environment.execute("first")["returncode"] == 0
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        socket_path.unlink()

        server, thread = start_server()
        assert environment.execute("second")["returncode"] == 0
        assert server.requests[0]["body"]["taskKey"] == TASK_KEY
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        os.close(token_fd)


def test_rejects_oversized_command_and_stdin_before_transport(runner_fixture):
    server, socket_path, token_fd = runner_fixture
    environment = _environment(socket_path, token_fd)

    assert environment.execute("x" * 65_537) == {"output": "", "returncode": 1}
    assert environment.execute("true", stdin_data="x" * 1_048_577) == {
        "output": "",
        "returncode": 1,
    }
    assert server.requests == []


def test_constructor_rejects_current_user_owned_token_by_default(runner_fixture):
    _server, socket_path, token_fd = runner_fixture

    with pytest.raises(RuntimeError, match="Sandbox runner credential is unavailable"):
        SandboxRunnerEnvironment(
            task_key=TASK_KEY,
            socket_path=str(socket_path),
            token_fd=token_fd,
            initialize_session=False,
        )


def test_socket_metadata_drift_is_rejected(runner_fixture):
    _server, socket_path, token_fd = runner_fixture
    socket_path.chmod(0o666)

    with pytest.raises(RuntimeError, match="Sandbox runner transport is unavailable"):
        _environment(socket_path, token_fd)


def test_live_readiness_requires_uds_policy_and_bearer_auth(
    runner_fixture, monkeypatch
):
    _server, socket_path, token_fd = runner_fixture
    monkeypatch.setenv("HERMES_SANDBOX_RUNNER_SOCKET_PATH", str(socket_path))
    monkeypatch.setenv("HERMES_SANDBOX_RUNNER_TOKEN_FD", str(token_fd))

    # Unit tests own their fixture token. The production contract additionally
    # requires the inherited source owner to differ from uid 10000, which is
    # covered by the constructor test and T1 readback.
    with pytest.MonkeyPatch.context() as probe_patch:
        original = SandboxRunnerEnvironment._validate_token_fd
        probe_patch.setattr(
            SandboxRunnerEnvironment,
            "_validate_token_fd",
            staticmethod(
                lambda fd, *, owner_must_differ: original(
                    fd,
                    owner_must_differ=False,
                )
            ),
        )
        assert sandbox_runner_ready_from_environment() is True

    os.fchmod(token_fd, 0o644)
    assert sandbox_runner_ready_from_environment() is False
