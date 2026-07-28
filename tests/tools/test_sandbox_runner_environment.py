from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

import tools.environments.sandbox_runner as sandbox_runner
from tools.environments.sandbox_runner import (
    SANDBOX_RUNNER_CANARY_CHECKS,
    SandboxRunnerEnvironment,
    read_sandbox_runner_artifact,
    run_sandbox_runner_isolation_canary,
    sandbox_runner_ready_from_environment,
)
from tools.terminal_tool import scoped_task_env_overrides


TASK_KEY = "sandbox-v1-" + ("a" * 43)
TASK_KEY_B = "sandbox-v1-" + ("b" * 43)
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
        self.response_content_type = "application/json"
        self.request_started = threading.Event()
        self.release_response = threading.Event()
        self.disconnect_observed = threading.Event()
        self.block_response = False
        self.health_delay_seconds = 0.0
        self.capabilities: dict[str, object] = {
            "schemaVersion": 1,
            "isolation": "per_task_overlay",
            "network": "disabled",
            "imageFingerprint": "sha256:" + ("a" * 64),
            "artifactExport": {
                "outbox": "/workspace/artifacts",
                "pathPolicy": "plain_filename_no_follow",
                "maxBytes": 16 * 1_048_576,
            },
            "limits": {"maxTimeoutMs": 300_000},
        }
        self.cleanup_response: dict[str, object] = {
            "schemaVersion": 1,
            "ok": True,
            "removed": True,
        }
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
        response = (
            self.server.cleanup_response
            if self.path == "/v1/cleanup"
            else self.server.response
        )
        payload = json.dumps(response).encode()
        try:
            self.send_response(self.server.status)
            self.send_header("content-type", self.server.response_content_type)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            self.server.disconnect_observed.set()

    def do_GET(self):
        if self.path == "/health":
            time.sleep(self.server.health_delay_seconds)
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
            payload = self.server.capabilities
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


def _task_ref(task_key: str) -> str:
    digest = hashlib.sha256(
        b"agent-saas-sandbox-runner-v1\0" + task_key.encode("utf-8")
    ).hexdigest()
    return f"sbx-{digest}"


def _artifact_response(
    task_key: str,
    *,
    filename: str = "report.bin",
    content: bytes = b"artifact-bytes",
) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "ok": True,
        "taskRef": _task_ref(task_key),
        "filename": filename,
        "sizeBytes": len(content),
        "checksumSha256": hashlib.sha256(content).hexdigest(),
        "contentBase64": base64.b64encode(content).decode("ascii"),
    }


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


def test_isolation_canary_uses_isolated_probe_unique_nonce_and_removes_mismatch(
    monkeypatch,
):
    observed: list[tuple[str, str]] = []
    deleted: list[str] = []
    nonces = iter(("1" * 32, "2" * 32))

    class FakeEnvironment:
        @staticmethod
        def _validate_task_key(task_key):
            return SandboxRunnerEnvironment._validate_task_key(task_key)

        def __init__(self, *, task_key, **_kwargs):
            self.task_key = task_key

        def execute(self, command, **_kwargs):
            observed.append((self.task_key, command))
            if "/usr/local/bin/python3 -I -S -P - <<'PY'" in command:
                return {
                    "returncode": 0,
                    "output": "HERMES_SANDBOX_CANARY:"
                    + json.dumps(
                        {
                            "workspaceBindingPresent": True,
                            "workspaceWriteRead": True,
                            "secretEnvDenied": True,
                            "egressDenied": True,
                        }
                    ),
                }
            return {"returncode": 0, "output": ""}

        def delete_remote_overlay(self):
            deleted.append(self.task_key)
            return True

        def cleanup(self):
            return None

    monkeypatch.setattr(sandbox_runner, "SandboxRunnerEnvironment", FakeEnvironment)
    monkeypatch.setattr(sandbox_runner.secrets, "token_hex", lambda _size: next(nonces))

    first = run_sandbox_runner_isolation_canary(TASK_KEY)
    second = run_sandbox_runner_isolation_canary(TASK_KEY)

    assert first == {name: True for name in SANDBOX_RUNNER_CANARY_CHECKS}
    assert second == {name: True for name in SANDBOX_RUNNER_CANARY_CHECKS}
    assert len(observed) == 8
    assert observed[0][0] == TASK_KEY
    assert observed[1][0] == TASK_KEY
    assert observed[2][0] != TASK_KEY
    assert observed[2][0].startswith("sandbox-v1-")
    assert observed[3][0] == TASK_KEY
    assert observed[2][0] != observed[6][0]
    assert deleted == [observed[2][0], observed[6][0]]
    assert "cd / && /usr/local/bin/python3 -I -S -P -" in observed[0][1]
    assert 'open("/proc/net/dev"' in observed[0][1]
    assert 'open("/proc/net/route"' in observed[0][1]
    assert 'open("/proc/net/ipv6_route"' in observed[0][1]
    assert "/sys/class/net" not in observed[0][1]
    assert "socket.create_connection" not in observed[0][1]
    assert "test -f /workspace/.agent-saas-canary-" in observed[1][1]
    assert "test ! -e /workspace/.agent-saas-canary-" in observed[2][1]
    assert observed[0][1] != observed[4][1]


def test_isolated_python_flags_ignore_task_controlled_standard_library_shadow(
    tmp_path,
):
    (tmp_path / "json.py").write_text(
        "raise RuntimeError('task-controlled import executed')\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-P",
            "-c",
            "import json; print(json.dumps({'isolated': True}, sort_keys=True))",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == '{"isolated": true}'
    assert "task-controlled import executed" not in result.stderr


def test_isolation_canary_fails_closed_without_runner_transport(monkeypatch):
    def unavailable_environment(*_args, **_kwargs):
        raise RuntimeError("private transport detail")

    monkeypatch.setattr(sandbox_runner, "SandboxRunnerEnvironment", unavailable_environment)

    checks = run_sandbox_runner_isolation_canary(TASK_KEY)

    assert checks == {name: False for name in SANDBOX_RUNNER_CANARY_CHECKS}
    assert "private transport detail" not in json.dumps(checks)


def test_isolation_canary_attempts_mismatch_cleanup_after_ambiguous_exec_failure(
    monkeypatch,
):
    deleted: list[str] = []

    class AmbiguousEnvironment:
        @staticmethod
        def _validate_task_key(task_key):
            return SandboxRunnerEnvironment._validate_task_key(task_key)

        def __init__(self, *, task_key, **_kwargs):
            self.task_key = task_key

        def execute(self, command, **_kwargs):
            if self.task_key != TASK_KEY:
                raise TimeoutError("response lost after possible overlay creation")
            if "/usr/local/bin/python3 -I -S -P -" in command:
                return {
                    "returncode": 0,
                    "output": "HERMES_SANDBOX_CANARY:"
                    + json.dumps(
                        {
                            "workspaceBindingPresent": True,
                            "workspaceWriteRead": True,
                            "secretEnvDenied": True,
                            "egressDenied": True,
                        }
                    ),
                }
            return {"returncode": 0, "output": ""}

        def delete_remote_overlay(self):
            deleted.append(self.task_key)
            return True

        def cleanup(self):
            return None

    monkeypatch.setattr(
        sandbox_runner,
        "SandboxRunnerEnvironment",
        AmbiguousEnvironment,
    )

    checks = run_sandbox_runner_isolation_canary(TASK_KEY)

    assert len(deleted) == 1
    assert deleted[0] != TASK_KEY
    assert checks["mismatchOverlayRemoved"] is True
    assert checks["overlayMismatchDenied"] is False


def test_artifact_export_uses_authenticated_uds_and_validates_the_task_bound_result(
    runner_fixture,
):
    server, socket_path, token_fd = runner_fixture
    server.response = _artifact_response(TASK_KEY)
    environment = _environment(socket_path, token_fd)

    result = environment.read_artifact("report.bin")

    assert result == {
        "taskRef": _task_ref(TASK_KEY),
        "filename": "report.bin",
        "sizeBytes": len(b"artifact-bytes"),
        "checksumSha256": hashlib.sha256(b"artifact-bytes").hexdigest(),
        "contentBase64": base64.b64encode(b"artifact-bytes").decode("ascii"),
    }
    assert server.requests == [
        {
            "path": "/v1/artifacts/read",
            "authorization": f"Bearer {TOKEN}",
            "contentType": "application/json",
            "body": {
                "schemaVersion": 1,
                "taskKey": TASK_KEY,
                "filename": "report.bin",
            },
        }
    ]


def test_artifact_export_accepts_an_integrity_valid_empty_file(runner_fixture):
    server, socket_path, token_fd = runner_fixture
    server.response = _artifact_response(TASK_KEY, content=b"")
    environment = _environment(socket_path, token_fd)

    result = environment.read_artifact("report.bin")

    assert result == {
        "taskRef": _task_ref(TASK_KEY),
        "filename": "report.bin",
        "sizeBytes": 0,
        "checksumSha256": hashlib.sha256(b"").hexdigest(),
        "contentBase64": "",
    }


def test_request_scoped_artifact_helper_never_reuses_another_task_capability(
    runner_fixture,
    monkeypatch,
):
    server, socket_path, token_fd = runner_fixture
    monkeypatch.setenv("HERMES_SANDBOX_RUNNER_SOCKET_PATH", str(socket_path))
    monkeypatch.setenv("HERMES_SANDBOX_RUNNER_TOKEN_FD", str(token_fd))
    monkeypatch.setattr(
        sandbox_runner,
        "_effective_uid",
        lambda: os.fstat(token_fd).st_uid + 1,
    )

    for task_id, task_key, content in (
        ("sandbox-task-" + ("a" * 64), TASK_KEY, b"A-ONLY"),
        ("sandbox-task-" + ("b" * 64), TASK_KEY_B, b"B-ONLY"),
    ):
        server.response = _artifact_response(task_key, content=content)
        with scoped_task_env_overrides(
            task_id,
            {"env_type": "sandbox_runner", "sandbox_task_key": task_key},
        ):
            result = read_sandbox_runner_artifact(task_id, "report.bin")
        assert base64.b64decode(result["contentBase64"], validate=True) == content
        assert result["taskRef"] == _task_ref(task_key)
        assert task_key not in json.dumps(result)
        assert server.requests[-1]["body"]["taskKey"] == task_key

        with pytest.raises(
            RuntimeError,
            match="Sandbox runner task identity is unavailable",
        ):
            read_sandbox_runner_artifact(task_id, "report.bin")

    with scoped_task_env_overrides(
        "default",
        {"env_type": "sandbox_runner", "sandbox_task_key": TASK_KEY_B},
    ):
        with pytest.raises(
            RuntimeError,
            match="Sandbox runner task identity is unavailable",
        ):
            read_sandbox_runner_artifact(
                "sandbox-task-" + ("c" * 64),
                "report.bin",
            )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda response: {**response, "extra": "field"},
        lambda response: {**response, "taskRef": _task_ref(TASK_KEY_B)},
        lambda response: {**response, "filename": "other.bin"},
        lambda response: {**response, "sizeBytes": True},
        lambda response: {**response, "sizeBytes": response["sizeBytes"] + 1},
        lambda response: {**response, "sizeBytes": 16 * 1_048_576 + 1},
        lambda response: {**response, "checksumSha256": "0" * 64},
        lambda response: {**response, "contentBase64": "YQ"},
    ],
)
def test_artifact_export_rejects_malformed_or_mismatched_runner_responses(
    runner_fixture,
    mutate,
    caplog,
):
    server, socket_path, token_fd = runner_fixture
    server.response = mutate(_artifact_response(TASK_KEY))
    environment = _environment(socket_path, token_fd)

    with pytest.raises(RuntimeError, match="Sandbox runner response is invalid"):
        environment.read_artifact("report.bin")

    assert TOKEN not in caplog.text
    assert TASK_KEY not in caplog.text


def test_artifact_export_rejects_json_lookalike_content_type(runner_fixture):
    server, socket_path, token_fd = runner_fixture
    server.response = _artifact_response(TASK_KEY)
    server.response_content_type = "application/jsonp"
    environment = _environment(socket_path, token_fd)

    with pytest.raises(RuntimeError, match="Sandbox runner response is invalid"):
        environment.read_artifact("report.bin")


@pytest.mark.parametrize("status", [401, 404, 413, 503])
def test_artifact_export_collapses_runner_statuses_without_secret_leakage(
    runner_fixture,
    status,
    caplog,
):
    server, socket_path, token_fd = runner_fixture
    server.status = status
    server.response = {
        "schemaVersion": 1,
        "error": {"code": "untrusted-runner-detail"},
    }
    environment = _environment(socket_path, token_fd)

    with pytest.raises(RuntimeError, match="Sandbox runner request was rejected"):
        environment.read_artifact("report.bin")

    assert "untrusted-runner-detail" not in caplog.text
    assert TOKEN not in caplog.text
    assert TASK_KEY not in caplog.text


@pytest.mark.parametrize(
    "filename",
    [
        "",
        ".",
        "..",
        "../report.bin",
        "nested/report.bin",
        "line\nbreak.bin",
        "\ud800",
    ],
)
def test_artifact_export_rejects_non_plain_filenames_before_transport(
    runner_fixture,
    filename,
):
    server, socket_path, token_fd = runner_fixture
    environment = _environment(socket_path, token_fd)

    with pytest.raises(RuntimeError, match="artifact filename is invalid"):
        environment.read_artifact(filename)

    assert server.requests == []


def test_artifact_export_rechecks_socket_and_token_metadata(runner_fixture):
    server, socket_path, token_fd = runner_fixture
    server.response = _artifact_response(TASK_KEY)
    environment = _environment(socket_path, token_fd)

    socket_path.chmod(0o666)
    with pytest.raises(RuntimeError, match="transport is unavailable"):
        environment.read_artifact("report.bin")
    assert server.requests == []

    socket_path.chmod(0o660)
    os.fchmod(token_fd, 0o644)
    with pytest.raises(RuntimeError, match="credential is unavailable"):
        environment.read_artifact("report.bin")
    assert server.requests == []


def test_artifact_export_suppresses_transport_cause_and_socket_path(runner_fixture):
    server, socket_path, token_fd = runner_fixture
    environment = _environment(socket_path, token_fd)
    socket_path.unlink()

    with pytest.raises(RuntimeError, match="transport is unavailable") as exc_info:
        environment.read_artifact("report.bin")

    rendered = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
        )
    )
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert exc_info.value.__suppress_context__ is True
    assert str(socket_path) not in rendered
    assert server.requests == []


def test_artifact_export_rechecks_the_fd_owner_contract(
    runner_fixture,
    monkeypatch,
):
    server, socket_path, token_fd = runner_fixture
    server.response = _artifact_response(TASK_KEY)
    token_owner = os.fstat(token_fd).st_uid
    monkeypatch.setattr(sandbox_runner, "_effective_uid", lambda: token_owner + 1)
    environment = SandboxRunnerEnvironment(
        task_key=TASK_KEY,
        socket_path=str(socket_path),
        token_fd=token_fd,
        initialize_session=False,
    )

    monkeypatch.setattr(sandbox_runner, "_effective_uid", lambda: token_owner)
    with pytest.raises(RuntimeError, match="credential is unavailable"):
        environment.read_artifact("report.bin")
    assert server.requests == []


def test_artifact_export_timeout_closes_the_uds_request(
    runner_fixture,
    monkeypatch,
):
    server, socket_path, token_fd = runner_fixture
    server.block_response = True
    environment = _environment(socket_path, token_fd)
    monkeypatch.setattr(
        sandbox_runner,
        "_ARTIFACT_REQUEST_TIMEOUT_SECONDS",
        0.1,
    )

    with pytest.raises(RuntimeError, match="artifact export failed closed"):
        environment.read_artifact("report.bin")

    server.release_response.set()
    assert server.disconnect_observed.wait(timeout=2)


def test_live_readiness_requires_the_exact_artifact_export_policy(
    runner_fixture,
    monkeypatch,
):
    server, socket_path, token_fd = runner_fixture
    monkeypatch.setenv("HERMES_SANDBOX_RUNNER_SOCKET_PATH", str(socket_path))
    monkeypatch.setenv("HERMES_SANDBOX_RUNNER_TOKEN_FD", str(token_fd))
    monkeypatch.setattr(
        sandbox_runner,
        "_effective_uid",
        lambda: os.fstat(token_fd).st_uid + 1,
    )

    assert sandbox_runner_ready_from_environment() is True
    artifact_export = server.capabilities["artifactExport"]
    assert isinstance(artifact_export, dict)
    artifact_export["pathPolicy"] = "unsafe_follow"
    assert sandbox_runner_ready_from_environment() is False


def test_live_readiness_allows_a_bounded_slow_host_probe(
    runner_fixture,
    monkeypatch,
):
    server, socket_path, token_fd = runner_fixture
    server.health_delay_seconds = 2.1
    monkeypatch.setenv("HERMES_SANDBOX_RUNNER_SOCKET_PATH", str(socket_path))
    monkeypatch.setenv("HERMES_SANDBOX_RUNNER_TOKEN_FD", str(token_fd))
    monkeypatch.setattr(
        sandbox_runner,
        "_effective_uid",
        lambda: os.fstat(token_fd).st_uid + 1,
    )

    assert sandbox_runner_ready_from_environment() is True


def test_live_readiness_still_fails_closed_after_its_bounded_timeout(
    runner_fixture,
    monkeypatch,
):
    server, socket_path, token_fd = runner_fixture
    server.health_delay_seconds = 0.2
    monkeypatch.setenv("HERMES_SANDBOX_RUNNER_SOCKET_PATH", str(socket_path))
    monkeypatch.setenv("HERMES_SANDBOX_RUNNER_TOKEN_FD", str(token_fd))
    monkeypatch.setattr(
        sandbox_runner,
        "_effective_uid",
        lambda: os.fstat(token_fd).st_uid + 1,
    )
    monkeypatch.setattr(
        sandbox_runner,
        "_READINESS_REQUEST_TIMEOUT_SECONDS",
        0.05,
    )

    assert sandbox_runner_ready_from_environment() is False


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


def test_explicit_remote_overlay_delete_uses_authenticated_runner_cleanup(
    runner_fixture,
):
    server, socket_path, token_fd = runner_fixture
    environment = _environment(socket_path, token_fd)

    assert environment.delete_remote_overlay() is True
    assert server.requests == [
        {
            "path": "/v1/cleanup",
            "authorization": f"Bearer {TOKEN}",
            "contentType": "application/json",
            "body": {
                "schemaVersion": 1,
                "taskKey": TASK_KEY,
            },
        }
    ]


def test_explicit_remote_overlay_delete_rejects_malformed_runner_response(
    runner_fixture,
):
    server, socket_path, token_fd = runner_fixture
    server.cleanup_response = {
        "schemaVersion": 1,
        "ok": True,
        "removed": "yes",
    }
    environment = _environment(socket_path, token_fd)

    with pytest.raises(RuntimeError, match="Sandbox runner response is invalid"):
        environment.delete_remote_overlay()


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

    # Unit tests own their fixture token. Simulate the production split owner;
    # both constructor and every positional token read re-check this contract.
    monkeypatch.setattr(
        sandbox_runner,
        "_effective_uid",
        lambda: os.fstat(token_fd).st_uid + 1,
    )
    assert sandbox_runner_ready_from_environment() is True

    os.fchmod(token_fd, 0o644)
    assert sandbox_runner_ready_from_environment() is False
