"""Host-local Agent SaaS Sandbox Runner execution backend.

This backend is intentionally transport-only. The opaque task capability is
injected by the authenticated API-server request scope, while image, overlay,
network, quota, and host paths remain fixed by the Runner.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import re
import secrets
import shlex
import socket
import stat
import threading
from typing import Any

from tools.environments.base import BaseEnvironment, _ThreadedProcessHandle


DEFAULT_SOCKET_PATH = "/run/agent-saas-sandbox-runner/runner.sock"
DEFAULT_TOKEN_FD = 8
_TASK_KEY_RE = re.compile(r"^sandbox-v1-[A-Za-z0-9_-]{43}$")
_IMAGE_FINGERPRINT_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
_RUNNER_INSTANCE_ID_RE = re.compile(r"^sandbox-runner-v1-[a-f0-9]{32}$")
_MAX_COMMAND_CHARS = 65_536
_MAX_STDIN_CHARS = 1_048_576
_MAX_OUTPUT_BYTES = 1_048_576
_MAX_RESPONSE_BYTES = 2 * 1_048_576 + 4096
_MAX_ARTIFACT_BYTES = 16 * 1_048_576
_MAX_ARTIFACT_RESPONSE_BYTES = ((_MAX_ARTIFACT_BYTES + 2) // 3) * 4 + 65_536
_ARTIFACT_REQUEST_TIMEOUT_SECONDS = 35.0
_CLEANUP_REQUEST_TIMEOUT_SECONDS = 15.0
_READINESS_REQUEST_TIMEOUT_SECONDS = 5.0
_CANARY_EXECUTION_TIMEOUT_SECONDS = 30
_MAX_ARTIFACT_FILENAME_BYTES = 240
_MIN_TOKEN_BYTES = 32
_MAX_TOKEN_BYTES = 512
SANDBOX_RUNNER_CANARY_CHECKS = (
    "sandboxTaskKeyHeaderAccepted",
    "taskIdIsolated",
    "taskIdFallbackNotDefault",
    "workspaceBindingPresent",
    "workspaceWriteRead",
    "overlayMismatchDenied",
    "secretEnvDenied",
    "egressDenied",
    "primaryMarkerRemoved",
    "mismatchOverlayRemoved",
)
_CANARY_OUTPUT_PREFIX = "HERMES_SANDBOX_CANARY:"


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None)
    if not callable(getter):
        raise RuntimeError("Sandbox runner is available only on POSIX hosts.")
    return int(getter())


def _effective_gid() -> int:
    getter = getattr(os, "getegid", None)
    if not callable(getter):
        raise RuntimeError("Sandbox runner is available only on POSIX hosts.")
    return int(getter())


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self):
        transport = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        transport.settimeout(self.timeout)
        try:
            transport.connect(self._socket_path)
        except Exception:
            transport.close()
            raise
        self.sock = transport


class _CancelableRunnerCall:
    def __init__(self):
        self._lock = threading.Lock()
        self._connection: _UnixHTTPConnection | None = None
        self._cancelled = False

    def bind(self, connection: _UnixHTTPConnection):
        with self._lock:
            if self._cancelled:
                connection.close()
                raise RuntimeError("Sandbox runner execution was cancelled.")
            self._connection = connection

    def release(self, connection: _UnixHTTPConnection):
        with self._lock:
            if self._connection is connection:
                self._connection = None

    def cancel(self):
        with self._lock:
            self._cancelled = True
            connection = self._connection
            self._connection = None
        if connection is None:
            return
        try:
            if connection.sock is not None:
                connection.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        connection.close()


class SandboxRunnerEnvironment(BaseEnvironment):
    """Execute shell work through the authenticated host-local UDS Runner."""

    def __init__(
        self,
        *,
        task_key: str,
        socket_path: str = DEFAULT_SOCKET_PATH,
        token_fd: int = DEFAULT_TOKEN_FD,
        cwd: str = "/workspace",
        timeout: int = 180,
        initialize_session: bool = True,
        token_owner_must_differ: bool = True,
        canary_capacity_reservation: bool = False,
    ):
        super().__init__(cwd="/workspace" if not cwd else cwd, timeout=timeout)
        self._task_key = self._validate_task_key(task_key)
        self._socket_path = self._validate_socket_path(socket_path)
        self._token_owner_must_differ = token_owner_must_differ
        if not isinstance(canary_capacity_reservation, bool):
            raise RuntimeError("Sandbox runner lifecycle is unavailable.")
        self._canary_capacity_reservation = canary_capacity_reservation
        self._token_fd = self._validate_token_fd(
            token_fd,
            owner_must_differ=token_owner_must_differ,
        )
        self._calls: set[_CancelableRunnerCall] = set()
        self._calls_lock = threading.Lock()
        self._closed = False
        self._assert_socket_ready()
        if initialize_session:
            self.init_session()
        else:
            # Tests and narrowly-scoped callers may skip the shell snapshot.
            # Use a non-login shell rather than allowing BaseEnvironment to
            # probe a backend it was explicitly asked not to initialize.
            self._prefer_nonlogin = True

    @staticmethod
    def _validate_task_key(task_key: str) -> str:
        if not isinstance(task_key, str) or not _TASK_KEY_RE.fullmatch(task_key):
            raise RuntimeError("Sandbox runner task identity is unavailable.")
        return task_key

    @staticmethod
    def _validate_socket_path(socket_path: str) -> str:
        if (
            not isinstance(socket_path, str)
            or not os.path.isabs(socket_path)
            or "\x00" in socket_path
            or "\n" in socket_path
            or "\r" in socket_path
            or len(os.fsencode(socket_path)) > 100
        ):
            raise RuntimeError("Sandbox runner transport is unavailable.")
        return socket_path

    @staticmethod
    def _validate_token_fd(token_fd: int, *, owner_must_differ: bool) -> int:
        if isinstance(token_fd, bool) or not isinstance(token_fd, int) or token_fd < 3:
            raise RuntimeError("Sandbox runner credential is unavailable.")
        try:
            metadata = os.fstat(token_fd)
        except OSError as exc:
            raise RuntimeError("Sandbox runner credential is unavailable.") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (owner_must_differ and metadata.st_uid == _effective_uid())
        ):
            raise RuntimeError("Sandbox runner credential is unavailable.")
        return token_fd

    def _assert_socket_ready(self):
        try:
            metadata = os.lstat(self._socket_path)
        except OSError as exc:
            raise RuntimeError("Sandbox runner transport is unavailable.") from exc
        if (
            not stat.S_ISSOCK(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o660
            or metadata.st_gid != _effective_gid()
        ):
            raise RuntimeError("Sandbox runner transport is unavailable.")

    def _read_token(self) -> str:
        try:
            metadata = os.fstat(self._token_fd)
            raw = os.pread(self._token_fd, _MAX_TOKEN_BYTES + 2, 0)
        except OSError as exc:
            raise RuntimeError("Sandbox runner credential is unavailable.") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (
                self._token_owner_must_differ
                and metadata.st_uid == _effective_uid()
            )
        ):
            raise RuntimeError("Sandbox runner credential is unavailable.")
        if raw.endswith(b"\n"):
            raw = raw[:-1]
        if (
            len(raw) < _MIN_TOKEN_BYTES
            or len(raw) > _MAX_TOKEN_BYTES
            or any(byte in b" \t\r\n\v\f" for byte in raw)
        ):
            raise RuntimeError("Sandbox runner credential is unavailable.")
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("Sandbox runner credential is unavailable.") from exc

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ):
        call = _CancelableRunnerCall()

        def execute() -> tuple[str, int]:
            with self._calls_lock:
                if self._closed:
                    raise RuntimeError("Sandbox runner environment is closed.")
                self._calls.add(call)
            try:
                return self._execute_remote(
                    call,
                    cmd_string=cmd_string,
                    login=login,
                    timeout=timeout,
                    stdin_data=stdin_data,
                )
            finally:
                with self._calls_lock:
                    self._calls.discard(call)

        return _ThreadedProcessHandle(execute, cancel_fn=call.cancel)

    def _execute_remote(
        self,
        call: _CancelableRunnerCall,
        *,
        cmd_string: str,
        login: bool,
        timeout: int,
        stdin_data: str | None,
    ) -> tuple[str, int]:
        try:
            self._assert_socket_ready()
            if (
                not isinstance(cmd_string, str)
                or not cmd_string
                or len(cmd_string) > _MAX_COMMAND_CHARS
            ):
                raise RuntimeError(
                    "Sandbox runner command is outside the supported boundary."
                )
            if stdin_data is not None and (
                not isinstance(stdin_data, str)
                or len(stdin_data) > _MAX_STDIN_CHARS
                or len(stdin_data.encode("utf-8")) > _MAX_STDIN_CHARS
            ):
                raise RuntimeError(
                    "Sandbox runner stdin is outside the supported boundary."
                )
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or timeout < 0.1
                or timeout > 300
            ):
                raise RuntimeError(
                    "Sandbox runner timeout is outside the supported boundary."
                )

            shell_command = f"bash {'-l ' if login else ''}-c {shlex.quote(cmd_string)}"
            payload: dict[str, Any] = {
                "schemaVersion": 1,
                "taskKey": self._task_key,
                "command": shell_command,
                "timeoutMs": int(timeout * 1000),
            }
            if self._canary_capacity_reservation:
                payload["canaryCapacityReservation"] = True
            if stdin_data is not None:
                payload["stdin"] = stdin_data

            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            token = self._read_token()
            connection = _UnixHTTPConnection(
                self._socket_path,
                timeout=float(timeout) + 5.0,
            )
            call.bind(connection)
            try:
                connection.request(
                    "POST",
                    "/v1/exec",
                    body=body,
                    headers={
                        "authorization": f"Bearer {token}",
                        "content-type": "application/json",
                        "content-length": str(len(body)),
                    },
                )
                response = connection.getresponse()
                response_body = self._read_bounded_response(response)
            finally:
                call.release(connection)
                connection.close()

            if response.status != 200:
                raise RuntimeError("Sandbox runner request was rejected.")
            return self._parse_execution_response(response_body)
        except Exception as exc:
            if isinstance(exc, RuntimeError) and str(exc).startswith("Sandbox runner "):
                raise
            raise RuntimeError("Sandbox runner execution failed closed.") from exc

    @staticmethod
    def _read_bounded_response(
        response: http.client.HTTPResponse,
        *,
        max_response_bytes: int = _MAX_RESPONSE_BYTES,
    ) -> bytes:
        content_type = response.getheader("content-type", "")
        media_type = content_type.partition(";")[0].strip().lower()
        if media_type != "application/json":
            raise RuntimeError("Sandbox runner response is invalid.")
        content_length = response.getheader("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError as exc:
                raise RuntimeError("Sandbox runner response is invalid.") from exc
            if declared_length < 0 or declared_length > max_response_bytes:
                raise RuntimeError("Sandbox runner response is invalid.")
        body = response.read(max_response_bytes + 1)
        if len(body) > max_response_bytes:
            raise RuntimeError("Sandbox runner response is invalid.")
        return body

    @staticmethod
    def _parse_execution_response(body: bytes) -> tuple[str, int]:
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Sandbox runner response is invalid.") from exc
        allowed = {
            "schemaVersion",
            "ok",
            "exitCode",
            "stdout",
            "stderr",
            "timedOut",
        }
        if not isinstance(value, dict) or set(value) != allowed:
            raise RuntimeError("Sandbox runner response is invalid.")

        ok = value.get("ok")
        exit_code = value.get("exitCode")
        stdout = value.get("stdout")
        stderr = value.get("stderr")
        timed_out = value.get("timedOut")
        valid_exit_code = (
            isinstance(exit_code, int)
            and not isinstance(exit_code, bool)
            and 0 <= exit_code <= 255
        )
        if (
            value.get("schemaVersion") != 1
            or not isinstance(ok, bool)
            or not isinstance(stdout, str)
            or not isinstance(stderr, str)
            or not isinstance(timed_out, bool)
            or (timed_out and exit_code is not None)
            or (not timed_out and not valid_exit_code)
            or ok is not (not timed_out and exit_code == 0)
        ):
            raise RuntimeError("Sandbox runner response is invalid.")

        stdout_bytes = stdout.encode("utf-8")
        stderr_bytes = stderr.encode("utf-8")
        if len(stdout_bytes) + len(stderr_bytes) > _MAX_OUTPUT_BYTES:
            raise RuntimeError("Sandbox runner response is invalid.")

        output = stdout
        if stderr:
            output = f"{stdout}\n{stderr}" if stdout else stderr
        return output, 124 if timed_out else int(exit_code)

    def read_artifact(self, filename: str) -> dict[str, Any]:
        """Read one fixed-outbox artifact through the authenticated Runner."""
        filename = self._validate_artifact_filename(filename)
        call = _CancelableRunnerCall()
        with self._calls_lock:
            if self._closed:
                raise RuntimeError("Sandbox runner environment is closed.")
            self._calls.add(call)
        try:
            return self._read_artifact_remote(call, filename)
        finally:
            with self._calls_lock:
                self._calls.discard(call)

    def _read_artifact_remote(
        self,
        call: _CancelableRunnerCall,
        filename: str,
    ) -> dict[str, Any]:
        safe_error: str | None = None
        try:
            self._assert_socket_ready()
            payload = {
                "schemaVersion": 1,
                "taskKey": self._task_key,
                "filename": filename,
            }
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            token = self._read_token()
            connection = _UnixHTTPConnection(
                self._socket_path,
                timeout=_ARTIFACT_REQUEST_TIMEOUT_SECONDS,
            )
            call.bind(connection)
            try:
                connection.request(
                    "POST",
                    "/v1/artifacts/read",
                    body=body,
                    headers={
                        "authorization": f"Bearer {token}",
                        "content-type": "application/json",
                        "content-length": str(len(body)),
                    },
                )
                response = connection.getresponse()
                response_body = self._read_bounded_response(
                    response,
                    max_response_bytes=_MAX_ARTIFACT_RESPONSE_BYTES,
                )
            finally:
                call.release(connection)
                connection.close()

            if response.status != 200:
                raise RuntimeError("Sandbox runner request was rejected.")
            return self._parse_artifact_response(
                response_body,
                expected_filename=filename,
                expected_task_ref=self._task_ref(),
            )
        except Exception as exc:
            if isinstance(exc, RuntimeError) and str(exc).startswith("Sandbox runner "):
                safe_error = str(exc)
            else:
                safe_error = "Sandbox runner artifact export failed closed."

        # Raise after the except block so the new stable exception has neither
        # a cause nor a hidden context chain back to transport paths/details.
        raise RuntimeError(safe_error) from None

    @staticmethod
    def _validate_artifact_filename(filename: str) -> str:
        if not isinstance(filename, str):
            raise RuntimeError("Sandbox runner artifact filename is invalid.")
        try:
            encoded_filename = filename.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise RuntimeError("Sandbox runner artifact filename is invalid.") from exc
        if (
            not filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or any(
                ord(character) < 32 or 127 <= ord(character) <= 159
                for character in filename
            )
            or len(encoded_filename) > _MAX_ARTIFACT_FILENAME_BYTES
        ):
            raise RuntimeError("Sandbox runner artifact filename is invalid.")
        return filename

    def _task_ref(self) -> str:
        digest = hashlib.sha256(
            b"agent-saas-sandbox-runner-v1\0" + self._task_key.encode("utf-8")
        ).hexdigest()
        return f"sbx-{digest}"

    @staticmethod
    def _parse_artifact_response(
        body: bytes,
        *,
        expected_filename: str,
        expected_task_ref: str,
    ) -> dict[str, Any]:
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Sandbox runner response is invalid.") from exc
        allowed = {
            "schemaVersion",
            "ok",
            "taskRef",
            "filename",
            "sizeBytes",
            "checksumSha256",
            "contentBase64",
        }
        if not isinstance(value, dict) or set(value) != allowed:
            raise RuntimeError("Sandbox runner response is invalid.")

        size_bytes = value.get("sizeBytes")
        content_base64 = value.get("contentBase64")
        checksum = value.get("checksumSha256")
        if (
            value.get("schemaVersion") != 1
            or value.get("ok") is not True
            or value.get("taskRef") != expected_task_ref
            or value.get("filename") != expected_filename
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or size_bytes > _MAX_ARTIFACT_BYTES
            or not isinstance(checksum, str)
            or not re.fullmatch(r"[a-f0-9]{64}", checksum)
            or not isinstance(content_base64, str)
        ):
            raise RuntimeError("Sandbox runner response is invalid.")

        try:
            content = base64.b64decode(content_base64, validate=True)
        except Exception as exc:
            raise RuntimeError("Sandbox runner response is invalid.") from exc
        if (
            len(content) != size_bytes
            or base64.b64encode(content).decode("ascii") != content_base64
            or hashlib.sha256(content).hexdigest() != checksum
        ):
            raise RuntimeError("Sandbox runner response is invalid.")
        return {
            "taskRef": expected_task_ref,
            "filename": expected_filename,
            "sizeBytes": size_bytes,
            "checksumSha256": checksum,
            "contentBase64": content_base64,
        }

    def cleanup(self):
        """Release local transports only; durable overlay deletion is #625."""
        with self._calls_lock:
            if self._closed:
                return
            self._closed = True
            calls = list(self._calls)
            self._task_key = ""
        for call in calls:
            call.cancel()

    def delete_remote_overlay(self) -> bool:
        """Delete this task's durable overlay through the authenticated Runner."""
        call = _CancelableRunnerCall()
        with self._calls_lock:
            if self._closed:
                raise RuntimeError("Sandbox runner environment is closed.")
            self._calls.add(call)
        try:
            self._assert_socket_ready()
            payload = {
                "schemaVersion": 1,
                "taskKey": self._task_key,
            }
            if self._canary_capacity_reservation:
                payload["canaryCapacityReservation"] = True
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            token = self._read_token()
            connection = _UnixHTTPConnection(
                self._socket_path,
                timeout=_CLEANUP_REQUEST_TIMEOUT_SECONDS,
            )
            call.bind(connection)
            try:
                connection.request(
                    "POST",
                    "/v1/cleanup",
                    body=body,
                    headers={
                        "authorization": f"Bearer {token}",
                        "content-type": "application/json",
                        "content-length": str(len(body)),
                    },
                )
                response = connection.getresponse()
                response_body = self._read_bounded_response(response)
            finally:
                call.release(connection)
                connection.close()
            if response.status != 200:
                raise RuntimeError("Sandbox runner request was rejected.")
            value = json.loads(response_body.decode("utf-8"))
            if (
                not isinstance(value, dict)
                or set(value) != {"schemaVersion", "ok", "removed"}
                or value.get("schemaVersion") != 1
                or value.get("ok") is not True
                or not isinstance(value.get("removed"), bool)
            ):
                raise RuntimeError("Sandbox runner response is invalid.")
            return value["removed"]
        except Exception as exc:
            if isinstance(exc, RuntimeError) and str(exc).startswith("Sandbox runner "):
                raise
            raise RuntimeError("Sandbox runner cleanup failed closed.") from exc
        finally:
            with self._calls_lock:
                self._calls.discard(call)


def read_sandbox_runner_artifact(
    task_id: str,
    filename: str,
) -> dict[str, Any]:
    """Export one artifact using only the current request-scoped task lease."""
    from tools.terminal_tool import resolve_exact_task_overrides

    if not isinstance(task_id, str) or not task_id:
        raise RuntimeError("Sandbox runner task identity is unavailable.")
    overrides = resolve_exact_task_overrides(task_id)
    task_key = overrides.get("sandbox_task_key")
    if overrides.get("env_type") != "sandbox_runner" or not isinstance(task_key, str):
        raise RuntimeError("Sandbox runner task identity is unavailable.")

    raw_token_fd = os.getenv(
        "HERMES_SANDBOX_RUNNER_TOKEN_FD",
        str(DEFAULT_TOKEN_FD),
    )
    try:
        token_fd = int(raw_token_fd)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Sandbox runner credential is unavailable.") from exc
    socket_path = os.getenv(
        "HERMES_SANDBOX_RUNNER_SOCKET_PATH",
        DEFAULT_SOCKET_PATH,
    )
    environment = SandboxRunnerEnvironment(
        task_key=task_key,
        socket_path=socket_path,
        token_fd=token_fd,
        cwd="/workspace",
        initialize_session=False,
    )
    try:
        return environment.read_artifact(filename)
    finally:
        environment.cleanup()


def run_sandbox_runner_isolation_canary(task_key: str) -> dict[str, bool]:
    """Run bounded behavioral isolation checks through the real Runner."""
    checks = {name: False for name in SANDBOX_RUNNER_CANARY_CHECKS}
    environments: list[SandboxRunnerEnvironment] = []
    primary_key: str | None = None
    mismatch: SandboxRunnerEnvironment | None = None
    mismatch_executed = False
    marker: str | None = None
    try:
        primary_key = SandboxRunnerEnvironment._validate_task_key(task_key)
        nonce = secrets.token_hex(16)
        mismatch_key = "sandbox-v1-" + base64.urlsafe_b64encode(
            hashlib.sha256(
                b"agent-saas-sandbox-canary-mismatch-v1\0"
                + primary_key.encode("utf-8")
                + b"\0"
                + nonce.encode("ascii")
            ).digest()
        ).decode("ascii").rstrip("=")
        mismatch_key = SandboxRunnerEnvironment._validate_task_key(mismatch_key)
        marker = ".agent-saas-canary-" + nonce

        checks["sandboxTaskKeyHeaderAccepted"] = True
        checks["taskIdIsolated"] = primary_key != mismatch_key
        checks["taskIdFallbackNotDefault"] = (
            primary_key != "default" and mismatch_key != "default"
        )

        primary = _canary_environment(primary_key)
        environments.append(primary)
        probe = primary.execute(
            _canary_probe_script(marker),
            cwd="/workspace",
            timeout=_CANARY_EXECUTION_TIMEOUT_SECONDS,
        )
        probe_checks = _parse_canary_probe(probe)
        for name in (
            "workspaceBindingPresent",
            "workspaceWriteRead",
            "secretEnvDenied",
            "egressDenied",
        ):
            checks[name] = probe_checks.get(name) is True

        persisted = _canary_environment(primary_key)
        environments.append(persisted)
        persistence_result = persisted.execute(
            f"test -f /workspace/{marker}",
            cwd="/workspace",
            timeout=_CANARY_EXECUTION_TIMEOUT_SECONDS,
        )
        checks["workspaceWriteRead"] = (
            checks["workspaceWriteRead"]
            and persistence_result.get("returncode") == 0
        )

        # Reserve one additional capacity slot without changing the task-key,
        # durable overlay, or Apptainer execution path being tested. A task-id
        # alias with the primary therefore fails closed instead of being hidden
        # by a guaranteed-fresh temporary storage namespace.
        mismatch = _canary_environment(
            mismatch_key, canary_capacity_reservation=True
        )
        environments.append(mismatch)
        mismatch_result = mismatch.execute(
            f"test ! -e /workspace/{marker}",
            cwd="/workspace",
            timeout=_CANARY_EXECUTION_TIMEOUT_SECONDS,
        )
        mismatch_executed = True
        checks["overlayMismatchDenied"] = mismatch_result.get("returncode") == 0
    except Exception:
        pass
    finally:
        if mismatch is not None:
            try:
                removed = mismatch.delete_remote_overlay()
                checks["mismatchOverlayRemoved"] = removed or not mismatch_executed
            except Exception:
                checks["mismatchOverlayRemoved"] = False
        if primary_key is not None and marker is not None:
            try:
                cleanup = _canary_environment(primary_key)
                environments.append(cleanup)
                cleanup_result = cleanup.execute(
                    f"rm -f /workspace/{marker} && test ! -e /workspace/{marker}",
                    cwd="/workspace",
                    timeout=_CANARY_EXECUTION_TIMEOUT_SECONDS,
                )
                checks["primaryMarkerRemoved"] = (
                    cleanup_result.get("returncode") == 0
                )
            except Exception:
                checks["primaryMarkerRemoved"] = False
        for environment in environments:
            try:
                environment.cleanup()
            except Exception:
                pass
    return checks


def _canary_environment(
    task_key: str, *, canary_capacity_reservation: bool = False
) -> SandboxRunnerEnvironment:
    raw_token_fd = os.getenv(
        "HERMES_SANDBOX_RUNNER_TOKEN_FD",
        str(DEFAULT_TOKEN_FD),
    )
    try:
        token_fd = int(raw_token_fd)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Sandbox runner credential is unavailable.") from exc
    socket_path = os.getenv(
        "HERMES_SANDBOX_RUNNER_SOCKET_PATH",
        DEFAULT_SOCKET_PATH,
    )
    return SandboxRunnerEnvironment(
        task_key=task_key,
        socket_path=socket_path,
        token_fd=token_fd,
        cwd="/workspace",
        initialize_session=False,
        canary_capacity_reservation=canary_capacity_reservation,
    )


def _canary_probe_script(marker: str) -> str:
    marker_literal = json.dumps(marker)
    return f"""cd / && /usr/local/bin/python3 -I -S -P - <<'PY'
import json
import os

marker = {marker_literal}
marker_path = "/workspace/" + marker
checks = {{}}
checks["workspaceBindingPresent"] = (
    os.path.isdir("/workspace") and os.access("/workspace", os.W_OK)
)
try:
    with open(marker_path, "w", encoding="utf-8") as stream:
        stream.write(marker + "\\n")
    with open(marker_path, "r", encoding="utf-8") as stream:
        checks["workspaceWriteRead"] = stream.read().strip() == marker
except Exception:
    checks["workspaceWriteRead"] = False

secret_prefixes = (
    "API_SERVER_KEY",
    "AWS_",
    "AZURE_",
    "COOLIFY_",
    "DATABASE_URL",
    "DIRECT_URL",
    "GITHUB_",
    "GOOGLE_",
    "OP_",
    "SSH_",
    "KB_MCP_TOKEN",
    "GEMINI_API_KEY",
    "OPENAI_CODEX_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AGENT_SAAS_ARTIFACT_BRIDGE",
    "AGENT_SAAS_MEMORY_PROVIDER",
    "AGENT_SAAS_BUSINESS_TOOLS",
)
secret_names = [
    key for key in os.environ
    if key.endswith("_API_KEY")
    or key.endswith("_TOKEN")
    or key.endswith("_PASSWORD")
    or key.endswith("_SECRET")
    or any(
        key == prefix
        or key.startswith(prefix if prefix.endswith("_") else prefix + "_")
        for prefix in secret_prefixes
    )
]
try:
    with open("/proc/net/dev", "r", encoding="ascii") as stream:
        interfaces = sorted(
            line.split(":", 1)[0].strip()
            for line in stream.read().splitlines()[2:]
            if ":" in line
        )
    with open("/proc/net/route", "r", encoding="ascii") as stream:
        ipv4_route_rows = [
            line.split()
            for line in stream.read().splitlines()[1:]
            if line.strip()
        ]
    with open("/proc/net/ipv6_route", "r", encoding="ascii") as stream:
        ipv6_route_rows = [
            line.split()
            for line in stream.read().splitlines()
            if line.strip()
        ]
    checks["egressDenied"] = (
        interfaces == ["lo"]
        and all(row and row[0] == "lo" for row in ipv4_route_rows)
        and all(len(row) >= 10 and row[9] == "lo" for row in ipv6_route_rows)
    )
except Exception:
    checks["egressDenied"] = False
checks["secretEnvDenied"] = len(secret_names) == 0
print({_CANARY_OUTPUT_PREFIX!r} + json.dumps(checks, sort_keys=True))
PY"""


def _parse_canary_probe(result: dict[str, Any]) -> dict[str, bool]:
    if result.get("returncode") != 0:
        return {}
    output = result.get("output")
    if not isinstance(output, str):
        return {}
    for line in output.splitlines():
        if not line.startswith(_CANARY_OUTPUT_PREFIX):
            continue
        try:
            value = json.loads(line[len(_CANARY_OUTPUT_PREFIX):])
        except json.JSONDecodeError:
            return {}
        if not isinstance(value, dict):
            return {}
        return {
            str(name): item is True
            for name, item in value.items()
        }
    return {}


def sandbox_runner_ready_from_environment() -> bool:
    """Authenticate to the mounted Runner and validate its live policy.

    Called by the API-server health/capability surfaces. All failures collapse
    to ``False`` so transport or credential details never enter an HTTP body or
    log message.
    """
    state = _sandbox_runner_live_state_from_environment()
    if state is None:
        return False
    return _sandbox_runner_live_policy_ready(*state)


def sandbox_runner_identity_from_environment() -> dict[str, str] | None:
    """Return the live, readiness-verified Runner generation and SIF fingerprint.

    The unauthenticated health probe recomputes the configured SIF digest before
    reporting ready, while the authenticated capabilities response carries the
    bounded fingerprint. Returning no paths or transport details keeps this safe
    for the API-server's control-plane identity endpoint.
    """
    state = _sandbox_runner_live_state_from_environment()
    if state is None or not _sandbox_runner_live_policy_ready(*state):
        return None
    fingerprint = state[1].get("imageFingerprint")
    runner_instance_id = state[1].get("runnerInstanceId")
    if not isinstance(fingerprint, str) or not _IMAGE_FINGERPRINT_RE.fullmatch(
        fingerprint
    ):
        return None
    if not isinstance(runner_instance_id, str) or not _RUNNER_INSTANCE_ID_RE.fullmatch(
        runner_instance_id
    ):
        return None
    return {
        "runnerInstanceId": runner_instance_id,
        "imageFingerprint": fingerprint,
    }


def _sandbox_runner_live_state_from_environment() -> tuple[dict, dict] | None:
    try:
        raw_token_fd = os.getenv(
            "HERMES_SANDBOX_RUNNER_TOKEN_FD",
            str(DEFAULT_TOKEN_FD),
        )
        token_fd = int(raw_token_fd)
        socket_path = os.getenv(
            "HERMES_SANDBOX_RUNNER_SOCKET_PATH",
            DEFAULT_SOCKET_PATH,
        )
        probe = object.__new__(SandboxRunnerEnvironment)
        probe._socket_path = SandboxRunnerEnvironment._validate_socket_path(
            socket_path,
        )
        probe._token_fd = SandboxRunnerEnvironment._validate_token_fd(
            token_fd,
            owner_must_differ=True,
        )
        probe._token_owner_must_differ = True
        probe._assert_socket_ready()
        token = probe._read_token()

        health = _runner_probe_json(
            socket_path=probe._socket_path,
            path="/health",
            token=None,
        )
        capabilities = _runner_probe_json(
            socket_path=probe._socket_path,
            path="/v1/capabilities",
            token=token,
        )
        return health, capabilities
    except Exception:
        return None


def _sandbox_runner_live_policy_ready(health: dict, capabilities: dict) -> bool:
    return (
        health.get("schemaVersion") == 1
        and health.get("status") == "ready"
        and isinstance(health.get("checks"), dict)
        and isinstance(health.get("runnerInstanceId"), str)
        and bool(_RUNNER_INSTANCE_ID_RE.fullmatch(health["runnerInstanceId"]))
        and capabilities.get("schemaVersion") == 1
        and isinstance(capabilities.get("runnerInstanceId"), str)
        and bool(
            _RUNNER_INSTANCE_ID_RE.fullmatch(capabilities["runnerInstanceId"])
        )
        and health.get("runnerInstanceId") == capabilities.get("runnerInstanceId")
        and capabilities.get("isolation") == "per_task_overlay"
        and capabilities.get("network") == "disabled"
        and isinstance(capabilities.get("imageFingerprint"), str)
        and bool(
            _IMAGE_FINGERPRINT_RE.fullmatch(capabilities["imageFingerprint"])
        )
        and capabilities.get("artifactExport")
        == {
            "outbox": "/workspace/artifacts",
            "pathPolicy": "plain_filename_no_follow",
            "maxBytes": _MAX_ARTIFACT_BYTES,
        }
        and isinstance(capabilities.get("limits"), dict)
    )


def _runner_probe_json(
    *,
    socket_path: str,
    path: str,
    token: str | None,
) -> dict[str, Any]:
    connection = _UnixHTTPConnection(
        socket_path,
        timeout=_READINESS_REQUEST_TIMEOUT_SECONDS,
    )
    headers = {"accept": "application/json"}
    if token is not None:
        headers["authorization"] = f"Bearer {token}"
    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        body = SandboxRunnerEnvironment._read_bounded_response(response)
    finally:
        connection.close()
    if response.status != 200:
        raise RuntimeError("Sandbox runner readiness probe failed.")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Sandbox runner readiness probe failed.") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Sandbox runner readiness probe failed.")
    return value
