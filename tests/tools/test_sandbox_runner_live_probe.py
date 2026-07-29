from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from scripts import sandbox_runner_live_e2e as live_probe


def _gateway_payload() -> bytes:
    return json.dumps({
        "source": "hermes.sandbox_runner_identity",
        "runtimeInstanceId": "hermes-runtime-v1-" + ("a" * 32),
        "runnerInstanceId": "sandbox-runner-v1-" + ("b" * 32),
        "runnerImageFingerprint": "sha256:" + ("c" * 64),
    }).encode("utf-8")


def _start_gateway_server(mode: str):
    state = {"redirect_target_seen": False, "authorization": None}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            state["authorization"] = self.headers.get("Authorization")
            if self.path == "/redirect-target":
                state["redirect_target_seen"] = True
                body = _gateway_payload()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if mode == "redirect":
                self.send_response(307)
                self.send_header(
                    "Location",
                    f"http://127.0.0.1:{self.server.server_port}/redirect-target",
                )
                self.end_headers()
                return
            if mode == "valid":
                body = _gateway_payload()
            elif mode == "malformed":
                body = b"{"
            else:
                body = b"x" * 8193
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, state


class _FakeLifecycle:
    def __init__(self, cleanup_results: list[bool]):
        self.cleanup_results = iter(cleanup_results)
        self.closed = False

    def delete_remote_overlay(self) -> bool:
        return next(self.cleanup_results)

    def cleanup(self) -> None:
        self.closed = True


def _configure_gateway_probe(monkeypatch, server, key="test-secret"):
    config = GatewayConfig(
        platforms={
            Platform.API_SERVER: PlatformConfig(
                enabled=True,
                extra={"key": key, "port": server.server_port},
            )
        }
    )
    monkeypatch.setattr(live_probe, "load_gateway_config", lambda: config)


@pytest.mark.parametrize("mode", ["malformed", "oversize"])
def test_gateway_identity_rejects_malformed_or_oversize_loopback_response(
    monkeypatch,
    mode,
):
    server, thread, state = _start_gateway_server(mode)
    _configure_gateway_probe(monkeypatch, server)
    try:
        with pytest.raises(RuntimeError, match="failed closed"):
            live_probe._gateway_identity("sandbox-v1-test")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert state["authorization"] == "Bearer test-secret"


def test_gateway_identity_never_follows_redirect_with_bearer_key(monkeypatch):
    server, thread, state = _start_gateway_server("redirect")
    _configure_gateway_probe(monkeypatch, server)
    try:
        with pytest.raises(RuntimeError, match="failed closed"):
            live_probe._gateway_identity("sandbox-v1-test")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert state["redirect_target_seen"] is False


def test_gateway_identity_uses_effective_config_and_bypasses_http_proxy(monkeypatch):
    server, thread, state = _start_gateway_server("valid")
    _configure_gateway_probe(monkeypatch, server, key="config-only-secret")
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    try:
        payload = live_probe._gateway_identity("sandbox-v1-test")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert payload["source"] == "hermes.sandbox_runner_identity"
    assert state["authorization"] == "Bearer config-only-secret"


def test_live_probe_requires_exact_checks_and_cleanup(monkeypatch):
    lifecycle = _FakeLifecycle([False, True, False, False])
    identity_calls = []
    monkeypatch.setattr(live_probe, "_environment", lambda _task_key: lifecycle)
    monkeypatch.setattr(
        live_probe,
        "_identity_snapshot",
        lambda task_key: (
            identity_calls.append(task_key)
            or (
                "hermes-runtime-v1-" + ("a" * 32),
                "sandbox-runner-v1-" + ("b" * 32),
                "sha256:" + ("c" * 64),
            )
        ),
    )
    monkeypatch.setattr(
        live_probe,
        "run_sandbox_runner_isolation_canary",
        lambda _task_key: {
            name: True for name in live_probe.SANDBOX_RUNNER_CANARY_CHECKS
        },
    )

    live_probe.run_live_cold_overlay_e2e()

    assert lifecycle.closed is True
    assert len(identity_calls) == 2


def test_live_probe_rejects_gateway_or_runner_identity_drift(monkeypatch):
    lifecycle = _FakeLifecycle([False, True, False, False])
    snapshots = iter((
        (
            "hermes-runtime-v1-" + ("a" * 32),
            "sandbox-runner-v1-" + ("b" * 32),
            "sha256:" + ("c" * 64),
        ),
        (
            "hermes-runtime-v1-" + ("a" * 32),
            "sandbox-runner-v1-" + ("d" * 32),
            "sha256:" + ("c" * 64),
        ),
    ))
    monkeypatch.setattr(live_probe, "_environment", lambda _task_key: lifecycle)
    monkeypatch.setattr(
        live_probe, "_identity_snapshot", lambda _task_key: next(snapshots)
    )
    monkeypatch.setattr(
        live_probe,
        "run_sandbox_runner_isolation_canary",
        lambda _task_key: {
            name: True for name in live_probe.SANDBOX_RUNNER_CANARY_CHECKS
        },
    )

    with pytest.raises(RuntimeError, match="identity changed"):
        live_probe.run_live_cold_overlay_e2e()

    assert lifecycle.closed is True


def test_optimized_python_cannot_ignore_failed_canary():
    script = """
from scripts import sandbox_runner_live_e2e as probe

class Lifecycle:
    def delete_remote_overlay(self):
        return False
    def cleanup(self):
        return None

probe._environment = lambda _task_key: Lifecycle()
probe._identity_snapshot = lambda _task_key: (
    "hermes-runtime-v1-" + ("a" * 32),
    "sandbox-runner-v1-" + ("b" * 32),
    "sha256:" + ("c" * 64),
)
probe.run_sandbox_runner_isolation_canary = lambda _task_key: {
    name: False for name in probe.SANDBOX_RUNNER_CANARY_CHECKS
}
try:
    probe.run_live_cold_overlay_e2e()
except RuntimeError:
    raise SystemExit(0)
raise SystemExit(1)
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
