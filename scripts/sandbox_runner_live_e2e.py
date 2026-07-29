#!/usr/bin/env python3
"""Run the cold-overlay canary through a split runtime's real Runner.

This probe ships in the production image and has no pytest dependency. Run it
as the Hermes user with the production Runner token already open on the
configured descriptor::

    HERMES_SANDBOX_RUNNER_E2E=1 \
      /opt/hermes/.venv/bin/python3 scripts/sandbox_runner_live_e2e.py

It creates only random synthetic task capabilities and cleans both the primary
and mismatch overlays before returning. It never prints either capability.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import urllib.request

from tools.environments.sandbox_runner import (
    DEFAULT_SOCKET_PATH,
    DEFAULT_TOKEN_FD,
    SANDBOX_RUNNER_CANARY_CHECKS,
    SandboxRunnerEnvironment,
    run_sandbox_runner_isolation_canary,
    sandbox_runner_identity_from_environment,
    sandbox_runner_ready_from_environment,
)


_RUNTIME_INSTANCE_ID_RE = re.compile(r"^hermes-runtime-v1-[a-f0-9]{32}$")
_RUNNER_INSTANCE_ID_RE = re.compile(r"^sandbox-runner-v1-[a-f0-9]{32}$")
_IMAGE_FINGERPRINT_RE = re.compile(r"^sha256:[a-f0-9]{64}$")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward the Gateway bearer credential across a redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _gateway_loopback_url() -> str:
    """Build the only permitted Gateway origin: cleartext loopback + local port."""
    raw_port = os.environ.get("API_SERVER_PORT", "8642").strip()
    if not raw_port.isascii() or not raw_port.isdigit():
        raise RuntimeError("Gateway identity probe failed closed.")
    port = int(raw_port)
    if not 1 <= port <= 65535:
        raise RuntimeError("Gateway identity probe failed closed.")
    return f"http://127.0.0.1:{port}"


def _unique_task_key() -> str:
    return "sandbox-v1-" + base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(
        "ascii"
    ).rstrip("=")


def _environment(task_key: str) -> SandboxRunnerEnvironment:
    return SandboxRunnerEnvironment(
        task_key=task_key,
        socket_path=os.environ.get(
            "HERMES_SANDBOX_RUNNER_SOCKET_PATH",
            DEFAULT_SOCKET_PATH,
        ),
        token_fd=int(
            os.environ.get(
                "HERMES_SANDBOX_RUNNER_TOKEN_FD",
                str(DEFAULT_TOKEN_FD),
            )
        ),
        initialize_session=False,
    )


def _gateway_identity(task_key: str) -> dict[str, str]:
    """Read the production Gateway identity endpoint without exposing secrets."""
    api_key = os.environ.get("API_SERVER_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Gateway identity probe failed closed.")
    base_url = _gateway_loopback_url()
    request = urllib.request.Request(
        f"{base_url}/v1/dedicated-sandbox-identity",
        data=b"{}",
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Hermes-Sandbox-Task-Key": task_key,
        },
    )
    try:
        opener = urllib.request.build_opener(_NoRedirectHandler())
        with opener.open(request, timeout=15) as response:
            if response.status != 200:
                raise RuntimeError("Gateway identity probe failed closed.")
            raw = response.read(8193)
    except Exception as exc:
        raise RuntimeError("Gateway identity probe failed closed.") from exc
    if len(raw) > 8192:
        raise RuntimeError("Gateway identity probe failed closed.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Gateway identity probe failed closed.") from exc
    if not isinstance(payload, dict) or payload.get("source") != (
        "hermes.sandbox_runner_identity"
    ):
        raise RuntimeError("Gateway identity probe failed closed.")
    return payload


def _identity_snapshot(task_key: str) -> tuple[str, str, str]:
    """Bind live Runner readiness to the real Gateway identity response."""
    if not sandbox_runner_ready_from_environment():
        raise RuntimeError("Runner readiness probe failed closed.")
    runner = sandbox_runner_identity_from_environment()
    gateway = _gateway_identity(task_key)
    if runner is None:
        raise RuntimeError("Runner identity probe failed closed.")
    runtime_instance_id = gateway.get("runtimeInstanceId")
    runner_instance_id = gateway.get("runnerInstanceId")
    fingerprint = gateway.get("runnerImageFingerprint")
    if (
        not isinstance(runtime_instance_id, str)
        or not _RUNTIME_INSTANCE_ID_RE.fullmatch(runtime_instance_id)
        or not isinstance(runner_instance_id, str)
        or not _RUNNER_INSTANCE_ID_RE.fullmatch(runner_instance_id)
        or not isinstance(fingerprint, str)
        or not _IMAGE_FINGERPRINT_RE.fullmatch(fingerprint)
        or runner.get("runnerInstanceId") != runner_instance_id
        or runner.get("imageFingerprint") != fingerprint
    ):
        raise RuntimeError("Gateway identity probe failed closed.")
    return runtime_instance_id, runner_instance_id, fingerprint


def run_live_cold_overlay_e2e() -> None:
    """Require an absent primary overlay, real canary pass, and full cleanup."""
    task_key = _unique_task_key()
    lifecycle = _environment(task_key)
    try:
        identity_before = _identity_snapshot(task_key)
        if lifecycle.delete_remote_overlay() is not False:
            raise RuntimeError("Cold-overlay precondition failed closed.")
        checks = run_sandbox_runner_isolation_canary(task_key)
        if checks != {name: True for name in SANDBOX_RUNNER_CANARY_CHECKS}:
            raise RuntimeError("Cold-overlay canary failed closed.")
        if lifecycle.delete_remote_overlay() is not True:
            raise RuntimeError("Cold-overlay cleanup failed closed.")
        if lifecycle.delete_remote_overlay() is not False:
            raise RuntimeError("Cold-overlay cleanup verification failed closed.")
        if _identity_snapshot(task_key) != identity_before:
            raise RuntimeError("Runtime identity changed during live canary.")
    finally:
        try:
            lifecycle.delete_remote_overlay()
        finally:
            lifecycle.cleanup()


def main() -> None:
    if os.environ.get("HERMES_SANDBOX_RUNNER_E2E") != "1":
        raise SystemExit("HERMES_SANDBOX_RUNNER_E2E=1 is required.")
    run_live_cold_overlay_e2e()
    print("sandbox runner cold-overlay e2e passed")


if __name__ == "__main__":
    main()
