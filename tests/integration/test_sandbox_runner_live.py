"""Opt-in live E2E for the authenticated host-local Sandbox Runner.

Run inside a split-runtime Hermes container whose production Runner socket and
root-owned token descriptor are already mounted::

    HERMES_SANDBOX_RUNNER_E2E=1 \
        pytest -m integration tests/integration/test_sandbox_runner_live.py -q

The test starts with a unique absent primary overlay, drives the same canary
used by the API, and deletes the primary overlay before returning. The canary
itself creates and deletes its unique mismatch overlay.
"""

from __future__ import annotations

import base64
import os
import secrets

from tools.environments.sandbox_runner import (
    DEFAULT_SOCKET_PATH,
    DEFAULT_TOKEN_FD,
    SANDBOX_RUNNER_CANARY_CHECKS,
    SandboxRunnerEnvironment,
    run_sandbox_runner_isolation_canary,
)


LIVE = os.environ.get("HERMES_SANDBOX_RUNNER_E2E") == "1"

try:
    import pytest
except ModuleNotFoundError:  # Production images run this file directly.
    pytest = None

if pytest is not None:
    pytestmark = [
        pytest.mark.integration,
        pytest.mark.skipif(
            not LIVE,
            reason="live-only: set HERMES_SANDBOX_RUNNER_E2E=1 in a split runtime",
        ),
    ]


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


def test_cold_overlay_canary_through_real_authenticated_runner():
    task_key = _unique_task_key()
    lifecycle = _environment(task_key)
    try:
        # A random capability should have no durable state before this test.
        assert lifecycle.delete_remote_overlay() is False

        checks = run_sandbox_runner_isolation_canary(task_key)
        assert checks == {name: True for name in SANDBOX_RUNNER_CANARY_CHECKS}

        # The primary cold overlay was created, while the canary already proved
        # and cleaned its separate mismatch overlay.
        assert lifecycle.delete_remote_overlay() is True
        assert lifecycle.delete_remote_overlay() is False
    finally:
        # Idempotent fail-safe cleanup if an assertion or transport step failed.
        try:
            lifecycle.delete_remote_overlay()
        finally:
            lifecycle.cleanup()


if __name__ == "__main__":
    if not LIVE:
        raise SystemExit("HERMES_SANDBOX_RUNNER_E2E=1 is required.")
    test_cold_overlay_canary_through_real_authenticated_runner()
    print("sandbox runner cold-overlay e2e passed")
