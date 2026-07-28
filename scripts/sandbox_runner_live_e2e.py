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
import os
import secrets

from tools.environments.sandbox_runner import (
    DEFAULT_SOCKET_PATH,
    DEFAULT_TOKEN_FD,
    SANDBOX_RUNNER_CANARY_CHECKS,
    SandboxRunnerEnvironment,
    run_sandbox_runner_isolation_canary,
)


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


def run_live_cold_overlay_e2e() -> None:
    """Require an absent primary overlay, real canary pass, and full cleanup."""
    task_key = _unique_task_key()
    lifecycle = _environment(task_key)
    try:
        if lifecycle.delete_remote_overlay() is not False:
            raise RuntimeError("Cold-overlay precondition failed closed.")
        checks = run_sandbox_runner_isolation_canary(task_key)
        if checks != {name: True for name in SANDBOX_RUNNER_CANARY_CHECKS}:
            raise RuntimeError("Cold-overlay canary failed closed.")
        if lifecycle.delete_remote_overlay() is not True:
            raise RuntimeError("Cold-overlay cleanup failed closed.")
        if lifecycle.delete_remote_overlay() is not False:
            raise RuntimeError("Cold-overlay cleanup verification failed closed.")
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
