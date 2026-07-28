"""Pytest wrapper for the shipped host-local Sandbox Runner live probe.

Run inside a split-runtime Hermes container whose production Runner socket and
root-owned token descriptor are already mounted::

    HERMES_SANDBOX_RUNNER_E2E=1 \
        pytest -m integration tests/integration/test_sandbox_runner_live.py -q

The executable probe itself lives in ``scripts/`` so split-runtime images can
run it without shipping tests or installing pytest.
"""

from __future__ import annotations

import os

import pytest

from scripts.sandbox_runner_live_e2e import run_live_cold_overlay_e2e


LIVE = os.environ.get("HERMES_SANDBOX_RUNNER_E2E") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not LIVE,
        reason="live-only: set HERMES_SANDBOX_RUNNER_E2E=1 in a split runtime",
    ),
]


def test_cold_overlay_canary_through_real_authenticated_runner():
    run_live_cold_overlay_e2e()
