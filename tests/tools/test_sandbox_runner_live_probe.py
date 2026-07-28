from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import sandbox_runner_live_e2e as live_probe


class _FakeLifecycle:
    def __init__(self, cleanup_results: list[bool]):
        self.cleanup_results = iter(cleanup_results)
        self.closed = False

    def delete_remote_overlay(self) -> bool:
        return next(self.cleanup_results)

    def cleanup(self) -> None:
        self.closed = True


def test_live_probe_requires_exact_checks_and_cleanup(monkeypatch):
    lifecycle = _FakeLifecycle([False, True, False, False])
    monkeypatch.setattr(live_probe, "_environment", lambda _task_key: lifecycle)
    monkeypatch.setattr(
        live_probe,
        "run_sandbox_runner_isolation_canary",
        lambda _task_key: {
            name: True for name in live_probe.SANDBOX_RUNNER_CANARY_CHECKS
        },
    )

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


def test_compatibility_image_packages_runner_probe_and_implementation():
    dockerfile = (
        Path(__file__).resolve().parents[2] / "Dockerfile.agent-saas-session-root"
    ).read_text(encoding="utf-8")
    assert "tools/environments/sandbox_runner.py" in dockerfile
    assert "scripts/sandbox_runner_live_e2e.py" in dockerfile
