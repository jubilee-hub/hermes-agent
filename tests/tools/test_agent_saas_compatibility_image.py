"""CI-discovered image contract for the Agent SaaS compatibility overlay."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PATCH_REVISION = "compatibility-image-contract-test"
_OPTIMIZED_PROBE = """
import sys
import types
from pathlib import Path

base = types.ModuleType("tools.environments.base")
base.BaseEnvironment = type("BaseEnvironment", (), {"__init__": lambda self, *args, **kwargs: None})
base._ThreadedProcessHandle = type("_ThreadedProcessHandle", (), {})
sys.modules["tools.environments.base"] = base

handler = Path("/opt/hermes/gateway/platforms/api_server.py")
compile(handler.read_text(encoding="utf-8"), str(handler), "exec")
import tools.environments.sandbox_runner
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
    print("compatibility image probe fail-closed passed")
else:
    raise SystemExit("optimized compatibility probe accepted failed checks")
"""


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return (
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon is unavailable for the compatibility image contract",
)


def test_compatibility_image_packages_and_executes_runner_probe():
    image = f"hermes-agent-compat-contract:{uuid.uuid4().hex}"
    built = False
    try:
        build = subprocess.run(
            [
                "docker",
                "build",
                "--target",
                "agent_saas_compatibility_contract",
                "-f",
                "Dockerfile.agent-saas-session-root",
                "--build-arg",
                f"PATCH_REVISION={_PATCH_REVISION}",
                "-t",
                image,
                ".",
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        assert build.returncode == 0, build.stderr[-4000:]
        built = True

        revision = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                '{{index .Config.Labels "io.jubilee.hermes.patch-revision"}}',
                image,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert revision.returncode == 0, revision.stderr
        assert revision.stdout.strip() == _PATCH_REVISION

        execution = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                "hermes",
                "--workdir",
                "/opt/hermes",
                "--entrypoint",
                "/usr/local/bin/python3",
                image,
                "-O",
                "-c",
                _OPTIMIZED_PROBE,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert execution.returncode == 0, execution.stderr
        assert "compatibility image probe fail-closed passed" in execution.stdout
    finally:
        if built:
            subprocess.run(
                ["docker", "image", "rm", "--force", image],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
