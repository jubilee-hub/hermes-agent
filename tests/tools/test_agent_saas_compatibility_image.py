"""CI-discovered contract for the pinned Agent SaaS production image."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PATCH_REVISION = "compatibility-image-contract-test"
_OFFICIAL_BASE_DIGEST = (
    "sha256:f7b35053268f532f98955195c909f15a230470fbcbdacaa9fdecb95707dad04a"
)
_OPTIMIZED_PROBE = """
import hashlib
import json
import os
import tempfile
from pathlib import Path

import gateway.platforms.api_server
from hermes_state import SessionDB
import tools.environments.sandbox_runner
from tools.terminal_tool import scoped_task_env_overrides
from scripts import sandbox_runner_live_e2e as probe

if os.geteuid() != 10000:
    raise RuntimeError("compatibility image probe did not run as hermes")

scoped_task_id = "sandbox-task-" + ("d" * 64)
with scoped_task_env_overrides(
    scoped_task_id,
    {
        "env_type": "sandbox_runner",
        "sandbox_task_key": "sandbox-v1-" + ("D" * 43),
    },
):
    pass

raw_task_key = "sandbox-v1-" + ("A" * 43)
context_a = "sha256:" + hashlib.sha256(raw_task_key.encode()).hexdigest()
context_b = "sha256:" + ("b" * 64)
state_path = Path(tempfile.mkdtemp()) / "state.db"
db = SessionDB(state_path)
if not db.bind_sandbox_context("image-contract-session", context_a):
    raise RuntimeError("compatibility image did not bind sandbox context")
if not db.bind_sandbox_context("image-contract-session", context_a):
    raise RuntimeError("compatibility image sandbox context was not idempotent")
if db.bind_sandbox_context("image-contract-session", context_b):
    raise RuntimeError("compatibility image accepted sandbox context rebinding")
stored = db.get_session("image-contract-session")
if stored["sandbox_context_hash"] != context_a:
    raise RuntimeError("compatibility image replaced the bound sandbox context")
if raw_task_key in json.dumps(stored) or raw_task_key.encode() in state_path.read_bytes():
    raise RuntimeError("compatibility image persisted the raw sandbox task key")
db.close()

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
                '{{index .Config.Labels "io.jubilee.hermes.patch-revision"}} {{index .Config.Labels "org.opencontainers.image.base.digest"}}',
                image,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert revision.returncode == 0, revision.stderr
        assert revision.stdout.strip() == f"{_PATCH_REVISION} {_OFFICIAL_BASE_DIGEST}"

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
                "/opt/hermes/.venv/bin/python3",
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
