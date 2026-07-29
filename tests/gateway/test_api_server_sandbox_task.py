"""Security contracts for SaaS-controlled sandbox task identities."""

import asyncio
import hashlib
import json
import threading
import uuid
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _IdempotencyCache
from hermes_state import SessionDB


AUTH = {"Authorization": "Bearer sk-test"}
SANDBOX_A = "sandbox-v1-" + ("A" * 43)
SANDBOX_B = "sandbox-v1-" + ("B" * 43)
RUNNER_INSTANCE_A = "sandbox-runner-v1-" + ("a" * 32)
RUNNER_INSTANCE_B = "sandbox-runner-v1-" + ("b" * 32)


def _execution_task_id(key):
    return "sandbox-task-" + hashlib.sha256(key.encode()).hexdigest()


def _adapter(tmp_path, *, required=True):
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "key": "sk-test",
                "sandbox_task_key_required": required,
            },
        )
    )
    adapter._session_db = SessionDB(tmp_path / "state.db")
    adapter._response_store = adapter._response_store.__class__(
        db_path=str(tmp_path / "responses.db")
    )
    return adapter


def _app(adapter):
    app = web.Application()
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_get("/health/detailed", adapter._handle_health_detailed)
    app.router.add_post(
        "/api/sessions/{session_id}/fork",
        adapter._handle_fork_session,
    )
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_post(
        "/v1/dedicated-sandbox-canary",
        adapter._handle_dedicated_sandbox_canary,
    )
    app.router.add_post(
        "/v1/dedicated-sandbox-identity",
        adapter._handle_dedicated_sandbox_identity,
    )
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return app


def _chat_body(*, stream=False):
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
    }


def _responses_body(*, stream=False, previous_response_id=None):
    body = {
        "model": "test-model",
        "input": "hello",
        "stream": stream,
    }
    if previous_response_id:
        body["previous_response_id"] = previous_response_id
    return body


def _result(text="ok"):
    return (
        {
            "final_response": text,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": text},
            ],
            "api_calls": 1,
        },
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )


@pytest.fixture(autouse=True)
def _sandbox_api_toolsets():
    """The controlled runtime enables code, but never delegation."""
    with patch(
        "gateway.run._load_gateway_config",
        return_value={
            "platform_toolsets": {
                "api_server": ["code_execution"],
            }
        },
    ):
        yield


@pytest.mark.asyncio
async def test_required_header_rejects_missing_malformed_and_unauthenticated_requests(
    tmp_path,
):
    adapter = _adapter(tmp_path)
    adapter._run_agent = pytest.fail
    app = _app(adapter)

    async with TestClient(TestServer(app)) as client:
        missing = await client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json=_chat_body(),
        )
        malformed = await client.post(
            "/v1/chat/completions",
            headers={**AUTH, "X-Hermes-Sandbox-Task-Key": "sandbox-v1-short"},
            json=_chat_body(),
        )
        unauthenticated = await client.post(
            "/v1/chat/completions",
            headers={"X-Hermes-Sandbox-Task-Key": SANDBOX_A},
            json=_chat_body(),
        )
        missing_body = await missing.json()
        malformed_body = await malformed.json()

    assert missing.status == 400
    assert missing_body["error"]["code"] == "sandbox_task_key_required"
    assert malformed.status == 400
    assert malformed_body["error"]["code"] == "invalid_sandbox_task_key"
    assert unauthenticated.status == 401


@pytest.mark.asyncio
async def test_dedicated_canary_requires_auth_and_uses_only_header_task_identity(
    tmp_path,
):
    adapter = _adapter(tmp_path)
    checks = {
        "sandboxTaskKeyHeaderAccepted": True,
        "taskIdIsolated": True,
        "taskIdFallbackNotDefault": True,
        "workspaceBindingPresent": True,
        "workspaceWriteRead": True,
        "overlayMismatchDenied": True,
        "secretEnvDenied": True,
        "egressDenied": True,
        "primaryMarkerRemoved": True,
        "mismatchOverlayRemoved": True,
    }
    app = _app(adapter)

    with patch(
        "tools.environments.sandbox_runner.run_sandbox_runner_isolation_canary",
        return_value=checks,
    ) as canary, patch(
        "tools.environments.sandbox_runner.sandbox_runner_identity_from_environment",
        side_effect=[
            {
                "runnerInstanceId": RUNNER_INSTANCE_A,
                "imageFingerprint": "sha256:" + ("a" * 64),
            },
            {
                "runnerInstanceId": RUNNER_INSTANCE_A,
                "imageFingerprint": "sha256:" + ("a" * 64),
            },
        ],
    ) as identity:
        async with TestClient(TestServer(app)) as client:
            unauthenticated = await client.post(
                "/v1/dedicated-sandbox-canary",
                headers={"X-Hermes-Sandbox-Task-Key": SANDBOX_A},
                json={},
            )
            missing = await client.post(
                "/v1/dedicated-sandbox-canary",
                headers=AUTH,
                json={},
            )
            response = await client.post(
                "/v1/dedicated-sandbox-canary",
                headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_A},
                json={},
            )
            payload = await response.json()

    assert unauthenticated.status == 401
    assert missing.status == 400
    assert response.status == 200
    assert payload == {
        "source": "hermes.sandbox_runner_canary",
        "runtimeInstanceId": adapter._runtime_instance_id,
        "runnerInstanceId": RUNNER_INSTANCE_A,
        "runnerImageFingerprint": "sha256:" + ("a" * 64),
        "checks": checks,
    }
    canary.assert_called_once_with(SANDBOX_A)
    assert identity.call_count == 2


@pytest.mark.asyncio
async def test_dedicated_canary_fails_closed_when_runner_identity_changes(tmp_path):
    adapter = _adapter(tmp_path)
    checks = {"workspaceWriteRead": True}
    app = _app(adapter)

    with patch(
        "tools.environments.sandbox_runner.run_sandbox_runner_isolation_canary",
        return_value=checks,
    ) as canary, patch(
        "tools.environments.sandbox_runner.sandbox_runner_identity_from_environment",
        side_effect=[
            {
                "runnerInstanceId": RUNNER_INSTANCE_A,
                "imageFingerprint": "sha256:" + ("a" * 64),
            },
            {
                "runnerInstanceId": RUNNER_INSTANCE_A,
                "imageFingerprint": "sha256:" + ("b" * 64),
            },
        ],
    ):
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/dedicated-sandbox-canary",
                headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_A},
                json={},
            )
            payload = await response.json()

    assert response.status == 503
    assert payload["error"]["code"] == "sandbox_runner_identity_changed_during_canary"
    canary.assert_called_once_with(SANDBOX_A)


@pytest.mark.asyncio
async def test_dedicated_canary_fails_closed_when_runner_restarts_with_same_sif(
    tmp_path,
):
    adapter = _adapter(tmp_path)
    app = _app(adapter)
    fingerprint = "sha256:" + ("a" * 64)

    with patch(
        "tools.environments.sandbox_runner.run_sandbox_runner_isolation_canary",
        return_value={"workspaceWriteRead": True},
    ), patch(
        "tools.environments.sandbox_runner.sandbox_runner_identity_from_environment",
        side_effect=[
            {
                "runnerInstanceId": RUNNER_INSTANCE_A,
                "imageFingerprint": fingerprint,
            },
            {
                "runnerInstanceId": RUNNER_INSTANCE_B,
                "imageFingerprint": fingerprint,
            },
        ],
    ):
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/dedicated-sandbox-canary",
                headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_A},
                json={},
            )
            payload = await response.json()

    assert response.status == 503
    assert payload["error"]["code"] == "sandbox_runner_identity_changed_during_canary"


@pytest.mark.asyncio
async def test_dedicated_canary_does_not_run_without_initial_runner_identity(tmp_path):
    adapter = _adapter(tmp_path)
    app = _app(adapter)

    with patch(
        "tools.environments.sandbox_runner.run_sandbox_runner_isolation_canary",
    ) as canary, patch(
        "tools.environments.sandbox_runner.sandbox_runner_identity_from_environment",
        return_value=None,
    ):
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/dedicated-sandbox-canary",
                headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_A},
                json={},
            )
            payload = await response.json()

    assert response.status == 503
    assert payload["error"]["code"] == "sandbox_runner_identity_unavailable"
    canary.assert_not_called()


@pytest.mark.asyncio
async def test_dedicated_identity_is_stable_per_process_and_requires_live_runner(
    tmp_path,
):
    adapter = _adapter(tmp_path)
    restarted_path = tmp_path / "restarted"
    restarted_path.mkdir()
    restarted_adapter = _adapter(restarted_path)
    app = _app(adapter)
    runner_fingerprint = "sha256:" + ("b" * 64)

    with patch(
        "tools.environments.sandbox_runner.sandbox_runner_identity_from_environment",
        side_effect=[
            {
                "runnerInstanceId": RUNNER_INSTANCE_A,
                "imageFingerprint": runner_fingerprint,
            },
            {
                "runnerInstanceId": RUNNER_INSTANCE_A,
                "imageFingerprint": runner_fingerprint,
            },
            None,
        ],
    ):
        async with TestClient(TestServer(app)) as client:
            unauthenticated = await client.post(
                "/v1/dedicated-sandbox-identity",
                headers={"X-Hermes-Sandbox-Task-Key": SANDBOX_A},
                json={},
            )
            missing_task_key = await client.post(
                "/v1/dedicated-sandbox-identity",
                headers=AUTH,
                json={},
            )
            first = await client.post(
                "/v1/dedicated-sandbox-identity",
                headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_A},
                json={},
            )
            second = await client.post(
                "/v1/dedicated-sandbox-identity",
                headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_A},
                json={},
            )
            unavailable = await client.post(
                "/v1/dedicated-sandbox-identity",
                headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_A},
                json={},
            )
            first_payload = await first.json()
            second_payload = await second.json()

    assert unauthenticated.status == 401
    assert missing_task_key.status == 400
    assert first.status == 200
    assert second.status == 200
    assert first_payload == second_payload == {
        "source": "hermes.sandbox_runner_identity",
        "runtimeInstanceId": adapter._runtime_instance_id,
        "runnerInstanceId": RUNNER_INSTANCE_A,
        "runnerImageFingerprint": runner_fingerprint,
    }
    assert restarted_adapter._runtime_instance_id != adapter._runtime_instance_id
    assert unavailable.status == 503


@pytest.mark.asyncio
async def test_dedicated_canary_fails_closed_when_live_runner_identity_is_missing(
    tmp_path,
):
    adapter = _adapter(tmp_path)
    app = _app(adapter)

    with patch(
        "tools.environments.sandbox_runner.run_sandbox_runner_isolation_canary",
        return_value={"sandboxTaskKeyHeaderAccepted": True},
    ), patch(
        "tools.environments.sandbox_runner.sandbox_runner_identity_from_environment",
        return_value=None,
    ):
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/dedicated-sandbox-canary",
                headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_A},
                json={},
            )
            payload = await response.json()

    assert response.status == 503
    assert payload["error"]["code"] == "sandbox_runner_identity_unavailable"


@pytest.mark.asyncio
async def test_dedicated_canary_rejects_non_split_runtime(tmp_path):
    adapter = _adapter(tmp_path, required=False)
    app = _app(adapter)

    with patch(
        "tools.environments.sandbox_runner.run_sandbox_runner_isolation_canary",
    ) as canary:
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/dedicated-sandbox-canary",
                headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_A},
                json={},
            )
            payload = await response.json()

    assert response.status == 503
    assert payload["error"]["code"] == "sandbox_runner_not_configured"
    canary.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
async def test_both_api_surfaces_and_stream_modes_use_header_only_identity(
    tmp_path,
    endpoint,
    stream,
):
    adapter = _adapter(tmp_path)
    observed = []

    async def fake_run_agent(**kwargs):
        observed.append(kwargs)
        return _result()

    adapter._run_agent = fake_run_agent
    app = _app(adapter)
    path = "/v1/chat/completions" if endpoint == "chat" else "/v1/responses"
    body = _chat_body(stream=stream) if endpoint == "chat" else _responses_body(stream=stream)

    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            path,
            headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_A},
            json=body,
        )
        response_text = await response.text()

    assert response.status == 200
    assert len(observed) == 1
    assert observed[0]["sandbox_task_key"] == SANDBOX_A
    assert SANDBOX_A not in response_text
    assert SANDBOX_A not in json.dumps(dict(response.headers))
    assert "X-Hermes-Sandbox-Task-Key" not in response.headers
    if endpoint == "responses":
        if stream:
            response_id = next(
                json.loads(line[6:])["response"]["id"]
                for line in response_text.splitlines()
                if line.startswith("data: ")
                and '"type": "response.created"' in line
            )
        else:
            response_id = json.loads(response_text)["id"]
        stored = adapter._response_store.get(response_id)
        assert stored["sandbox_context_hash"].startswith("sha256:")
        assert SANDBOX_A not in json.dumps(stored)


@pytest.mark.asyncio
async def test_chat_session_continuation_rejects_a_different_sandbox(tmp_path):
    adapter = _adapter(tmp_path)
    calls = []

    async def fake_run_agent(**kwargs):
        calls.append(kwargs["sandbox_task_key"])
        return _result()

    adapter._run_agent = fake_run_agent
    app = _app(adapter)
    session_id = "sandbox-continuation"

    async with TestClient(TestServer(app)) as client:
        first = await client.post(
            "/v1/chat/completions",
            headers={
                **AUTH,
                "X-Hermes-Session-Id": session_id,
                "X-Hermes-Sandbox-Task-Key": SANDBOX_A,
            },
            json=_chat_body(),
        )
        mismatch = await client.post(
            "/v1/chat/completions",
            headers={
                **AUTH,
                "X-Hermes-Session-Id": session_id,
                "X-Hermes-Sandbox-Task-Key": SANDBOX_B,
            },
            json=_chat_body(),
        )
        mismatch_body = await mismatch.json()

    assert first.status == 200
    assert mismatch.status == 409
    assert mismatch_body["error"]["code"] == "sandbox_task_context_mismatch"
    assert calls == [SANDBOX_A]


@pytest.mark.asyncio
async def test_derived_chat_session_id_is_scoped_to_the_sandbox(tmp_path):
    adapter = _adapter(tmp_path)
    calls = []

    async def fake_run_agent(**kwargs):
        calls.append(kwargs["sandbox_task_key"])
        return _result()

    adapter._run_agent = fake_run_agent
    app = _app(adapter)

    async with TestClient(TestServer(app)) as client:
        first_a = await client.post(
            "/v1/chat/completions",
            headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_A},
            json=_chat_body(),
        )
        first_b = await client.post(
            "/v1/chat/completions",
            headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_B},
            json=_chat_body(),
        )
        second_a = await client.post(
            "/v1/chat/completions",
            headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_A},
            json=_chat_body(),
        )

    assert [first_a.status, first_b.status, second_a.status] == [200, 200, 200]
    session_a = first_a.headers["X-Hermes-Session-Id"]
    session_b = first_b.headers["X-Hermes-Session-Id"]
    assert session_a != session_b
    assert second_a.headers["X-Hermes-Session-Id"] == session_a
    assert SANDBOX_A not in session_a
    assert SANDBOX_B not in session_b
    assert calls == [SANDBOX_A, SANDBOX_B, SANDBOX_A]


@pytest.mark.asyncio
async def test_controlled_sandbox_rejects_session_fork_before_mutation(tmp_path):
    adapter = _adapter(tmp_path)
    db = adapter._session_db
    db.create_session("sandbox-source", "api_server")
    db.append_message("sandbox-source", "user", "A-only history")
    assert db.bind_sandbox_context(
        "sandbox-source",
        adapter._sandbox_context_hash(SANDBOX_A),
    )
    app = _app(adapter)

    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/api/sessions/sandbox-source/fork",
            headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_A},
            json={"id": "sandbox-fork"},
        )
        body = await response.json()

    assert response.status == 503
    assert body["error"]["code"] == "sandbox_endpoint_unsupported"
    assert db.get_session("sandbox-source")["ended_at"] is None
    assert db.get_session("sandbox-fork") is None


@pytest.mark.asyncio
async def test_optional_sandbox_rejects_fork_of_bound_session_before_mutation(tmp_path):
    adapter = _adapter(tmp_path, required=False)
    db = adapter._session_db
    db.create_session("sandbox-source", "api_server")
    db.append_message("sandbox-source", "user", "A-only history")
    assert db.bind_sandbox_context(
        "sandbox-source",
        adapter._sandbox_context_hash(SANDBOX_A),
    )
    app = _app(adapter)

    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/api/sessions/sandbox-source/fork",
            headers=AUTH,
            json={"id": "sandbox-fork"},
        )
        body = await response.json()

    assert response.status == 503
    assert body["error"]["code"] == "sandbox_endpoint_unsupported"
    assert db.get_session("sandbox-source")["ended_at"] is None
    assert db.get_session("sandbox-fork") is None


def test_atomic_fork_loses_to_an_existing_sandbox_binding(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("sandbox-source", "api_server")
    db.append_message("sandbox-source", "user", "A-only history")
    assert db.bind_sandbox_context(
        "sandbox-source",
        "sha256:" + ("a" * 64),
    )

    assert (
        db.fork_session_if_unbound("sandbox-source", "sandbox-fork")
        == "source_sandbox_bound"
    )
    assert db.get_session("sandbox-source")["ended_at"] is None
    assert db.get_session("sandbox-fork") is None


@pytest.mark.asyncio
async def test_responses_previous_response_rejects_a_different_sandbox(tmp_path):
    adapter = _adapter(tmp_path)
    calls = []

    async def fake_run_agent(**kwargs):
        calls.append(kwargs["sandbox_task_key"])
        return _result()

    adapter._run_agent = fake_run_agent
    app = _app(adapter)

    async with TestClient(TestServer(app)) as client:
        first = await client.post(
            "/v1/responses",
            headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_A},
            json=_responses_body(),
        )
        first_body = await first.json()
        mismatch = await client.post(
            "/v1/responses",
            headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_B},
            json=_responses_body(previous_response_id=first_body["id"]),
        )
        mismatch_body = await mismatch.json()

    assert first.status == 200
    assert mismatch.status == 409
    assert mismatch_body["error"]["code"] == "sandbox_task_context_mismatch"
    assert calls == [SANDBOX_A]


@pytest.mark.asyncio
async def test_idempotency_key_does_not_reuse_results_across_sandboxes(tmp_path):
    adapter = _adapter(tmp_path)
    calls = []
    adapter_module = __import__(
        "gateway.platforms.api_server",
        fromlist=["_idem_cache"],
    )
    adapter_module._idem_cache = _IdempotencyCache()

    async def fake_run_agent(**kwargs):
        calls.append(kwargs["sandbox_task_key"])
        return _result(f"run-{len(calls)}")

    adapter._run_agent = fake_run_agent
    app = _app(adapter)
    idempotency_key = f"sandbox-{uuid.uuid4()}"

    async with TestClient(TestServer(app)) as client:
        for session_id, sandbox_key in [
            ("idem-a", SANDBOX_A),
            ("idem-b", SANDBOX_B),
        ]:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    **AUTH,
                    "Idempotency-Key": idempotency_key,
                    "X-Hermes-Session-Id": session_id,
                    "X-Hermes-Sandbox-Task-Key": sandbox_key,
                },
                json=_chat_body(),
            )
            assert response.status == 200

    assert calls == [SANDBOX_A, SANDBOX_B]


@pytest.mark.asyncio
async def test_run_agent_scopes_override_by_sandbox_and_cleans_it_on_all_paths(
    tmp_path,
):
    adapter = _adapter(tmp_path)
    observed = []

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, session_id, fail=False):
            self.session_id = session_id
            self.fail = fail

        def run_conversation(self, *, task_id, **_kwargs):
            from tools.terminal_tool import resolve_task_overrides

            observed.append((task_id, resolve_task_overrides(task_id)))
            if self.fail:
                raise RuntimeError(f"backend failed for {task_id}")
            return {"final_response": "ok"}

    monkey = patch.object(
        adapter,
        "_create_agent",
        side_effect=[
            FakeAgent("sandbox-success"),
            FakeAgent("sandbox-failure", fail=True),
        ],
    )
    with monkey:
        await adapter._run_agent(
            user_message="ok",
            conversation_history=[],
            session_id="sandbox-success",
            sandbox_task_key=SANDBOX_A,
        )
        with pytest.raises(RuntimeError) as error:
            await adapter._run_agent(
                user_message="fail",
                conversation_history=[],
                session_id="sandbox-failure",
                sandbox_task_key=SANDBOX_B,
            )

    from tools.terminal_tool import _task_env_overrides

    assert observed == [
        (
            _execution_task_id(SANDBOX_A),
            {
                "env_type": "sandbox_runner",
                "sandbox_task_key": SANDBOX_A,
            },
        ),
        (
            _execution_task_id(SANDBOX_B),
            {
                "env_type": "sandbox_runner",
                "sandbox_task_key": SANDBOX_B,
            },
        ),
    ]
    assert _execution_task_id(SANDBOX_A) not in _task_env_overrides
    assert _execution_task_id(SANDBOX_B) not in _task_env_overrides
    assert SANDBOX_B not in str(error.value)


@pytest.mark.asyncio
async def test_concurrent_same_sandbox_keeps_override_until_both_requests_finish(
    tmp_path,
):
    adapter = _adapter(tmp_path)
    barrier = threading.Barrier(2)
    release_first = threading.Event()
    observations = []

    class ConcurrentAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, session_id):
            self.session_id = session_id

        def run_conversation(self, *, task_id, **_kwargs):
            from tools.terminal_tool import resolve_task_overrides

            barrier.wait(timeout=5)
            observations.append(resolve_task_overrides(task_id))
            if self.session_id == "parallel-a":
                release_first.wait(timeout=5)
            else:
                release_first.set()
            observations.append(resolve_task_overrides(task_id))
            return {"final_response": "ok"}

    with patch.object(
        adapter,
        "_create_agent",
        side_effect=lambda **kwargs: ConcurrentAgent(kwargs["session_id"]),
    ):
        await asyncio.gather(
            adapter._run_agent(
                user_message="a",
                conversation_history=[],
                session_id="parallel-a",
                sandbox_task_key=SANDBOX_A,
            ),
            adapter._run_agent(
                user_message="b",
                conversation_history=[],
                session_id="parallel-b",
                sandbox_task_key=SANDBOX_A,
            ),
        )

    from tools.terminal_tool import _task_env_overrides

    assert observations == [
        {
            "env_type": "sandbox_runner",
            "sandbox_task_key": SANDBOX_A,
        }
    ] * 4
    assert _execution_task_id(SANDBOX_A) not in _task_env_overrides


@pytest.mark.asyncio
async def test_cancelled_request_keeps_override_until_executor_finishes(tmp_path):
    adapter = _adapter(tmp_path)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    execution_task_id = _execution_task_id(SANDBOX_A)

    class BlockingAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "cancelled-request"

        def run_conversation(self, **_kwargs):
            started.set()
            release.wait(timeout=5)
            finished.set()
            return {"final_response": "ok"}

    with patch.object(adapter, "_create_agent", return_value=BlockingAgent()):
        request_task = asyncio.create_task(
            adapter._run_agent(
                user_message="cancel",
                conversation_history=[],
                session_id="cancelled-request",
                sandbox_task_key=SANDBOX_A,
            )
        )
        assert await asyncio.to_thread(started.wait, 5)

        request_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request_task

        from tools.terminal_tool import resolve_task_overrides

        assert resolve_task_overrides(execution_task_id) == {
            "env_type": "sandbox_runner",
            "sandbox_task_key": SANDBOX_A,
        }

        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        for _ in range(100):
            if not resolve_task_overrides(execution_task_id):
                break
            await asyncio.sleep(0.01)

    assert resolve_task_overrides(execution_task_id) == {}


@pytest.mark.asyncio
async def test_sandbox_request_rejects_delegation_enabled_runtime(tmp_path):
    adapter = _adapter(tmp_path)
    adapter._run_agent = pytest.fail
    app = _app(adapter)

    with patch(
        "gateway.run._load_gateway_config",
        return_value={
            "platform_toolsets": {
                "api_server": ["code_execution", "delegation"],
            }
        },
    ):
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_A},
                json=_chat_body(),
            )
            response_body = await response.json()

    assert response.status == 503
    assert response_body["error"]["code"] == "sandbox_delegation_unsupported"


@pytest.mark.asyncio
async def test_controlled_mode_rejects_agent_endpoints_without_identity_bridge(
    tmp_path,
):
    adapter = _adapter(tmp_path)
    app = _app(adapter)

    async with TestClient(TestServer(app)) as client:
        missing = await client.post(
            "/v1/runs",
            headers=AUTH,
            json={"input": "hello"},
        )
        unsupported = await client.post(
            "/v1/runs",
            headers={**AUTH, "X-Hermes-Sandbox-Task-Key": SANDBOX_A},
            json={"input": "hello"},
        )
        missing_body = await missing.json()
        unsupported_body = await unsupported.json()

    assert missing.status == 400
    assert missing_body["error"]["code"] == "sandbox_task_key_required"
    assert unsupported.status == 503
    assert unsupported_body["error"]["code"] == "sandbox_endpoint_unsupported"


@pytest.mark.asyncio
async def test_capabilities_advertise_controlled_sandbox_contract(tmp_path):
    adapter = _adapter(tmp_path)
    app = _app(adapter)

    async with TestClient(TestServer(app)) as client:
        with patch(
            "tools.environments.sandbox_runner.sandbox_runner_ready_from_environment",
            return_value=True,
        ):
            response = await client.get("/v1/capabilities", headers=AUTH)
            body = await response.json()

    assert response.status == 200
    assert body["runtime"] == {
        "mode": "controlled_sandbox",
        "tool_execution": "sandbox_runner_required",
        "split_runtime": True,
        "description": (
            "Agent turns require a control-plane sandbox identity and "
            "execution is routed to the sandbox Runner."
        ),
    }
    assert body["features"]["sandbox_task_key_header"] == (
        "X-Hermes-Sandbox-Task-Key"
    )
    assert body["features"]["sandbox_task_key_required"] is True
    assert body["features"]["session_fork"] is False
    supported_endpoints = set(body["features"]["sandbox_task_supported_endpoints"])
    # This is a security allowlist, not a catalog snapshot: adding any agent
    # endpoint requires an explicit identity-bridge review and test update.
    assert supported_endpoints == {
        "/v1/chat/completions",
        "/v1/dedicated-sandbox-canary",
        "/v1/dedicated-sandbox-identity",
        "/v1/responses",
    }
    assert body["endpoints"]["dedicated_sandbox_canary"] == {
        "method": "POST",
        "path": "/v1/dedicated-sandbox-canary",
    }
    assert body["endpoints"]["dedicated_sandbox_identity"] == {
        "method": "POST",
        "path": "/v1/dedicated-sandbox-identity",
    }
    assert body["features"]["run_submission"] is False
    assert body["features"]["session_chat"] is False


@pytest.mark.asyncio
async def test_health_and_capabilities_fail_closed_when_runner_is_not_ready(tmp_path):
    adapter = _adapter(tmp_path)

    async with TestClient(TestServer(_app(adapter))) as client:
        with patch(
            "tools.environments.sandbox_runner.sandbox_runner_ready_from_environment",
            return_value=False,
        ):
            health = await client.get("/health")
            health_body = await health.json()
            capabilities = await client.get("/v1/capabilities", headers=AUTH)
            capabilities_body = await capabilities.json()

    assert health.status == 503
    assert health_body["status"] == "not_ready"
    assert capabilities.status == 503
    assert capabilities_body["error"]["code"] == "sandbox_runner_not_ready"


@pytest.mark.asyncio
async def test_detailed_health_reports_runner_degraded_without_transport_details(tmp_path):
    adapter = _adapter(tmp_path)

    async with TestClient(TestServer(_app(adapter))) as client:
        with (
            patch(
                "tools.environments.sandbox_runner.sandbox_runner_ready_from_environment",
                return_value=False,
            ),
            patch(
                "gateway.status.read_runtime_status",
                return_value={"gateway_state": "running"},
            ),
            patch(
                "gateway.run._resolve_gateway_model",
                return_value="test/model",
            ),
        ):
            response = await client.get("/health/detailed", headers=AUTH)
            body = await response.json()

    assert response.status == 200
    assert body["status"] == "degraded"
    assert body["readiness"]["checks"]["sandbox_runner"] == {"status": "degraded"}
    assert "socket" not in json.dumps(body).lower()
    assert "token" not in json.dumps(body).lower()


def test_session_db_binds_only_a_hash_and_rejects_conflicting_context(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    context_a = "sha256:" + ("a" * 64)
    context_b = "sha256:" + ("b" * 64)

    assert db.bind_sandbox_context("bound-session", context_a) is True
    assert db.bind_sandbox_context("bound-session", context_a) is True
    assert db.bind_sandbox_context("bound-session", context_b) is False

    session = db.get_session("bound-session")
    assert session["sandbox_context_hash"] == context_a
    assert SANDBOX_A not in json.dumps(session)
