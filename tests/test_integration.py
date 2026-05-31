import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from httpx import AsyncClient

from pr_checker.github_client import GitHubClient
from pr_checker.review_orchestrator import ReviewOrchestrator
from tests.conftest import make_signature

_SECRET = "test-secret"  # noqa: S105


def _pr_payload(action: str = "opened") -> dict[str, Any]:
    return {
        "action": action,
        "number": 1,
        "pull_request": {
            "number": 1,
            "title": "Add feature",
            "body": None,
            "head": {"sha": "abc123", "ref": "feature-branch"},
            "base": {"sha": "def456", "ref": "main"},
        },
        "repository": {"full_name": "owner/repo"},
    }


def _post_webhook(
    payload: bytes,
    event: str = "pull_request",
    secret: str = _SECRET,
) -> dict[str, Any]:
    return {
        "content": payload,
        "headers": {
            "Content-Type": "application/json",
            "X-GitHub-Event": event,
            "X-Hub-Signature-256": make_signature(payload, secret),
        },
    }


# --- routing & validation ---


async def test_valid_pr_returns_202(client: AsyncClient, managed_app: FastAPI) -> None:
    payload = json.dumps(_pr_payload()).encode()
    with (
        patch.object(
            GitHubClient, "get_pr_head_sha", new_callable=AsyncMock, return_value="abc123"
        ),
        patch.object(GitHubClient, "post_commit_status", new_callable=AsyncMock),
        patch.object(ReviewOrchestrator, "run", new_callable=AsyncMock),
    ):
        r = await client.post("/webhook", **_post_webhook(payload))
        assert r.status_code == 202
        await asyncio.wait_for(managed_app.state.queue.join(), timeout=5)


async def test_invalid_signature_returns_401(client: AsyncClient) -> None:
    payload = json.dumps(_pr_payload()).encode()
    r = await client.post(
        "/webhook",
        content=payload,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": "sha256=badhex",
        },
    )
    assert r.status_code == 401


async def test_missing_signature_returns_401(client: AsyncClient) -> None:
    payload = json.dumps(_pr_payload()).encode()
    r = await client.post(
        "/webhook",
        content=payload,
        headers={"Content-Type": "application/json", "X-GitHub-Event": "pull_request"},
    )
    assert r.status_code == 401


async def test_non_pr_event_returns_204(client: AsyncClient) -> None:
    payload = b'{"ref":"refs/heads/main"}'
    r = await client.post("/webhook", **_post_webhook(payload, event="push"))
    assert r.status_code == 204


async def test_pr_closed_action_returns_204(client: AsyncClient) -> None:
    payload = json.dumps(_pr_payload("closed")).encode()
    r = await client.post("/webhook", **_post_webhook(payload))
    assert r.status_code == 204


async def test_pr_closed_action_calls_cancel_pr(client: AsyncClient, managed_app: FastAPI) -> None:
    payload = json.dumps(_pr_payload("closed")).encode()
    with patch.object(managed_app.state.queue, "cancel_pr", new_callable=AsyncMock) as mock_cancel:
        r = await client.post("/webhook", **_post_webhook(payload))
    assert r.status_code == 204
    mock_cancel.assert_called_once_with("owner/repo", 1)


async def test_pr_closed_with_non_numeric_pr_number_returns_204(client: AsyncClient) -> None:
    payload = json.dumps(
        {
            "action": "closed",
            "pull_request": {"number": "not-a-number"},
            "repository": {"full_name": "owner/repo"},
        }
    ).encode()
    r = await client.post("/webhook", **_post_webhook(payload))
    assert r.status_code == 204


# --- full flow ---


async def test_malformed_json_returns_400(client: AsyncClient) -> None:
    payload = b"not valid json{"
    r = await client.post("/webhook", **_post_webhook(payload))
    assert r.status_code == 400


async def test_full_review_flow(client: AsyncClient, managed_app: FastAPI) -> None:
    payload = json.dumps(_pr_payload()).encode()

    with (
        patch.object(GitHubClient, "post_commit_status", new_callable=AsyncMock) as mock_status,
        patch.object(
            GitHubClient, "get_pr_head_sha", new_callable=AsyncMock, return_value="abc123"
        ),
        patch.object(ReviewOrchestrator, "run", new_callable=AsyncMock) as mock_run,
    ):
        r = await client.post("/webhook", **_post_webhook(payload))
        assert r.status_code == 202

        await asyncio.wait_for(managed_app.state.queue.join(), timeout=5)

    states = [c.args[2] for c in mock_status.call_args_list]
    assert states == ["pending", "success"]
    mock_run.assert_called_once()
