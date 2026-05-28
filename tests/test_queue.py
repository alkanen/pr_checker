import asyncio
from collections.abc import AsyncGenerator
from typing import cast
from unittest.mock import AsyncMock

import pytest

from pr_checker.db import PersistenceLayer
from pr_checker.github_client import GitHubClient
from pr_checker.models import JobStatus, PRJob, ReviewTrigger
from pr_checker.queue import ReviewQueue
from pr_checker.review_orchestrator import ReviewOrchestrator


@pytest.fixture
async def persistence() -> AsyncGenerator[PersistenceLayer, None]:
    layer = PersistenceLayer("sqlite+aiosqlite:///:memory:")
    await layer.init()
    yield layer
    await layer.dispose()


@pytest.fixture
def mock_github() -> AsyncMock:
    return AsyncMock(spec=GitHubClient)


@pytest.fixture
def mock_orchestrator() -> AsyncMock:
    return AsyncMock(spec=ReviewOrchestrator)


@pytest.fixture
async def queue(
    persistence: PersistenceLayer,
    mock_github: AsyncMock,
    mock_orchestrator: AsyncMock,
) -> AsyncGenerator[ReviewQueue, None]:
    q = ReviewQueue(
        persistence,
        cast(GitHubClient, mock_github),
        cast(ReviewOrchestrator, mock_orchestrator),
    )
    await q.start()
    yield q
    await q.stop()


def _job() -> PRJob:
    return PRJob(
        repo_full_name="owner/repo",
        pr_number=1,
        pr_title="Test PR",
        head_sha="abc123",
        base_sha="def456",
        head_branch="feature",
        base_branch="main",
        trigger=ReviewTrigger.OPENED,
    )


async def test_successful_review_marks_job_completed(
    queue: ReviewQueue, persistence: PersistenceLayer
) -> None:
    job = _job()
    await queue.enqueue(job)
    await asyncio.wait_for(queue.join(), timeout=5)

    fetched = await persistence.get_job(job.job_id)
    assert fetched is not None
    assert fetched.status == JobStatus.COMPLETED
    assert fetched.started_at is not None
    assert fetched.completed_at is not None


async def test_successful_review_posts_status_and_calls_orchestrator(
    queue: ReviewQueue, mock_github: AsyncMock, mock_orchestrator: AsyncMock
) -> None:
    await queue.enqueue(_job())
    await asyncio.wait_for(queue.join(), timeout=5)

    assert mock_github.post_commit_status.call_count == 2
    states = [c.args[2] for c in mock_github.post_commit_status.call_args_list]
    assert states == ["pending", "success"]
    mock_orchestrator.run.assert_called_once()


async def test_orchestrator_failure_marks_job_failed_and_posts_error(
    queue: ReviewQueue,
    persistence: PersistenceLayer,
    mock_github: AsyncMock,
    mock_orchestrator: AsyncMock,
) -> None:
    mock_orchestrator.run.side_effect = Exception("LLM timeout")

    job = _job()
    await queue.enqueue(job)
    await asyncio.wait_for(queue.join(), timeout=5)

    fetched = await persistence.get_job(job.job_id)
    assert fetched is not None
    assert fetched.status == JobStatus.FAILED
    assert fetched.error_message == "LLM timeout"
    mock_github.post_pr_comment.assert_called_once()


async def test_error_comment_failure_is_swallowed(
    queue: ReviewQueue,
    persistence: PersistenceLayer,
    mock_github: AsyncMock,
    mock_orchestrator: AsyncMock,
) -> None:
    mock_orchestrator.run.side_effect = Exception("broken")
    mock_github.post_pr_comment.side_effect = Exception("also broken")

    job = _job()
    await queue.enqueue(job)
    await asyncio.wait_for(queue.join(), timeout=5)

    fetched = await persistence.get_job(job.job_id)
    assert fetched is not None
    assert fetched.status == JobStatus.FAILED
