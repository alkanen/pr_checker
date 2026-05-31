import asyncio
from collections.abc import AsyncGenerator
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest

from pr_checker.db import PersistenceLayer
from pr_checker.exceptions import StalePRError
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
    mock = AsyncMock(spec=GitHubClient)
    mock.get_pr_head_sha.return_value = "abc123"
    return mock


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


def _job(pr_number: int = 1, head_sha: str = "abc123") -> PRJob:
    return PRJob(
        repo_full_name="owner/repo",
        pr_number=pr_number,
        pr_title="Test PR",
        head_sha=head_sha,
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


# --- stale-SHA / eager cancellation ---


async def test_in_flight_job_cancelled_when_new_push_arrives(
    queue: ReviewQueue,
    persistence: PersistenceLayer,
    mock_github: AsyncMock,
    mock_orchestrator: AsyncMock,
) -> None:
    j1_started = asyncio.Event()

    async def stalling_run(job: PRJob) -> None:
        if job.job_id == j1.job_id:
            j1_started.set()
            await asyncio.sleep(3600)  # cancelled by j2's enqueue

    mock_orchestrator.run.side_effect = stalling_run
    mock_github.get_pr_head_sha.side_effect = ["sha1", "sha2"]

    j1 = _job(head_sha="sha1")
    j2 = _job(head_sha="sha2")

    await queue.enqueue(j1)
    await asyncio.wait_for(j1_started.wait(), timeout=5)

    await queue.enqueue(j2)  # cancels j1, enqueues j2

    await asyncio.wait_for(queue.join(), timeout=5)

    j1_db = await persistence.get_job(j1.job_id)
    j2_db = await persistence.get_job(j2.job_id)
    assert j1_db is not None and j1_db.status == JobStatus.CANCELLED
    assert j2_db is not None and j2_db.status == JobStatus.COMPLETED
    mock_github.post_pr_comment.assert_not_called()


async def test_queued_job_superseded_before_it_starts(
    persistence: PersistenceLayer,
    mock_github: AsyncMock,
    mock_orchestrator: AsyncMock,
) -> None:
    # Enqueue three jobs before starting the worker so the supersede chain is
    # deterministic: j1 and j2 are marked superseded before any of them runs.
    q = ReviewQueue(
        persistence,
        cast(GitHubClient, mock_github),
        cast(ReviewOrchestrator, mock_orchestrator),
    )

    mock_github.get_pr_head_sha.return_value = "sha3"
    j1 = _job(head_sha="sha1")
    j2 = _job(head_sha="sha2")  # will be superseded by j3
    j3 = _job(head_sha="sha3")

    await q.enqueue(j1)  # _pending[key] = j1
    await q.enqueue(j2)  # j1 superseded, _pending[key] = j2
    await q.enqueue(j3)  # j2 superseded, _pending[key] = j3

    await q.start()
    await asyncio.wait_for(q.join(), timeout=5)
    await q.stop()

    j1_db = await persistence.get_job(j1.job_id)
    j2_db = await persistence.get_job(j2.job_id)
    j3_db = await persistence.get_job(j3.job_id)
    assert j1_db is not None and j1_db.status == JobStatus.CANCELLED
    assert j2_db is not None and j2_db.status == JobStatus.CANCELLED
    assert j3_db is not None and j3_db.status == JobStatus.COMPLETED

    mock_github.post_pr_comment.assert_not_called()


async def test_cancelled_job_does_not_post_error_comment(
    queue: ReviewQueue,
    mock_github: AsyncMock,
    mock_orchestrator: AsyncMock,
) -> None:
    started = asyncio.Event()

    async def stalling_run(job: PRJob) -> None:
        if job.job_id == j1.job_id:
            started.set()
            await asyncio.sleep(3600)

    mock_orchestrator.run.side_effect = stalling_run
    mock_github.get_pr_head_sha.side_effect = ["sha1", "sha2"]

    j1 = _job(head_sha="sha1")
    j2 = _job(head_sha="sha2")

    await queue.enqueue(j1)
    await asyncio.wait_for(started.wait(), timeout=5)
    await queue.enqueue(j2)

    await asyncio.wait_for(queue.join(), timeout=5)

    mock_github.post_pr_comment.assert_not_called()


async def test_stale_pr_marks_job_cancelled(
    queue: ReviewQueue,
    persistence: PersistenceLayer,
    mock_github: AsyncMock,
    mock_orchestrator: AsyncMock,
) -> None:
    mock_orchestrator.run.side_effect = StalePRError("SHA mismatch: abc123 != def456")

    job = _job()
    await queue.enqueue(job)
    await asyncio.wait_for(queue.join(), timeout=5)

    fetched = await persistence.get_job(job.job_id)
    assert fetched is not None
    assert fetched.status == JobStatus.CANCELLED
    mock_github.post_pr_comment.assert_not_called()


async def test_stale_pr_does_not_post_error_commit_status(
    queue: ReviewQueue,
    mock_github: AsyncMock,
    mock_orchestrator: AsyncMock,
) -> None:
    mock_orchestrator.run.side_effect = StalePRError("SHA mismatch")

    await queue.enqueue(_job())
    await asyncio.wait_for(queue.join(), timeout=5)

    states = [c.args[2] for c in mock_github.post_commit_status.call_args_list]
    assert "error" not in states
    assert "failure" not in states


async def test_cancel_pr_cancels_in_flight_job(
    queue: ReviewQueue,
    persistence: PersistenceLayer,
    mock_github: AsyncMock,
    mock_orchestrator: AsyncMock,
) -> None:
    started = asyncio.Event()

    async def stalling_run(job: PRJob) -> None:
        started.set()
        await asyncio.sleep(3600)

    mock_orchestrator.run.side_effect = stalling_run

    job = _job()
    await queue.enqueue(job)
    await asyncio.wait_for(started.wait(), timeout=5)

    await queue.cancel_pr(job.repo_full_name, job.pr_number)

    await asyncio.wait_for(queue.join(), timeout=5)

    fetched = await persistence.get_job(job.job_id)
    assert fetched is not None and fetched.status == JobStatus.CANCELLED
    mock_github.post_pr_comment.assert_not_called()


async def test_cancel_pr_supersedes_pending_job(
    persistence: PersistenceLayer,
    mock_github: AsyncMock,
    mock_orchestrator: AsyncMock,
) -> None:
    q = ReviewQueue(
        persistence,
        cast(GitHubClient, mock_github),
        cast(ReviewOrchestrator, mock_orchestrator),
    )

    job = _job()
    await q.enqueue(job)  # queued, worker not yet started
    await q.cancel_pr(job.repo_full_name, job.pr_number)

    await q.start()
    await asyncio.wait_for(q.join(), timeout=5)
    await q.stop()

    fetched = await persistence.get_job(job.job_id)
    assert fetched is not None and fetched.status == JobStatus.CANCELLED
    mock_orchestrator.run.assert_not_called()
    mock_github.post_pr_comment.assert_not_called()


async def test_cancel_pr_noop_when_no_job(
    queue: ReviewQueue,
    mock_orchestrator: AsyncMock,
) -> None:
    await queue.cancel_pr("owner/repo", 999)
    mock_orchestrator.run.assert_not_called()


async def test_different_prs_do_not_interfere(
    queue: ReviewQueue,
    persistence: PersistenceLayer,
    mock_orchestrator: AsyncMock,
) -> None:
    """Jobs for different PR numbers on the same repo run independently."""
    j1 = _job(pr_number=1)
    j2 = _job(pr_number=2)

    await queue.enqueue(j1)
    await queue.enqueue(j2)
    await asyncio.wait_for(queue.join(), timeout=5)

    j1_db = await persistence.get_job(j1.job_id)
    j2_db = await persistence.get_job(j2.job_id)
    assert j1_db is not None and j1_db.status == JobStatus.COMPLETED
    assert j2_db is not None and j2_db.status == JobStatus.COMPLETED
    assert mock_orchestrator.run.call_count == 2


async def test_create_job_failure_leaves_pending_job_intact(
    persistence: PersistenceLayer,
    mock_github: AsyncMock,
    mock_orchestrator: AsyncMock,
) -> None:
    q = ReviewQueue(
        persistence,
        cast(GitHubClient, mock_github),
        cast(ReviewOrchestrator, mock_orchestrator),
    )

    mock_github.get_pr_head_sha.return_value = "sha1"
    j1 = _job(head_sha="sha1")
    await q.enqueue(j1)  # pending in queue, worker not started

    j2 = _job(head_sha="sha2")
    original_create_job = persistence.create_job

    async def failing_create_job(job: PRJob) -> None:
        if job.job_id == j2.job_id:
            raise Exception("DB write failed")
        return await original_create_job(job)

    with patch.object(persistence, "create_job", side_effect=failing_create_job):
        with pytest.raises(Exception, match="DB write failed"):
            await q.enqueue(j2)

    assert not j1.superseded

    await q.start()
    await asyncio.wait_for(q.join(), timeout=5)
    await q.stop()

    j1_db = await persistence.get_job(j1.job_id)
    assert j1_db is not None and j1_db.status == JobStatus.COMPLETED


async def test_stop_with_active_job_does_not_hang(
    persistence: PersistenceLayer,
    mock_github: AsyncMock,
    mock_orchestrator: AsyncMock,
) -> None:
    """stop() must return promptly even when a job task is active and still running."""
    running = asyncio.Event()

    async def stalling_run(job: PRJob) -> None:
        running.set()
        await asyncio.sleep(3600)

    mock_orchestrator.run.side_effect = stalling_run

    q = ReviewQueue(
        persistence,
        cast(GitHubClient, mock_github),
        cast(ReviewOrchestrator, mock_orchestrator),
    )
    await q.start()
    await q.enqueue(_job())
    await asyncio.wait_for(running.wait(), timeout=5)

    # stop() must not hang: it cancels the active task and the worker, then returns.
    # 15s gives the internal 5s queue-drain timeout room to expire plus cleanup time.
    await asyncio.wait_for(q.stop(), timeout=15)
