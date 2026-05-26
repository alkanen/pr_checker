from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

import pytest

from pr_checker.db import PersistenceLayer
from pr_checker.models import JobStatus, PRJob, ReviewTrigger


@pytest.fixture
async def persistence() -> AsyncGenerator[PersistenceLayer, None]:
    layer = PersistenceLayer("sqlite+aiosqlite:///:memory:")
    await layer.init()
    yield layer
    await layer.dispose()


def _job(**kwargs: Any) -> PRJob:
    defaults: dict[str, Any] = {
        "repo_full_name": "owner/repo",
        "pr_number": 1,
        "pr_title": "Test PR",
        "head_sha": "abc123",
        "base_sha": "def456",
        "head_branch": "feature",
        "base_branch": "main",
        "trigger": ReviewTrigger.OPENED,
    }
    defaults.update(kwargs)
    return PRJob(**defaults)


async def test_create_and_get(persistence: PersistenceLayer) -> None:
    job = _job()
    await persistence.create_job(job)

    fetched = await persistence.get_job(job.job_id)
    assert fetched is not None
    assert fetched.job_id == job.job_id
    assert fetched.status == JobStatus.PENDING
    assert fetched.repo_full_name == "owner/repo"
    assert fetched.trigger == ReviewTrigger.OPENED
    assert fetched.started_at is None


async def test_update_to_in_progress(persistence: PersistenceLayer) -> None:
    job = _job()
    await persistence.create_job(job)

    job.status = JobStatus.IN_PROGRESS
    job.started_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    await persistence.update_job(job)

    fetched = await persistence.get_job(job.job_id)
    assert fetched is not None
    assert fetched.status == JobStatus.IN_PROGRESS
    assert fetched.started_at == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


async def test_update_to_completed(persistence: PersistenceLayer) -> None:
    job = _job()
    await persistence.create_job(job)

    job.status = JobStatus.COMPLETED
    job.completed_at = datetime(2024, 1, 1, 12, 5, 0, tzinfo=timezone.utc)
    await persistence.update_job(job)

    fetched = await persistence.get_job(job.job_id)
    assert fetched is not None
    assert fetched.status == JobStatus.COMPLETED
    assert fetched.completed_at == datetime(2024, 1, 1, 12, 5, 0, tzinfo=timezone.utc)
    assert fetched.error_message is None


async def test_update_to_failed(persistence: PersistenceLayer) -> None:
    job = _job()
    await persistence.create_job(job)

    job.status = JobStatus.FAILED
    job.error_message = "GitHub API timeout"
    await persistence.update_job(job)

    fetched = await persistence.get_job(job.job_id)
    assert fetched is not None
    assert fetched.status == JobStatus.FAILED
    assert fetched.error_message == "GitHub API timeout"


async def test_get_nonexistent_returns_none(persistence: PersistenceLayer) -> None:
    assert await persistence.get_job("no-such-id") is None


async def test_update_nonexistent_raises(persistence: PersistenceLayer) -> None:
    job = _job()
    with pytest.raises(ValueError, match="not found"):
        await persistence.update_job(job)
