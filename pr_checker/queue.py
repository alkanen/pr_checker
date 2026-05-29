import asyncio
import logging
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from typing import Any

from pr_checker.db import PersistenceLayer
from pr_checker.github_client import GitHubClient
from pr_checker.logging_setup import reset_job_id, set_job_id
from pr_checker.models import JobStatus, PRJob
from pr_checker.review_orchestrator import ReviewOrchestrator

_log = logging.getLogger(__name__)


class ReviewQueue:
    def __init__(
        self,
        persistence: PersistenceLayer,
        github: GitHubClient,
        orchestrator: ReviewOrchestrator,
    ) -> None:
        self._persistence = persistence
        self._github = github
        self._orchestrator = orchestrator
        self._queue: asyncio.Queue[PRJob] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._task is not None:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=5.0)
            except asyncio.TimeoutError:
                _log.warning("Queue did not drain within timeout; forcing shutdown")
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def enqueue(self, job: PRJob) -> None:
        await self._persistence.create_job(job)
        await self._queue.put(job)
        token = set_job_id(job.job_id)
        _log.info("Job %s accepted: %s #%d", job.job_id, job.repo_full_name, job.pr_number)
        reset_job_id(token)

    async def join(self) -> None:
        await self._queue.join()

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self._process(job)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                _log.exception("Unhandled error processing job %s; worker continues", job.job_id)
            finally:
                self._queue.task_done()

    async def _process(self, job: PRJob) -> None:
        token = set_job_id(job.job_id)
        try:
            await self._run_job(job)
        finally:
            reset_job_id(token)

    async def _run_job(self, job: PRJob) -> None:
        job.status = JobStatus.IN_PROGRESS
        job.started_at = datetime.now(timezone.utc)
        _log.info("Job %s started: %s #%d", job.job_id, job.repo_full_name, job.pr_number)

        try:
            await self._persistence.update_job(job)
            await self._github.post_commit_status(
                job.repo_full_name, job.head_sha, "pending", "PR review in progress"
            )
            await self._orchestrator.run(job)
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            elapsed = (job.completed_at - job.started_at).total_seconds()
            await self._persistence.update_job(job)
            await self._github.post_commit_status(
                job.repo_full_name, job.head_sha, "success", "PR review complete"
            )
            _log.info(
                "Job %s completed in %.1fs: %s #%d",
                job.job_id,
                elapsed,
                job.repo_full_name,
                job.pr_number,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error_message = str(exc)
            _log.error(
                "Job %s failed: %s #%d",
                job.job_id,
                job.repo_full_name,
                job.pr_number,
                exc_info=True,
            )
            _error_msg = (
                f"## PR Checker Error\n\nThe automated review failed (job `{job.job_id}`)."
                " Please check the server logs for details.\n\n---\n*Posted by pr-checker*"
            )
            _log.info(
                "Posting error comment for %s #%d: %s",
                job.repo_full_name,
                job.pr_number,
                _error_msg[:60].replace("\n", " "),
            )
            _steps: list[tuple[str, Callable[[], Coroutine[Any, Any, None]]]] = [
                ("db_update", lambda: self._persistence.update_job(job)),
                (
                    "commit_status",
                    lambda: self._github.post_commit_status(
                        job.repo_full_name, job.head_sha, "error", "PR review failed"
                    ),
                ),
                (
                    "error_comment",
                    lambda: self._github.post_pr_comment(
                        job.repo_full_name, job.pr_number, _error_msg
                    ),
                ),
            ]
            for step, fn in _steps:
                try:
                    await fn()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    _log.error(
                        "Best-effort error notification failed [%s] for job %s",
                        step,
                        job.job_id,
                        exc_info=True,
                    )
