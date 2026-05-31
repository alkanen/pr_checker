import asyncio
import logging
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from typing import Any

from pr_checker.db import PersistenceLayer
from pr_checker.exceptions import StalePRError
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
        self._shutdown: bool = False
        # Serialises enqueue/cancel_pr so concurrent webhooks for the same PR
        # cannot interleave between the DB write and the _pending/_active_tasks update.
        self._enqueue_lock: asyncio.Lock = asyncio.Lock()
        # Tracks the running (task, head_sha) per (repo, pr_number) so new pushes can
        # cancel the task and log both SHAs for diagnosing cancellations.
        self._active_tasks: dict[tuple[str, int], tuple[asyncio.Task[None], str]] = {}
        # Tracks the most-recently queued (unstarted) job per (repo, pr_number)
        # so a subsequent push can mark it superseded before it dequeues.
        self._pending: dict[tuple[str, int], PRJob] = {}

    async def start(self) -> None:
        self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        self._shutdown = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=5.0)
            except asyncio.TimeoutError:
                _log.warning("Queue did not drain within timeout; forcing shutdown")
            active_tasks = [t for t, _ in self._active_tasks.values()]
            for task in active_tasks:
                task.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def enqueue(self, job: PRJob) -> None:
        key = (job.repo_full_name, job.pr_number)

        async with self._enqueue_lock:
            # Persist first so a DB failure leaves the previous job intact.
            await self._persistence.create_job(job)

            existing_pending = self._pending.get(key)
            if existing_pending is not None:
                existing_pending.superseded = True
                _log.info(
                    "Job %s superseded by %s for %s #%d",
                    existing_pending.job_id,
                    job.job_id,
                    job.repo_full_name,
                    job.pr_number,
                )
            self._pending[key] = job

            existing_entry = self._active_tasks.get(key)
            if existing_entry is not None:
                existing_task, existing_sha = existing_entry
                if not existing_task.done():
                    _log.info(
                        "Cancelling in-flight job for %s #%d (sha %.8s superseded by sha %.8s)",
                        job.repo_full_name,
                        job.pr_number,
                        existing_sha,
                        job.head_sha,
                    )
                    existing_task.cancel()

            await self._queue.put(job)

        token = set_job_id(job.job_id)
        _log.info("Job %s accepted: %s #%d", job.job_id, job.repo_full_name, job.pr_number)
        reset_job_id(token)

    async def cancel_pr(self, repo_full_name: str, pr_number: int) -> None:
        key = (repo_full_name, pr_number)

        async with self._enqueue_lock:
            existing_pending = self._pending.pop(key, None)
            if existing_pending is not None:
                existing_pending.superseded = True
                _log.info(
                    "Job %s superseded (PR %s #%d closed)",
                    existing_pending.job_id,
                    repo_full_name,
                    pr_number,
                )

            existing_entry = self._active_tasks.get(key)
            if existing_entry is not None:
                existing_task, _ = existing_entry
                if not existing_task.done():
                    _log.info(
                        "Cancelling in-flight job for %s #%d (PR closed)",
                        repo_full_name,
                        pr_number,
                    )
                    existing_task.cancel()

    async def join(self) -> None:
        await self._queue.join()

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            key = (job.repo_full_name, job.pr_number)
            try:
                task: asyncio.Task[None] | None = None
                async with self._enqueue_lock:
                    if not job.superseded:
                        if self._pending.get(key) is job:
                            self._pending.pop(key)
                        task = asyncio.create_task(self._process(job))
                        self._active_tasks[key] = (task, job.head_sha)

                if task is None:
                    _log.info(
                        "Skipping superseded job %s for %s #%d",
                        job.job_id,
                        job.repo_full_name,
                        job.pr_number,
                    )
                    job.status = JobStatus.CANCELLED
                    job.completed_at = datetime.now(timezone.utc)
                    try:
                        await self._persistence.update_job(job)
                    except Exception:  # noqa: BLE001
                        _log.warning(
                            "Failed to persist CANCELLED for superseded job %s",
                            job.job_id,
                            exc_info=True,
                        )
                else:
                    try:
                        await task
                    except asyncio.CancelledError:
                        if task.cancelled() and not self._shutdown:
                            pass  # job task cancelled by newer push; worker continues
                        else:
                            raise  # worker itself was cancelled (shutdown)
                    except Exception:  # noqa: BLE001
                        _log.exception(
                            "Unhandled error in job task %s; worker continues", job.job_id
                        )
                    finally:
                        self._active_tasks.pop(key, None)
            except asyncio.CancelledError:
                raise  # Worker shutdown
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
            current_sha = await self._github.get_pr_head_sha(job.repo_full_name, job.pr_number)
            if current_sha != job.head_sha:
                raise StalePRError(
                    f"PR #{job.pr_number} head moved from {job.head_sha[:8]} to {current_sha[:8]}"
                )
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
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now(timezone.utc)
            _log.info(
                "Job %s cancelled (superseded by newer push): %s #%d",
                job.job_id,
                job.repo_full_name,
                job.pr_number,
            )
            try:
                await self._persistence.update_job(job)
            except Exception:  # noqa: BLE001
                _log.warning(
                    "Failed to persist CANCELLED status for job %s", job.job_id, exc_info=True
                )
            raise
        except StalePRError as exc:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now(timezone.utc)
            _log.info(
                "Job %s cancelled (stale SHA): %s #%d — %s",
                job.job_id,
                job.repo_full_name,
                job.pr_number,
                exc,
            )
            try:
                await self._persistence.update_job(job)
            except Exception:  # noqa: BLE001
                _log.warning(
                    "Failed to persist CANCELLED status for stale job %s",
                    job.job_id,
                    exc_info=True,
                )
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
