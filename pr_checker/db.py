from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.engine import Dialect
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from pr_checker.models import JobStatus, PRJob, ReviewTrigger


class UTCDateTime(TypeDecorator[datetime]):
    """Persists datetimes as naive UTC; restores UTC tzinfo on read."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is not None and value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class Base(DeclarativeBase):
    pass


class JobRow(Base):
    __tablename__ = "review_jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    repo_full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    pr_number: Mapped[int] = mapped_column(Integer, nullable=False)
    pr_title: Mapped[str] = mapped_column(Text, nullable=False)
    head_sha: Mapped[str] = mapped_column(String(40), nullable=False)
    base_sha: Mapped[str] = mapped_column(String(40), nullable=False)
    head_branch: Mapped[str] = mapped_column(String(255), nullable=False)
    base_branch: Mapped[str] = mapped_column(String(255), nullable=False)
    trigger: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    enqueued_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


def _row_to_job(row: JobRow) -> PRJob:
    return PRJob(
        job_id=row.job_id,
        repo_full_name=row.repo_full_name,
        pr_number=row.pr_number,
        pr_title=row.pr_title,
        head_sha=row.head_sha,
        base_sha=row.base_sha,
        head_branch=row.head_branch,
        base_branch=row.base_branch,
        trigger=ReviewTrigger(row.trigger),
        status=JobStatus(row.status),
        enqueued_at=row.enqueued_at,
        started_at=row.started_at,
        completed_at=row.completed_at,
        error_message=row.error_message,
    )


class PersistenceLayer:
    def __init__(self, database_url: str) -> None:
        self._engine: AsyncEngine = create_async_engine(database_url, echo=False)
        self._sessions: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self._engine.dispose()

    async def create_job(self, job: PRJob) -> None:
        async with self._sessions() as session:
            row = JobRow(
                job_id=job.job_id,
                repo_full_name=job.repo_full_name,
                pr_number=job.pr_number,
                pr_title=job.pr_title,
                head_sha=job.head_sha,
                base_sha=job.base_sha,
                head_branch=job.head_branch,
                base_branch=job.base_branch,
                trigger=job.trigger.value,
                status=job.status.value,
                enqueued_at=job.enqueued_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                error_message=job.error_message,
            )
            session.add(row)
            await session.commit()

    async def update_job(self, job: PRJob) -> None:
        async with self._sessions() as session:
            row = await session.get(JobRow, job.job_id)
            if row is None:
                raise ValueError(f"Job {job.job_id} not found")
            row.status = job.status.value
            row.started_at = job.started_at
            row.completed_at = job.completed_at
            row.error_message = job.error_message
            await session.commit()

    async def get_job(self, job_id: str) -> PRJob | None:
        async with self._sessions() as session:
            row = await session.get(JobRow, job_id)
            return _row_to_job(row) if row is not None else None
