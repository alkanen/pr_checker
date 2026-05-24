import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ReviewTrigger(str, Enum):
    OPENED = "opened"
    UPDATED = "updated"
    REVIEW_REQUESTED = "review_requested"
    ON_DEMAND = "on_demand"


@dataclass
class PRJob:
    repo_full_name: str
    pr_number: int
    pr_title: str
    head_sha: str
    base_sha: str
    head_branch: str
    base_branch: str
    trigger: ReviewTrigger
    status: JobStatus = JobStatus.PENDING
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    enqueued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
