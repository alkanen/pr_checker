import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

_CHUNK_NS = uuid.UUID("a3b4c5d6-e7f8-1234-abcd-1234567890ab")


class JobStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReviewTrigger(str, Enum):
    OPENED = "opened"
    UPDATED = "updated"
    REVIEW_REQUESTED = "review_requested"
    ON_DEMAND = "on_demand"


@dataclass
class DiffLine:
    kind: Literal["+", "-", " "]
    content: str
    old_lineno: int | None
    new_lineno: int | None


@dataclass
class DiffHunk:
    file_path: str
    header: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[DiffLine]


@dataclass
class LinkedIssue:
    number: int
    title: str
    body: str
    labels: list[str]
    assignees: list[str]


@dataclass
class StaticFinding:
    tool: str  # "ruff" | "mypy"
    file_path: str
    line: int | None
    col: int | None
    code: str
    message: str


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Finding:
    severity: Severity
    confidence: float  # [0.0, 1.0]
    category: str
    message: str
    suggestion: str | None = None
    file_path: str | None = None
    line_number: int | None = None  # new-file line number (right side)


@dataclass
class ReviewResult:
    findings: list[Finding]
    verdict: Literal["approve", "request_changes", "comment"]
    summary: str


@dataclass
class ProjectStandards:
    ruff: dict[str, Any] = field(default_factory=dict)
    mypy: dict[str, Any] = field(default_factory=dict)
    eslint: dict[str, Any] = field(default_factory=dict)
    prettier: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewContext:
    hunks: list[DiffHunk]
    standards: ProjectStandards
    linked_issues: list[LinkedIssue]
    static_findings: list[StaticFinding] = field(default_factory=list)


@dataclass
class CodeChunk:
    repo_full_name: str
    branch: str
    file_path: str
    chunk_type: Literal["function", "class", "module_block"]
    chunk_name: str
    content: str
    start_line: int
    end_line: int

    @property
    def point_id(self) -> str:
        # Stable identity based on location, not content: the same logical chunk
        # (same repo/branch/file/name/start-line) always has the same point_id so
        # that re-indexing an edited file upserts in-place rather than creating a
        # duplicate.  Orphan points from renamed/removed chunks are handled by
        # delete_file (incremental path) or delete_stale_generations (full re-index).
        key = (
            f"{self.repo_full_name}:{self.branch}:{self.file_path}"
            f":{self.chunk_name}:{self.start_line}"
        )
        return str(uuid.uuid5(_CHUNK_NS, key))


@dataclass
class IndexedBranch:
    repo_full_name: str
    branch: str
    last_indexed_sha: str
    indexed_at: datetime


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
    pr_body: str = ""
    status: JobStatus = JobStatus.PENDING
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    enqueued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    superseded: bool = False
