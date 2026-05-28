from __future__ import annotations

from dataclasses import dataclass, field

from pr_checker.models import Finding, ReviewResult, Severity
from pr_checker.reviewer_config import ReviewerConfig

_VERDICT_MAP = {
    "approve": "APPROVE",
    "request_changes": "REQUEST_CHANGES",
    "comment": "COMMENT",
}

_SEVERITY_EMOJI = {
    Severity.CRITICAL: "\U0001f534",
    Severity.HIGH: "\U0001f7e0",
    Severity.MEDIUM: "\U0001f7e1",
    Severity.LOW: "\U0001f535",
    Severity.INFO: "⚪",
}

_SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]


@dataclass
class InlineComment:
    path: str
    line: int
    body: str


@dataclass
class ReviewPayload:
    inline_comments: list[InlineComment] = field(default_factory=list)
    summary_comment: str | None = None
    review_event: str | None = None  # None means: do not submit a formal review
    review_body: str = ""


class ReviewFormatter:
    def format(self, result: ReviewResult, config: ReviewerConfig) -> ReviewPayload:
        payload = ReviewPayload()

        if config.output.inline_comments:
            for finding in result.findings:
                if finding.file_path and finding.line_number is not None:
                    payload.inline_comments.append(
                        InlineComment(
                            path=finding.file_path,
                            line=finding.line_number,
                            body=_format_finding(finding),
                        )
                    )

        if config.output.summary_comment:
            payload.summary_comment = _build_summary(result)

        if config.output.formal_review:
            payload.review_event = _VERDICT_MAP.get(result.verdict, "COMMENT")
            payload.review_body = result.summary

        return payload


def _format_finding(finding: Finding) -> str:
    emoji = _SEVERITY_EMOJI.get(finding.severity, "")
    lines = [
        f"{emoji} **{finding.severity.value.upper()}** ({finding.category}): {finding.message}"
    ]
    if finding.suggestion:
        lines.append(f"\n> **Suggestion:** {finding.suggestion}")
    return "\n".join(lines)


def _build_summary(result: ReviewResult) -> str:
    lines = ["## PR Checker Review\n", result.summary]

    if result.findings:
        lines.append("\n### Findings\n")
        sorted_findings = sorted(
            result.findings,
            key=lambda f: (
                _SEVERITY_ORDER.index(f.severity.value)
                if f.severity.value in _SEVERITY_ORDER
                else len(_SEVERITY_ORDER)
            ),
        )
        for f in sorted_findings:
            emoji = _SEVERITY_EMOJI.get(f.severity, "")
            loc = f" ({f.file_path}:{f.line_number})" if f.file_path and f.line_number else ""
            lines.append(f"- {emoji} **{f.category}**{loc}: {f.message}")

    lines.append("\n---\n*Posted by pr-checker*")
    return "\n".join(lines)
