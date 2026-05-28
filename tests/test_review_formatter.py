from pr_checker.models import Finding, ReviewResult, Severity
from pr_checker.review_formatter import ReviewFormatter
from pr_checker.reviewer_config import OutputConfig, ReviewerConfig

_formatter = ReviewFormatter()


def _finding(
    severity: Severity = Severity.HIGH,
    confidence: float = 0.9,
    category: str = "logic",
    message: str = "test message",
    suggestion: str | None = None,
    file_path: str | None = "src/main.py",
    line_number: int | None = 42,
) -> Finding:
    return Finding(
        severity=severity,
        confidence=confidence,
        category=category,
        message=message,
        suggestion=suggestion,
        file_path=file_path,
        line_number=line_number,
    )


def _result(
    findings: list[Finding] | None = None,
    verdict: str = "comment",
    summary: str = "Test summary",
) -> ReviewResult:
    return ReviewResult(
        findings=findings if findings is not None else [_finding()],
        verdict=verdict,  # type: ignore[arg-type]
        summary=summary,
    )


def _config(
    inline_comments: bool = True,
    summary_comment: bool = True,
    formal_review: bool = True,
) -> ReviewerConfig:
    return ReviewerConfig(
        output=OutputConfig(
            inline_comments=inline_comments,
            summary_comment=summary_comment,
            formal_review=formal_review,
        )
    )


# --- inline comments ---


def test_inline_comment_included_when_enabled() -> None:
    payload = _formatter.format(_result(), _config(inline_comments=True))
    assert len(payload.inline_comments) == 1
    assert payload.inline_comments[0].path == "src/main.py"
    assert payload.inline_comments[0].line == 42


def test_inline_comment_suppressed_when_disabled() -> None:
    payload = _formatter.format(_result(), _config(inline_comments=False))
    assert payload.inline_comments == []


def test_finding_without_file_excluded_from_inline() -> None:
    payload = _formatter.format(
        _result(findings=[_finding(file_path=None, line_number=None)]), _config()
    )
    assert payload.inline_comments == []


def test_finding_with_file_but_no_line_excluded_from_inline() -> None:
    payload = _formatter.format(
        _result(findings=[_finding(file_path="src/main.py", line_number=None)]), _config()
    )
    assert payload.inline_comments == []


def test_inline_comment_body_contains_severity_and_message() -> None:
    f = _finding(severity=Severity.HIGH, message="Dangerous query")
    payload = _formatter.format(_result(findings=[f]), _config())
    body = payload.inline_comments[0].body
    assert "HIGH" in body
    assert "Dangerous query" in body


def test_inline_comment_body_includes_suggestion_when_present() -> None:
    f = _finding(suggestion="Use parameterized queries")
    payload = _formatter.format(_result(findings=[f]), _config())
    body = payload.inline_comments[0].body
    assert "Use parameterized queries" in body


def test_multiple_findings_each_generate_inline_comment() -> None:
    findings = [
        _finding(file_path="a.py", line_number=1, message="msg1"),
        _finding(file_path="b.py", line_number=2, message="msg2"),
    ]
    payload = _formatter.format(_result(findings=findings), _config())
    assert len(payload.inline_comments) == 2
    paths = {c.path for c in payload.inline_comments}
    assert paths == {"a.py", "b.py"}


# --- summary comment ---


def test_summary_comment_included_when_enabled() -> None:
    payload = _formatter.format(_result(), _config(summary_comment=True))
    assert payload.summary_comment is not None
    assert "PR Checker" in payload.summary_comment


def test_summary_comment_suppressed_when_disabled() -> None:
    payload = _formatter.format(_result(), _config(summary_comment=False))
    assert payload.summary_comment is None


def test_summary_contains_finding_message() -> None:
    f = _finding(message="Dangerous SQL query")
    payload = _formatter.format(_result(findings=[f]), _config())
    assert payload.summary_comment is not None
    assert "Dangerous SQL query" in payload.summary_comment


def test_summary_findings_sorted_by_severity() -> None:
    findings = [
        _finding(severity=Severity.LOW, message="low"),
        _finding(severity=Severity.CRITICAL, message="critical"),
        _finding(severity=Severity.MEDIUM, message="medium"),
    ]
    payload = _formatter.format(_result(findings=findings), _config())
    assert payload.summary_comment is not None
    critical_pos = payload.summary_comment.index("critical")
    medium_pos = payload.summary_comment.index("medium")
    low_pos = payload.summary_comment.index("low")
    assert critical_pos < medium_pos < low_pos


# --- formal review ---


def test_verdict_approve_maps_to_APPROVE() -> None:
    payload = _formatter.format(_result(verdict="approve"), _config())
    assert payload.review_event == "APPROVE"


def test_verdict_request_changes_maps_to_REQUEST_CHANGES() -> None:
    payload = _formatter.format(_result(verdict="request_changes"), _config())
    assert payload.review_event == "REQUEST_CHANGES"


def test_verdict_comment_maps_to_COMMENT() -> None:
    payload = _formatter.format(_result(verdict="comment"), _config())
    assert payload.review_event == "COMMENT"


def test_formal_review_suppressed_sets_review_event_to_none() -> None:
    payload = _formatter.format(_result(), _config(formal_review=False))
    assert payload.review_event is None


def test_review_body_contains_summary() -> None:
    payload = _formatter.format(_result(summary="Detailed summary text"), _config())
    assert payload.review_body == "Detailed summary text"


def test_review_body_empty_when_formal_review_disabled() -> None:
    payload = _formatter.format(_result(summary="Some summary"), _config(formal_review=False))
    assert payload.review_body == ""


# --- flag independence ---


def test_disabling_inline_does_not_affect_summary_or_formal_review() -> None:
    payload = _formatter.format(
        _result(), _config(inline_comments=False, summary_comment=True, formal_review=True)
    )
    assert payload.inline_comments == []
    assert payload.summary_comment is not None
    assert payload.review_event is not None


def test_disabling_formal_review_does_not_affect_inline_or_summary() -> None:
    payload = _formatter.format(
        _result(), _config(inline_comments=True, summary_comment=True, formal_review=False)
    )
    assert len(payload.inline_comments) == 1
    assert payload.summary_comment is not None
    assert payload.review_event is None


def test_inline_comments_collected_even_when_formal_review_disabled() -> None:
    # Inline comments are non-empty so orchestrator can still deliver them
    findings = [_finding(file_path="x.py", line_number=5)]
    payload = _formatter.format(
        _result(findings=findings), _config(inline_comments=True, formal_review=False)
    )
    assert len(payload.inline_comments) == 1
    assert payload.review_event is None
