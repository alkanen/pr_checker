from unittest.mock import AsyncMock, MagicMock

from pytest_httpx import HTTPXMock

from pr_checker.github_client import GitHubClient
from pr_checker.issue_resolver import (
    IssueResolver,
    _extract_closing_keywords,
    _extract_from_branch,
    _extract_issue_urls,
)
from pr_checker.models import LinkedIssue

_ISSUE_42 = LinkedIssue(number=42, title="Fix bug", body="Bug body", labels=["bug"], assignees=[])

_ISSUE_42_JSON = {
    "number": 42,
    "title": "Fix bug",
    "body": "Bug body",
    "labels": [{"name": "bug"}],
    "assignees": [],
}


# --- _extract_closing_keywords ---


def test_closing_keywords_closes() -> None:
    assert _extract_closing_keywords("closes #42") == [42]


def test_closing_keywords_fixes() -> None:
    assert _extract_closing_keywords("Fixes #10 and fixes #11") == [10, 11]


def test_closing_keywords_resolves() -> None:
    assert _extract_closing_keywords("This resolves #7.") == [7]


def test_closing_keywords_case_insensitive() -> None:
    assert _extract_closing_keywords("CLOSE #99") == [99]


def test_closing_keywords_no_match() -> None:
    assert _extract_closing_keywords("No issues here") == []


def test_closing_keywords_malformed_non_digit() -> None:
    assert _extract_closing_keywords("closes #abc") == []


# --- _extract_issue_urls ---


def test_issue_url_match() -> None:
    assert _extract_issue_urls("https://github.com/owner/repo/issues/42") == [42]


def test_issue_url_multiple() -> None:
    body = "See https://github.com/a/b/issues/1 and https://github.com/a/b/issues/2"
    assert _extract_issue_urls(body) == [1, 2]


def test_issue_url_no_match() -> None:
    assert _extract_issue_urls("no links here") == []


def test_issue_url_pr_link_not_matched() -> None:
    assert _extract_issue_urls("https://github.com/owner/repo/pulls/42") == []


# --- _extract_from_branch ---


def test_branch_feat_number_slug() -> None:
    assert _extract_from_branch("feature/42-add-login") == [42]


def test_branch_issue_prefix() -> None:
    assert _extract_from_branch("issue-7") == [7]


def test_branch_bugfix_number() -> None:
    assert _extract_from_branch("bugfix/99") == [99]


def test_branch_feat_issue_prefix() -> None:
    assert _extract_from_branch("feat/issue-3-pipe-skeleton") == [3]


def test_branch_no_number() -> None:
    assert _extract_from_branch("main") == []


def test_branch_named_without_number() -> None:
    assert _extract_from_branch("feat/add-login") == []


def test_branch_date_like_not_matched() -> None:
    assert _extract_from_branch("release/2024-05") == []


def test_branch_version_tag_not_matched() -> None:
    assert _extract_from_branch("release/v2-stable") == []


# --- IssueResolver.resolve ---


async def test_resolve_closing_keyword(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/issues/42",
        json=_ISSUE_42_JSON,
    )
    github = GitHubClient(token="test-token")
    try:
        resolver = IssueResolver(github)
        issues = await resolver.resolve("owner/repo", "title", "closes #42", "feat/x")
    finally:
        await github.aclose()

    assert issues == [_ISSUE_42]


async def test_resolve_issue_url_fallback(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/issues/42",
        json=_ISSUE_42_JSON,
    )
    github = GitHubClient(token="test-token")
    try:
        resolver = IssueResolver(github)
        issues = await resolver.resolve(
            "owner/repo",
            "title",
            "See https://github.com/owner/repo/issues/42",
            "feat/x",
        )
    finally:
        await github.aclose()

    assert issues == [_ISSUE_42]


async def test_resolve_branch_name_fallback(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/issues/42",
        json=_ISSUE_42_JSON,
    )
    github = GitHubClient(token="test-token")
    try:
        resolver = IssueResolver(github)
        issues = await resolver.resolve("owner/repo", "title", "", "feature/42-add-login")
    finally:
        await github.aclose()

    assert issues == [_ISSUE_42]


async def test_resolve_llm_fallback() -> None:
    mock_llm = MagicMock()
    mock_llm.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="42"))])
    )
    mock_github = MagicMock(spec=GitHubClient)
    mock_github.get_issue = AsyncMock(return_value=_ISSUE_42)

    resolver = IssueResolver(mock_github, llm=mock_llm)
    issues = await resolver.resolve("owner/repo", "Fix the login bug", "", "feat/cleanup")

    assert issues == [_ISSUE_42]
    mock_llm.chat.completions.create.assert_called_once()


async def test_resolve_llm_error_returns_empty_list() -> None:
    mock_llm = MagicMock()
    mock_llm.chat.completions.create = AsyncMock(side_effect=RuntimeError("network error"))
    mock_github = MagicMock(spec=GitHubClient)

    resolver = IssueResolver(mock_github, llm=mock_llm)
    issues = await resolver.resolve("owner/repo", "Some PR", "", "feat/no-number")

    assert issues == []
    mock_github.get_issue.assert_not_called()


async def test_resolve_no_links_no_llm() -> None:
    mock_github = MagicMock(spec=GitHubClient)
    resolver = IssueResolver(mock_github)
    issues = await resolver.resolve("owner/repo", "Refactor something", "", "feat/cleanup")

    assert issues == []


async def test_resolve_deduplicates_same_issue(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/issues/42",
        json=_ISSUE_42_JSON,
    )
    github = GitHubClient(token="test-token")
    try:
        resolver = IssueResolver(github)
        # Closing keyword takes priority; duplicate number in branch is ignored
        issues = await resolver.resolve("owner/repo", "title", "closes #42", "feature/42-slug")
    finally:
        await github.aclose()

    assert len(issues) == 1
    assert issues[0].number == 42


async def test_resolve_skips_failed_fetch(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/issues/999",
        status_code=404,
    )
    github = GitHubClient(token="test-token")
    try:
        resolver = IssueResolver(github)
        issues = await resolver.resolve("owner/repo", "title", "closes #999", "branch")
    finally:
        await github.aclose()

    assert issues == []


async def test_resolve_closing_keyword_takes_priority_over_url(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/issues/10",
        json={
            "number": 10,
            "title": "Keyword issue",
            "body": "",
            "labels": [],
            "assignees": [],
        },
    )
    github = GitHubClient(token="test-token")
    try:
        resolver = IssueResolver(github)
        # Both a closing keyword (#10) and a URL (issue 20) are present.
        # Closing keyword wins; URL is never fetched.
        issues = await resolver.resolve(
            "owner/repo",
            "title",
            "closes #10\nhttps://github.com/owner/repo/issues/20",
            "branch",
        )
    finally:
        await github.aclose()

    assert len(issues) == 1
    assert issues[0].number == 10
