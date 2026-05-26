import base64
from collections.abc import AsyncGenerator

import pytest
from pytest_httpx import HTTPXMock

from pr_checker.github_client import GitHubClient
from pr_checker.models import LinkedIssue


@pytest.fixture
async def github() -> AsyncGenerator[GitHubClient, None]:
    client = GitHubClient(token="test-token")
    yield client
    await client.aclose()


def _file_entry(filename: str, patch: str | None = None) -> dict[str, object]:
    entry: dict[str, object] = {"filename": filename}
    if patch is not None:
        entry["patch"] = patch
    return entry


# --- get_pr_diff ---


async def test_get_pr_diff_parses_hunk(github: GitHubClient, httpx_mock: HTTPXMock) -> None:
    patch = "@@ -1,3 +1,4 @@\n context\n+added line\n-removed line\n context2"
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/pulls/1/files?per_page=100&page=1",
        json=[_file_entry("foo.py", patch)],
    )
    hunks = await github.get_pr_diff("owner/repo", 1)

    assert len(hunks) == 1
    h = hunks[0]
    assert h.file_path == "foo.py"
    assert h.old_start == 1
    assert h.old_count == 3
    assert h.new_start == 1
    assert h.new_count == 4
    assert len(h.lines) == 4
    assert h.lines[0].kind == " "
    assert h.lines[0].content == "context"
    assert h.lines[1].kind == "+"
    assert h.lines[1].new_lineno == 2
    assert h.lines[1].old_lineno is None
    assert h.lines[2].kind == "-"
    assert h.lines[2].old_lineno == 2
    assert h.lines[2].new_lineno is None


async def test_get_pr_diff_skips_binary_files(github: GitHubClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/pulls/1/files?per_page=100&page=1",
        json=[
            _file_entry("image.png"),
            _file_entry("code.py", "@@ -1 +1 @@\n-old\n+new"),
        ],
    )
    hunks = await github.get_pr_diff("owner/repo", 1)

    assert len(hunks) == 1
    assert hunks[0].file_path == "code.py"


async def test_get_pr_diff_multiple_hunks(github: GitHubClient, httpx_mock: HTTPXMock) -> None:
    patch = "@@ -1,2 +1,2 @@\n ctx\n-old1\n+new1\n@@ -10,2 +10,2 @@\n ctx\n-old2\n+new2"
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/pulls/5/files?per_page=100&page=1",
        json=[_file_entry("multi.py", patch)],
    )
    hunks = await github.get_pr_diff("owner/repo", 5)

    assert len(hunks) == 2
    assert hunks[0].old_start == 1
    assert hunks[1].old_start == 10


async def test_get_pr_diff_paginates_all_pages(github: GitHubClient, httpx_mock: HTTPXMock) -> None:
    simple_patch = "@@ -1 +1 @@\n-old\n+new"
    page1 = [_file_entry(f"file_{i}.py", simple_patch) for i in range(100)]
    page2 = [_file_entry(f"file_{i}.py", simple_patch) for i in range(100, 145)]
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/pulls/2/files?per_page=100&page=1",
        json=page1,
    )
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/pulls/2/files?per_page=100&page=2",
        json=page2,
    )
    hunks = await github.get_pr_diff("owner/repo", 2)

    assert len(hunks) == 145
    assert hunks[0].file_path == "file_0.py"
    assert hunks[144].file_path == "file_144.py"


async def test_get_pr_diff_truncates_at_max_files(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    simple_patch = "@@ -1 +1 @@\n-old\n+new"
    page1 = [_file_entry(f"file_{i}.py", simple_patch) for i in range(100)]
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/pulls/3/files?per_page=100&page=1",
        json=page1,
    )
    hunks = await github.get_pr_diff("owner/repo", 3, max_files=50)

    assert len(hunks) == 50
    assert hunks[49].file_path == "file_49.py"


# --- get_file_content ---


async def test_get_file_content_decodes_base64(github: GitHubClient, httpx_mock: HTTPXMock) -> None:
    raw = "print('hello')\n"
    encoded = base64.b64encode(raw.encode()).decode() + "\n"
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/contents/foo.py?ref=abc123",
        json={"size": len(raw), "content": encoded, "encoding": "base64"},
    )
    result = await github.get_file_content("owner/repo", "foo.py", "abc123")

    assert result == raw


async def test_get_file_content_above_threshold_returns_none(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/contents/big.py?ref=sha1",
        json={"size": 200 * 1024, "content": "", "encoding": "base64"},
    )
    result = await github.get_file_content(
        "owner/repo", "big.py", "sha1", size_threshold=100 * 1024
    )

    assert result is None


async def test_get_file_content_custom_threshold(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    raw = "x = 1\n"
    encoded = base64.b64encode(raw.encode()).decode()
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/contents/small.py?ref=sha1",
        json={"size": len(raw), "content": encoded, "encoding": "base64"},
    )
    result = await github.get_file_content("owner/repo", "small.py", "sha1", size_threshold=5)

    assert result is None


# --- get_issue ---


async def test_get_issue_returns_linked_issue(github: GitHubClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/issues/42",
        json={
            "number": 42,
            "title": "Fix the bug",
            "body": "Detailed description",
            "labels": [{"name": "bug"}, {"name": "urgent"}],
            "assignees": [{"login": "alice"}, {"login": "bob"}],
        },
    )
    issue = await github.get_issue("owner/repo", 42)

    assert issue == LinkedIssue(
        number=42,
        title="Fix the bug",
        body="Detailed description",
        labels=["bug", "urgent"],
        assignees=["alice", "bob"],
    )


async def test_get_issue_null_body_becomes_empty_string(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/issues/7",
        json={"number": 7, "title": "Minimal", "body": None, "labels": [], "assignees": []},
    )
    issue = await github.get_issue("owner/repo", 7)

    assert issue.body == ""
