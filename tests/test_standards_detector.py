import base64
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from pr_checker.github_client import GitHubClient
from pr_checker.models import ProjectStandards
from pr_checker.standards_detector import StandardsDetector

# The detector stops early when it finds ruff/mypy/eslint/prettier config, so some
# 404 mocks registered by _mock_all_absent will intentionally go unused.
pytestmark = pytest.mark.httpx_mock(assert_all_responses_were_requested=False)

FIXTURES = Path(__file__).parent / "fixtures" / "standards"

_ESLINT_PATHS = [".eslintrc", ".eslintrc.json", ".eslintrc.yaml", ".eslintrc.yml"]
_PRETTIER_PATHS = [".prettierrc", ".prettierrc.json", ".prettierrc.yaml", ".prettierrc.yml"]


@pytest.fixture
async def github() -> AsyncGenerator[GitHubClient, None]:
    client = GitHubClient(token="test-token")
    yield client
    await client.aclose()


def _encode(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def _file_response(content: str) -> dict[str, object]:
    return {"size": len(content), "content": _encode(content), "encoding": "base64"}


def _mock_file(httpx_mock: HTTPXMock, repo: str, path: str, ref: str, content: str) -> None:
    httpx_mock.add_response(
        url=f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}",
        json=_file_response(content),
    )


def _mock_absent(httpx_mock: HTTPXMock, repo: str, path: str, ref: str) -> None:
    httpx_mock.add_response(
        url=f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}",
        status_code=404,
        json={"message": "Not Found"},
    )


def _mock_all_absent(
    httpx_mock: HTTPXMock,
    repo: str,
    ref: str,
    skip: list[str] | None = None,
) -> None:
    skip_set = set(skip or [])
    for path in ["pyproject.toml", "ruff.toml", "mypy.ini", *_ESLINT_PATHS, *_PRETTIER_PATHS]:
        if path not in skip_set:
            _mock_absent(httpx_mock, repo, path, ref)


# --- empty standards ---


async def test_empty_standards_when_no_files_found(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    _mock_all_absent(httpx_mock, "owner/repo", "main")
    detector = StandardsDetector(github)
    standards = await detector.detect("owner/repo", "main")
    assert standards == ProjectStandards()


# --- pyproject.toml ---


async def test_detects_ruff_and_mypy_from_pyproject(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    content = (FIXTURES / "pyproject.toml").read_text()
    _mock_file(httpx_mock, "owner/repo", "pyproject.toml", "main", content)
    _mock_all_absent(httpx_mock, "owner/repo", "main", skip=["pyproject.toml"])

    detector = StandardsDetector(github)
    standards = await detector.detect("owner/repo", "main")

    assert standards.ruff["line-length"] == 88
    assert standards.ruff["target-version"] == "py311"
    assert standards.ruff["lint"]["select"] == ["E", "F", "I"]
    assert standards.mypy["strict"] is True
    assert standards.mypy["python_version"] == "3.11"
    assert standards.eslint == {}
    assert standards.prettier == {}


# --- ruff.toml fallback ---


async def test_detects_ruff_from_standalone_ruff_toml(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    content = (FIXTURES / "ruff.toml").read_text()
    _mock_absent(httpx_mock, "owner/repo", "pyproject.toml", "main")
    _mock_file(httpx_mock, "owner/repo", "ruff.toml", "main", content)
    _mock_all_absent(httpx_mock, "owner/repo", "main", skip=["pyproject.toml", "ruff.toml"])

    detector = StandardsDetector(github)
    standards = await detector.detect("owner/repo", "main")

    assert standards.ruff["line-length"] == 120
    assert standards.ruff["lint"]["select"] == ["E", "F", "I", "B"]
    assert standards.mypy == {}


# --- mypy.ini fallback ---


async def test_detects_mypy_from_ini(github: GitHubClient, httpx_mock: HTTPXMock) -> None:
    content = (FIXTURES / "mypy.ini").read_text()
    _mock_absent(httpx_mock, "owner/repo", "pyproject.toml", "main")
    _mock_absent(httpx_mock, "owner/repo", "ruff.toml", "main")
    _mock_file(httpx_mock, "owner/repo", "mypy.ini", "main", content)
    _mock_all_absent(
        httpx_mock, "owner/repo", "main", skip=["pyproject.toml", "ruff.toml", "mypy.ini"]
    )

    detector = StandardsDetector(github)
    standards = await detector.detect("owner/repo", "main")

    assert standards.mypy["python_version"] == "3.10"
    assert standards.mypy["strict"] is True
    assert standards.mypy["disallow_untyped_calls"] is True
    assert standards.ruff == {}


# --- eslint ---


async def test_detects_eslint_from_json(github: GitHubClient, httpx_mock: HTTPXMock) -> None:
    content = (FIXTURES / "eslintrc.json").read_text()
    _mock_absent(httpx_mock, "owner/repo", "pyproject.toml", "main")
    _mock_absent(httpx_mock, "owner/repo", "ruff.toml", "main")
    _mock_absent(httpx_mock, "owner/repo", "mypy.ini", "main")
    _mock_absent(httpx_mock, "owner/repo", ".eslintrc", "main")
    _mock_file(httpx_mock, "owner/repo", ".eslintrc.json", "main", content)
    _mock_all_absent(
        httpx_mock,
        "owner/repo",
        "main",
        skip=["pyproject.toml", "ruff.toml", "mypy.ini", ".eslintrc", ".eslintrc.json"],
    )

    detector = StandardsDetector(github)
    standards = await detector.detect("owner/repo", "main")

    assert standards.eslint["rules"]["no-console"] == "warn"
    assert standards.prettier == {}


# --- prettier ---


async def test_detects_prettier_from_json(github: GitHubClient, httpx_mock: HTTPXMock) -> None:
    content = (FIXTURES / "prettierrc.json").read_text()
    _mock_absent(httpx_mock, "owner/repo", "pyproject.toml", "main")
    _mock_absent(httpx_mock, "owner/repo", "ruff.toml", "main")
    _mock_absent(httpx_mock, "owner/repo", "mypy.ini", "main")
    _mock_all_absent(
        httpx_mock,
        "owner/repo",
        "main",
        skip=["pyproject.toml", "ruff.toml", "mypy.ini", *_PRETTIER_PATHS],
    )
    _mock_absent(httpx_mock, "owner/repo", ".prettierrc", "main")
    _mock_file(httpx_mock, "owner/repo", ".prettierrc.json", "main", content)
    _mock_all_absent(
        httpx_mock,
        "owner/repo",
        "main",
        skip=[
            "pyproject.toml",
            "ruff.toml",
            "mypy.ini",
            *_ESLINT_PATHS,
            ".prettierrc",
            ".prettierrc.json",
        ],
    )

    detector = StandardsDetector(github)
    standards = await detector.detect("owner/repo", "main")

    assert standards.prettier["semi"] is False
    assert standards.prettier["singleQuote"] is True
    assert standards.eslint == {}


# --- ruff.toml skipped when pyproject has ruff ---


async def test_ruff_toml_not_fetched_when_pyproject_has_ruff(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    """ruff.toml must not be requested if pyproject.toml already provided ruff config."""
    content = (FIXTURES / "pyproject.toml").read_text()
    _mock_file(httpx_mock, "owner/repo", "pyproject.toml", "main", content)
    # mypy.ini + eslint + prettier absent (ruff.toml intentionally not mocked)
    _mock_absent(httpx_mock, "owner/repo", "mypy.ini", "main")
    for path in [*_ESLINT_PATHS, *_PRETTIER_PATHS]:
        _mock_absent(httpx_mock, "owner/repo", path, "main")

    detector = StandardsDetector(github)
    standards = await detector.detect("owner/repo", "main")

    assert standards.ruff["line-length"] == 88


# --- extensionless YAML configs ---


async def test_detects_eslint_from_extensionless_yaml(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    content = (FIXTURES / "eslint_yaml.yml").read_text()
    _mock_all_absent(httpx_mock, "owner/repo", "main", skip=[".eslintrc"])
    _mock_file(httpx_mock, "owner/repo", ".eslintrc", "main", content)

    detector = StandardsDetector(github)
    standards = await detector.detect("owner/repo", "main")

    assert standards.eslint["rules"]["no-console"] == "error"
    assert standards.eslint["env"]["node"] is True


async def test_detects_prettier_from_extensionless_yaml(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    content = (FIXTURES / "prettier_yaml.yml").read_text()
    _mock_all_absent(httpx_mock, "owner/repo", "main", skip=[".prettierrc"])
    _mock_file(httpx_mock, "owner/repo", ".prettierrc", "main", content)

    detector = StandardsDetector(github)
    standards = await detector.detect("owner/repo", "main")

    assert standards.prettier["semi"] is True
    assert standards.prettier["tabWidth"] == 4
