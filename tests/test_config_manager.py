import base64
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from pr_checker.github_client import GitHubClient
from pr_checker.reviewer_config import (
    ConfigManager,
    ReviewerConfig,
)

FIXTURES = Path(__file__).parent / "fixtures" / "config"


@pytest.fixture
async def github() -> AsyncGenerator[GitHubClient, None]:
    client = GitHubClient(token="test-token")
    yield client
    await client.aclose()


def _encode(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def _file_response(content: str) -> dict[str, object]:
    return {"size": len(content), "content": _encode(content), "encoding": "base64"}


def _mock_repo(httpx_mock: HTTPXMock, repo: str = "owner/repo", branch: str = "main") -> None:
    httpx_mock.add_response(
        url=f"https://api.github.com/repos/{repo}",
        json={"default_branch": branch},
    )


def _mock_config(
    httpx_mock: HTTPXMock,
    content: str,
    repo: str = "owner/repo",
    branch: str = "main",
) -> None:
    httpx_mock.add_response(
        url=f"https://api.github.com/repos/{repo}/contents/.pr-checker.yml?ref={branch}",
        json=_file_response(content),
    )


def _mock_config_absent(
    httpx_mock: HTTPXMock,
    repo: str = "owner/repo",
    branch: str = "main",
) -> None:
    httpx_mock.add_response(
        url=f"https://api.github.com/repos/{repo}/contents/.pr-checker.yml?ref={branch}",
        status_code=404,
        json={"message": "Not Found"},
    )


# --- startup ---


def test_missing_config_file_uses_compiled_defaults() -> None:
    manager = ConfigManager(config_file=Path("/nonexistent/config.yml"))
    assert manager.server_config == ReviewerConfig()


def test_no_config_file_arg_uses_compiled_defaults() -> None:
    assert ConfigManager().server_config == ReviewerConfig()


def test_load_server_config_from_file() -> None:
    manager = ConfigManager(config_file=FIXTURES / "server_full.yml")
    cfg = manager.server_config
    assert cfg.vram_budget_gb == 16.0
    assert cfg.models.default == "gpt-4o-mini"
    assert cfg.models.tasks.code_review == "gpt-4o-mini"
    assert cfg.checks.ignore_patterns == ["tests/**"]
    assert cfg.output.formal_review is True


# --- for_repo: fallback ---


async def test_for_repo_absent_config_returns_server_defaults(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    _mock_repo(httpx_mock)
    _mock_config_absent(httpx_mock)
    result = await ConfigManager().for_repo("owner/repo", github)
    assert result == ReviewerConfig()


# --- for_repo: deep merge ---


async def test_deep_merge_nested_override(github: GitHubClient, httpx_mock: HTTPXMock) -> None:
    manager = ConfigManager(config_file=FIXTURES / "server_full.yml")
    _mock_repo(httpx_mock)
    _mock_config(httpx_mock, (FIXTURES / "repo_partial.yml").read_text())

    result = await manager.for_repo("owner/repo", github)

    # nested override: only code_review changed
    assert result.models.tasks.code_review == "claude-sonnet-4-6"
    assert result.models.tasks.synthesis == "gpt-4o-mini"
    assert result.models.default == "gpt-4o-mini"


async def test_deep_merge_missing_key_keeps_server_default(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    manager = ConfigManager(config_file=FIXTURES / "server_full.yml")
    _mock_repo(httpx_mock)
    _mock_config(httpx_mock, (FIXTURES / "repo_partial.yml").read_text())

    result = await manager.for_repo("owner/repo", github)

    assert result.vram_budget_gb == 16.0
    assert result.embedding.model == "text-embedding-3-small"
    assert result.qdrant.url == "http://localhost:6333"


async def test_deep_merge_list_replaced_not_appended(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    manager = ConfigManager(config_file=FIXTURES / "server_full.yml")
    _mock_repo(httpx_mock)
    _mock_config(httpx_mock, (FIXTURES / "repo_partial.yml").read_text())

    result = await manager.for_repo("owner/repo", github)

    assert result.checks.ignore_patterns == ["vendor/**"]


async def test_deep_merge_bool_override(github: GitHubClient, httpx_mock: HTTPXMock) -> None:
    manager = ConfigManager(config_file=FIXTURES / "server_full.yml")
    _mock_repo(httpx_mock)
    _mock_config(httpx_mock, (FIXTURES / "repo_partial.yml").read_text())

    result = await manager.for_repo("owner/repo", github)

    assert result.output.formal_review is False
    assert result.output.inline_comments is True
    assert result.output.summary_comment is True


async def test_deep_merge_null_value_treated_as_no_override(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    manager = ConfigManager(config_file=FIXTURES / "server_full.yml")
    _mock_repo(httpx_mock)
    _mock_config(httpx_mock, "vram_budget_gb: null\nchecks: null\n")

    result = await manager.for_repo("owner/repo", github)

    assert result.vram_budget_gb == 16.0
    assert result.checks == manager.server_config.checks


# --- for_repo: permission limits ---


async def test_repo_cannot_enable_server_disabled_check(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    manager = ConfigManager(config_file=FIXTURES / "security_disabled.yml")
    _mock_repo(httpx_mock)
    _mock_config(httpx_mock, "checks:\n  security: true\n")

    result = await manager.for_repo("owner/repo", github)

    assert result.checks.security is False


async def test_repo_cannot_exceed_server_vram_budget(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    manager = ConfigManager(config_file=FIXTURES / "server_full.yml")
    _mock_repo(httpx_mock)
    _mock_config(httpx_mock, "vram_budget_gb: 999.0\n")

    result = await manager.for_repo("owner/repo", github)

    assert result.vram_budget_gb == 16.0


async def test_repo_can_lower_vram_budget(github: GitHubClient, httpx_mock: HTTPXMock) -> None:
    manager = ConfigManager(config_file=FIXTURES / "server_full.yml")
    _mock_repo(httpx_mock)
    _mock_config(httpx_mock, "vram_budget_gb: 8.0\n")

    result = await manager.for_repo("owner/repo", github)

    assert result.vram_budget_gb == 8.0


async def test_repo_can_disable_output_type(github: GitHubClient, httpx_mock: HTTPXMock) -> None:
    manager = ConfigManager(config_file=FIXTURES / "server_full.yml")
    _mock_repo(httpx_mock)
    _mock_config(httpx_mock, "output:\n  inline_comments: false\n")

    result = await manager.for_repo("owner/repo", github)

    assert result.output.inline_comments is False
    assert result.output.summary_comment is True


async def test_repo_cannot_enable_server_disabled_output(
    tmp_path: Path, github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    cfg = tmp_path / "server.yml"
    cfg.write_text("output:\n  formal_review: false\n")

    manager = ConfigManager(config_file=cfg)
    _mock_repo(httpx_mock)
    _mock_config(httpx_mock, "output:\n  formal_review: true\n")

    result = await manager.for_repo("owner/repo", github)

    assert result.output.formal_review is False


async def test_repo_cannot_lower_bytes_per_param(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    manager = ConfigManager(config_file=FIXTURES / "server_full.yml")
    _mock_repo(httpx_mock)
    _mock_config(httpx_mock, "bytes_per_param: 0.5\n")

    result = await manager.for_repo("owner/repo", github)

    assert result.bytes_per_param == 2.0


async def test_repo_can_raise_bytes_per_param(github: GitHubClient, httpx_mock: HTTPXMock) -> None:
    manager = ConfigManager(config_file=FIXTURES / "server_full.yml")
    _mock_repo(httpx_mock)
    _mock_config(httpx_mock, "bytes_per_param: 4.0\n")

    result = await manager.for_repo("owner/repo", github)

    assert result.bytes_per_param == 4.0


async def test_repo_cannot_override_infrastructure_settings(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    manager = ConfigManager(config_file=FIXTURES / "server_full.yml")
    _mock_repo(httpx_mock)
    _mock_config(
        httpx_mock,
        "qdrant:\n  url: http://evil.example.com\nembedding:\n  model: attacker-model\n",
    )

    result = await manager.for_repo("owner/repo", github)

    assert result.qdrant.url == "http://localhost:6333"
    assert result.embedding.model == "text-embedding-3-small"
