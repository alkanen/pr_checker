"""Full orchestration tests with mocked GitHub HTTP and mocked LLM."""

import json
from collections.abc import AsyncGenerator
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from openai.types.chat import ChatCompletionMessageToolCallUnion
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from pytest_httpx import HTTPXMock

from pr_checker.github_client import GitHubClient
from pr_checker.issue_resolver import IssueResolver
from pr_checker.llm_client import LLMClient
from pr_checker.model_manager import ModelManager
from pr_checker.models import PRJob, ProjectStandards, ReviewTrigger
from pr_checker.review_formatter import ReviewFormatter
from pr_checker.review_orchestrator import ReviewOrchestrator
from pr_checker.reviewer_config import ConfigManager
from pr_checker.standards_detector import StandardsDetector


@pytest.fixture
async def github() -> AsyncGenerator[GitHubClient, None]:
    client = GitHubClient(token="test-token")
    yield client
    await client.aclose()


def _job(**kwargs: Any) -> PRJob:
    defaults: dict[str, Any] = {
        "repo_full_name": "owner/repo",
        "pr_number": 1,
        "pr_title": "Fix bug",
        "pr_body": "",
        "head_sha": "abc123",
        "base_sha": "def456",
        "head_branch": "fix/bug",
        "base_branch": "main",
        "trigger": ReviewTrigger.OPENED,
    }
    defaults.update(kwargs)
    return PRJob(**defaults)


def _submit_review_response(
    verdict: str = "request_changes",
    summary: str = "Issues found",
    findings: list[dict[str, Any]] | None = None,
) -> ChatCompletion:
    args: dict[str, Any] = {
        "findings": findings
        or [
            {
                "severity": "high",
                "confidence": 0.9,
                "category": "logic",
                "message": "Deliberate bug",
                "file_path": "main.py",
                "line_number": 4,
            }
        ],
        "verdict": verdict,
        "summary": summary,
    }
    tool_call = ChatCompletionMessageFunctionToolCall(
        id="call_1",
        type="function",
        function=Function(name="submit_review", arguments=json.dumps(args)),
    )
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=cast(list[ChatCompletionMessageToolCallUnion], [tool_call]),
    )
    choice = Choice(finish_reason="tool_calls", index=0, message=message, logprobs=None)
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[choice],
        created=0,
        model="gpt-4o-mini",
        object="chat.completion",
    )


def _mock_github_repo(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo",
        json={"default_branch": "main"},
    )


def _mock_pr_checker_yml_absent(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/contents/.pr-checker.yml?ref=main",
        status_code=404,
        json={"message": "Not Found"},
    )


def _mock_pr_diff(httpx_mock: HTTPXMock) -> None:
    patch = "@@ -1,3 +1,4 @@\n context\n+x = 1/0\n context2"
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/pulls/1/files?per_page=100&page=1",
        json=[{"filename": "main.py", "patch": patch}],
    )


def _mock_post_pr_comment(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/issues/1/comments",
        method="POST",
        json={"id": 1},
    )


def _mock_submit_review(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/pulls/1/reviews",
        method="POST",
        json={"id": 1},
    )


# --- full pipeline ---


async def test_full_pipeline_posts_summary_and_formal_review(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    _mock_github_repo(httpx_mock)
    _mock_pr_checker_yml_absent(httpx_mock)
    _mock_pr_diff(httpx_mock)
    _mock_post_pr_comment(httpx_mock)
    _mock_submit_review(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=_submit_review_response())

    with patch.object(StandardsDetector, "detect", return_value=ProjectStandards()):
        orchestrator = ReviewOrchestrator(
            github=github,
            config_manager=ConfigManager(),
            standards_detector=StandardsDetector(github),
            issue_resolver=IssueResolver(github, llm=None),
            model_manager=None,
            openai=mock_openai,
        )
        await orchestrator.run(_job())

    mock_openai.chat.completions.create.assert_called_once()
    requests = httpx_mock.get_requests()
    methods_urls = [(r.method, str(r.url)) for r in requests]
    assert any("issues/1/comments" in u for _, u in methods_urls)
    assert any("pulls/1/reviews" in u for _, u in methods_urls)


async def test_summary_comment_not_posted_when_disabled(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    # for_repo is patched so no GitHub repo/config HTTP calls are made
    _mock_pr_diff(httpx_mock)
    _mock_submit_review(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_submit_review_response(verdict="comment")
    )

    from pr_checker.reviewer_config import OutputConfig, ReviewerConfig

    config_no_summary = ReviewerConfig(
        output=OutputConfig(summary_comment=False, inline_comments=False, formal_review=True)
    )

    with (
        patch.object(StandardsDetector, "detect", return_value=ProjectStandards()),
        patch.object(
            ConfigManager, "for_repo", new_callable=AsyncMock, return_value=config_no_summary
        ),
    ):
        orchestrator = ReviewOrchestrator(
            github=github,
            config_manager=ConfigManager(),
            standards_detector=StandardsDetector(github),
            issue_resolver=IssueResolver(github, llm=None),
            model_manager=None,
            openai=mock_openai,
        )
        await orchestrator.run(_job())

    requests = httpx_mock.get_requests()
    urls = [str(r.url) for r in requests]
    assert not any("issues/1/comments" in u for u in urls)
    assert any("pulls/1/reviews" in u for u in urls)


async def test_inline_comments_posted_when_formal_review_disabled(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    """Inline comments are delivered via a COMMENT review even when formal_review=False."""
    # for_repo is patched so no GitHub repo/config HTTP calls are made
    _mock_pr_diff(httpx_mock)
    _mock_post_pr_comment(httpx_mock)
    _mock_submit_review(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=_submit_review_response())

    from pr_checker.reviewer_config import OutputConfig, ReviewerConfig

    config_no_formal = ReviewerConfig(
        output=OutputConfig(inline_comments=True, summary_comment=True, formal_review=False)
    )

    with (
        patch.object(StandardsDetector, "detect", return_value=ProjectStandards()),
        patch.object(
            ConfigManager, "for_repo", new_callable=AsyncMock, return_value=config_no_formal
        ),
    ):
        orchestrator = ReviewOrchestrator(
            github=github,
            config_manager=ConfigManager(),
            standards_detector=StandardsDetector(github),
            issue_resolver=IssueResolver(github, llm=None),
            model_manager=None,
            openai=mock_openai,
        )
        await orchestrator.run(_job())

    requests = httpx_mock.get_requests()
    urls = [str(r.url) for r in requests]
    # summary comment posted
    assert any("issues/1/comments" in u for u in urls)
    # review submitted to deliver inline comments
    assert any("pulls/1/reviews" in u for u in urls)
    # verify event is COMMENT (not a verdict)
    review_req = next(r for r in requests if "pulls/1/reviews" in str(r.url))
    body = json.loads(review_req.content)
    assert body["event"] == "COMMENT"


async def test_model_manager_used_when_provided(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    _mock_github_repo(httpx_mock)
    _mock_pr_checker_yml_absent(httpx_mock)
    _mock_pr_diff(httpx_mock)
    _mock_post_pr_comment(httpx_mock)
    _mock_submit_review(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=_submit_review_response())

    mock_model_manager = AsyncMock(spec=ModelManager)
    mock_model_manager.get_model_for_task = AsyncMock(return_value="custom-model")

    with patch.object(StandardsDetector, "detect", return_value=ProjectStandards()):
        orchestrator = ReviewOrchestrator(
            github=github,
            config_manager=ConfigManager(),
            standards_detector=StandardsDetector(github),
            issue_resolver=IssueResolver(github, llm=None),
            model_manager=mock_model_manager,
            openai=mock_openai,
        )
        await orchestrator.run(_job())

    mock_model_manager.get_model_for_task.assert_called_once()
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "custom-model"


async def test_no_review_submitted_when_all_output_disabled(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    # for_repo is patched so no GitHub repo/config HTTP calls are made
    _mock_pr_diff(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=_submit_review_response())

    from pr_checker.reviewer_config import OutputConfig, ReviewerConfig

    config_silent = ReviewerConfig(
        output=OutputConfig(inline_comments=False, summary_comment=False, formal_review=False)
    )

    with (
        patch.object(StandardsDetector, "detect", return_value=ProjectStandards()),
        patch.object(ConfigManager, "for_repo", new_callable=AsyncMock, return_value=config_silent),
    ):
        orchestrator = ReviewOrchestrator(
            github=github,
            config_manager=ConfigManager(),
            standards_detector=StandardsDetector(github),
            issue_resolver=IssueResolver(github, llm=None),
            model_manager=None,
            openai=mock_openai,
        )
        await orchestrator.run(_job())

    requests = httpx_mock.get_requests()
    urls = [str(r.url) for r in requests]
    assert not any("issues/1/comments" in u for u in urls)
    assert not any("pulls/1/reviews" in u for u in urls)


async def test_formatter_called_with_review_result(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    _mock_github_repo(httpx_mock)
    _mock_pr_checker_yml_absent(httpx_mock)
    _mock_pr_diff(httpx_mock)
    _mock_post_pr_comment(httpx_mock)
    _mock_submit_review(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_submit_review_response(verdict="approve", summary="All good")
    )

    captured_results = []

    original_format = ReviewFormatter.format

    def _spy_format(self: ReviewFormatter, result: Any, config: Any) -> Any:
        captured_results.append(result)
        return original_format(self, result, config)

    with (
        patch.object(StandardsDetector, "detect", return_value=ProjectStandards()),
        patch.object(ReviewFormatter, "format", _spy_format),
    ):
        orchestrator = ReviewOrchestrator(
            github=github,
            config_manager=ConfigManager(),
            standards_detector=StandardsDetector(github),
            issue_resolver=IssueResolver(github, llm=None),
            model_manager=None,
            openai=mock_openai,
        )
        await orchestrator.run(_job())

    assert len(captured_results) == 1
    assert captured_results[0].verdict == "approve"
    assert captured_results[0].summary == "All good"


# --- LLMClient gets correct repo context ---


async def test_llm_client_receives_correct_repo_and_sha(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    _mock_github_repo(httpx_mock)
    _mock_pr_checker_yml_absent(httpx_mock)
    _mock_pr_diff(httpx_mock)
    _mock_post_pr_comment(httpx_mock)
    _mock_submit_review(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=_submit_review_response())

    captured_clients: list[LLMClient] = []
    original_review = LLMClient.review

    async def _spy_review(self: LLMClient, context: Any) -> Any:
        captured_clients.append(self)
        return await original_review(self, context)

    with (
        patch.object(StandardsDetector, "detect", return_value=ProjectStandards()),
        patch.object(LLMClient, "review", _spy_review),
    ):
        orchestrator = ReviewOrchestrator(
            github=github,
            config_manager=ConfigManager(),
            standards_detector=StandardsDetector(github),
            issue_resolver=IssueResolver(github, llm=None),
            model_manager=None,
            openai=mock_openai,
        )
        await orchestrator.run(_job(head_sha="deadbeef"))

    assert len(captured_clients) == 1
    assert captured_clients[0]._repo == "owner/repo"
    assert captured_clients[0]._sha == "deadbeef"
