"""Full orchestration tests with mocked GitHub HTTP and mocked LLM."""

import base64
import json
from collections.abc import AsyncGenerator
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from openai.types.chat import ChatCompletionMessageToolCallUnion
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from pytest_httpx import HTTPXMock

from pr_checker.exceptions import StalePRError
from pr_checker.github_client import GitHubClient
from pr_checker.issue_resolver import IssueResolver
from pr_checker.llm_client import LLMClient
from pr_checker.model_manager import ModelManager
from pr_checker.models import PRJob, ProjectStandards, ReviewContext, ReviewTrigger, StaticFinding
from pr_checker.review_formatter import ReviewFormatter
from pr_checker.review_orchestrator import ReviewOrchestrator
from pr_checker.reviewer_config import ConfigManager
from pr_checker.standards_detector import StandardsDetector
from pr_checker.static_analyzer import StaticAnalyzer


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
                "line_number": 2,
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


class _MockStream:
    def __init__(self, chunks: list[ChatCompletionChunk]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> "_MockStream":
        return self

    async def __aexit__(self, *_: object) -> None:
        pass

    def __aiter__(self) -> Any:
        return self._iterate()

    async def _iterate(self) -> Any:
        for chunk in self._chunks:
            yield chunk


def _stream_from_response(response: ChatCompletion) -> _MockStream:
    chunks: list[ChatCompletionChunk] = []
    choice = response.choices[0]
    if choice.message.content:
        chunks.append(
            ChatCompletionChunk(
                id="chunk-test",
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(content=choice.message.content),
                        index=0,
                        finish_reason=None,
                        logprobs=None,
                    )
                ],
                created=0,
                model="gpt-4o-mini",
                object="chat.completion.chunk",
            )
        )
    if choice.message.tool_calls:
        for i, tc in enumerate(choice.message.tool_calls):
            if not isinstance(tc, ChatCompletionMessageFunctionToolCall):
                continue
            chunks.append(
                ChatCompletionChunk(
                    id="chunk-test",
                    choices=[
                        ChunkChoice(
                            delta=ChoiceDelta(
                                tool_calls=[
                                    ChoiceDeltaToolCall(
                                        index=i,
                                        id=tc.id,
                                        type="function",
                                        function=ChoiceDeltaToolCallFunction(
                                            name=tc.function.name, arguments=""
                                        ),
                                    )
                                ]
                            ),
                            index=0,
                            finish_reason=None,
                            logprobs=None,
                        )
                    ],
                    created=0,
                    model="gpt-4o-mini",
                    object="chat.completion.chunk",
                )
            )
            chunks.append(
                ChatCompletionChunk(
                    id="chunk-test",
                    choices=[
                        ChunkChoice(
                            delta=ChoiceDelta(
                                tool_calls=[
                                    ChoiceDeltaToolCall(
                                        index=i,
                                        function=ChoiceDeltaToolCallFunction(
                                            arguments=tc.function.arguments
                                        ),
                                    )
                                ]
                            ),
                            index=0,
                            finish_reason=None,
                            logprobs=None,
                        )
                    ],
                    created=0,
                    model="gpt-4o-mini",
                    object="chat.completion.chunk",
                )
            )
    chunks.append(
        ChatCompletionChunk(
            id="chunk-test",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(),
                    index=0,
                    finish_reason=choice.finish_reason or "stop",
                    logprobs=None,
                )
            ],
            created=0,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        )
    )
    return _MockStream(chunks)


def _mock_pr_head_sha(httpx_mock: HTTPXMock, sha: str = "abc123") -> None:
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/pulls/1",
        json={"head": {"sha": sha}},
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


def _mock_file_content(httpx_mock: HTTPXMock, path: str = "main.py", sha: str = "abc123") -> None:
    httpx_mock.add_response(
        url=f"https://api.github.com/repos/owner/repo/contents/{path}?ref={sha}",
        json={
            "size": 6,
            "content": base64.b64encode(b"x = 1\n").decode(),
            "encoding": "base64",
        },
    )


# --- full pipeline ---


async def test_full_pipeline_posts_summary_and_formal_review(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    _mock_pr_head_sha(httpx_mock)  # pre-flight
    _mock_pr_head_sha(httpx_mock)  # pre-output
    _mock_github_repo(httpx_mock)
    _mock_pr_checker_yml_absent(httpx_mock)
    _mock_pr_diff(httpx_mock)
    _mock_post_pr_comment(httpx_mock)
    _mock_submit_review(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_stream_from_response(_submit_review_response())
    )

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
    _mock_pr_head_sha(httpx_mock)  # pre-flight
    _mock_pr_head_sha(httpx_mock)  # pre-output
    _mock_pr_diff(httpx_mock)
    _mock_submit_review(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_stream_from_response(_submit_review_response(verdict="comment"))
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
    _mock_pr_head_sha(httpx_mock)  # pre-flight
    _mock_pr_head_sha(httpx_mock)  # pre-output
    _mock_pr_diff(httpx_mock)
    _mock_post_pr_comment(httpx_mock)
    _mock_submit_review(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_stream_from_response(_submit_review_response())
    )

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
    _mock_pr_head_sha(httpx_mock)  # pre-flight
    _mock_pr_head_sha(httpx_mock)  # pre-output
    _mock_github_repo(httpx_mock)
    _mock_pr_checker_yml_absent(httpx_mock)
    _mock_pr_diff(httpx_mock)
    _mock_post_pr_comment(httpx_mock)
    _mock_submit_review(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_stream_from_response(_submit_review_response())
    )

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
    _mock_pr_head_sha(httpx_mock)
    _mock_pr_diff(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_stream_from_response(_submit_review_response())
    )

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
    _mock_pr_head_sha(httpx_mock)  # pre-flight
    _mock_pr_head_sha(httpx_mock)  # pre-output
    _mock_github_repo(httpx_mock)
    _mock_pr_checker_yml_absent(httpx_mock)
    _mock_pr_diff(httpx_mock)
    _mock_post_pr_comment(httpx_mock)
    _mock_submit_review(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_stream_from_response(
            _submit_review_response(verdict="approve", summary="All good")
        )
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
    _mock_pr_head_sha(httpx_mock, sha="deadbeef")  # pre-flight
    _mock_pr_head_sha(httpx_mock, sha="deadbeef")  # pre-output
    _mock_github_repo(httpx_mock)
    _mock_pr_checker_yml_absent(httpx_mock)
    _mock_pr_diff(httpx_mock)
    _mock_post_pr_comment(httpx_mock)
    _mock_submit_review(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_stream_from_response(_submit_review_response())
    )

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


# --- static analysis pre-pass ---


async def test_static_findings_passed_to_llm_context(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    _mock_pr_head_sha(httpx_mock)  # pre-flight
    _mock_pr_head_sha(httpx_mock)  # pre-output
    _mock_github_repo(httpx_mock)
    _mock_pr_checker_yml_absent(httpx_mock)
    _mock_pr_diff(httpx_mock)
    _mock_file_content(httpx_mock)  # main.py fetched for static analysis
    _mock_post_pr_comment(httpx_mock)
    _mock_submit_review(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_stream_from_response(_submit_review_response())
    )

    injected_finding = StaticFinding(
        tool="ruff", file_path="main.py", line=1, col=1, code="F401", message="unused import"
    )
    mock_analyzer = AsyncMock(spec=StaticAnalyzer)
    mock_analyzer.run = AsyncMock(return_value=[injected_finding])

    captured_contexts: list[ReviewContext] = []
    original_review = LLMClient.review

    async def _spy_review(self: LLMClient, context: ReviewContext) -> Any:
        captured_contexts.append(context)
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
            static_analyzer=mock_analyzer,
        )
        await orchestrator.run(_job())

    assert len(captured_contexts) == 1
    assert captured_contexts[0].static_findings == [injected_finding]
    mock_analyzer.run.assert_called_once()
    call_kwargs = mock_analyzer.run.call_args
    assert "main.py" in call_kwargs.args[0]


async def test_static_analyzer_failure_does_not_block_review(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    _mock_pr_head_sha(httpx_mock)  # pre-flight
    _mock_pr_head_sha(httpx_mock)  # pre-output
    _mock_github_repo(httpx_mock)
    _mock_pr_checker_yml_absent(httpx_mock)
    _mock_pr_diff(httpx_mock)
    _mock_file_content(httpx_mock)
    _mock_post_pr_comment(httpx_mock)
    _mock_submit_review(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_stream_from_response(_submit_review_response())
    )

    mock_analyzer = AsyncMock(spec=StaticAnalyzer)
    mock_analyzer.run = AsyncMock(side_effect=RuntimeError("disk full"))

    with patch.object(StandardsDetector, "detect", return_value=ProjectStandards()):
        orchestrator = ReviewOrchestrator(
            github=github,
            config_manager=ConfigManager(),
            standards_detector=StandardsDetector(github),
            issue_resolver=IssueResolver(github, llm=None),
            model_manager=None,
            openai=mock_openai,
            static_analyzer=mock_analyzer,
        )
        await orchestrator.run(_job())

    mock_openai.chat.completions.create.assert_called_once()
    requests = httpx_mock.get_requests()
    urls = [str(r.url) for r in requests]
    assert any("pulls/1/reviews" in u for u in urls)


# --- pre-flight SHA check ---


async def test_stale_sha_raises_stale_pr_error(github: GitHubClient, httpx_mock: HTTPXMock) -> None:
    _mock_pr_head_sha(httpx_mock, sha="newsha99")  # current head differs from job's sha

    mock_openai = AsyncMock()

    orchestrator = ReviewOrchestrator(
        github=github,
        config_manager=ConfigManager(),
        standards_detector=StandardsDetector(github),
        issue_resolver=IssueResolver(github, llm=None),
        model_manager=None,
        openai=mock_openai,
    )

    with pytest.raises(StalePRError):
        await orchestrator.run(_job(head_sha="abc123"))  # abc123 != newsha99

    mock_openai.chat.completions.create.assert_not_called()


async def test_matching_sha_proceeds_to_review(github: GitHubClient, httpx_mock: HTTPXMock) -> None:
    _mock_pr_head_sha(httpx_mock, sha="abc123")  # pre-flight
    _mock_pr_head_sha(httpx_mock, sha="abc123")  # pre-output
    _mock_github_repo(httpx_mock)
    _mock_pr_checker_yml_absent(httpx_mock)
    _mock_pr_diff(httpx_mock)
    _mock_post_pr_comment(httpx_mock)
    _mock_submit_review(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_stream_from_response(_submit_review_response())
    )

    with patch.object(StandardsDetector, "detect", return_value=ProjectStandards()):
        orchestrator = ReviewOrchestrator(
            github=github,
            config_manager=ConfigManager(),
            standards_detector=StandardsDetector(github),
            issue_resolver=IssueResolver(github, llm=None),
            model_manager=None,
            openai=mock_openai,
        )
        await orchestrator.run(_job(head_sha="abc123"))

    mock_openai.chat.completions.create.assert_called_once()


async def test_stale_submit_422_raises_stale_pr_error(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    _mock_pr_head_sha(httpx_mock)  # pre-flight: abc123 matches
    _mock_pr_head_sha(httpx_mock)  # pre-output: abc123 matches, proceed to post
    _mock_pr_head_sha(httpx_mock, sha="newsha99")  # 422 re-check: head moved
    _mock_github_repo(httpx_mock)
    _mock_pr_checker_yml_absent(httpx_mock)
    _mock_pr_diff(httpx_mock)
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/pulls/1/reviews",
        method="POST",
        status_code=422,
        json={"message": "Reference does not exist"},
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_stream_from_response(_submit_review_response())
    )

    with patch.object(StandardsDetector, "detect", return_value=ProjectStandards()):
        orchestrator = ReviewOrchestrator(
            github=github,
            config_manager=ConfigManager(),
            standards_detector=StandardsDetector(github),
            issue_resolver=IssueResolver(github, llm=None),
            model_manager=None,
            openai=mock_openai,
        )
        with pytest.raises(StalePRError):
            await orchestrator.run(_job())


async def test_submit_422_validation_error_reraises(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    """A 422 where the head SHA has NOT moved is a validation error, not a stale PR."""
    import httpx as _httpx

    _mock_pr_head_sha(httpx_mock)  # pre-flight: abc123 matches
    _mock_pr_head_sha(httpx_mock)  # pre-output: abc123 matches, proceed to post
    _mock_pr_head_sha(httpx_mock)  # 422 re-check: still abc123 — head did not move
    _mock_github_repo(httpx_mock)
    _mock_pr_checker_yml_absent(httpx_mock)
    _mock_pr_diff(httpx_mock)
    httpx_mock.add_response(
        url="https://api.github.com/repos/owner/repo/pulls/1/reviews",
        method="POST",
        status_code=422,
        json={"message": "Invalid line position"},
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_stream_from_response(_submit_review_response())
    )

    with patch.object(StandardsDetector, "detect", return_value=ProjectStandards()):
        orchestrator = ReviewOrchestrator(
            github=github,
            config_manager=ConfigManager(),
            standards_detector=StandardsDetector(github),
            issue_resolver=IssueResolver(github, llm=None),
            model_manager=None,
            openai=mock_openai,
        )
        with pytest.raises(_httpx.HTTPStatusError):
            await orchestrator.run(_job())


async def test_head_moved_before_posting_raises_stale_pr_error(
    github: GitHubClient, httpx_mock: HTTPXMock
) -> None:
    """Pre-output SHA check fires even when only a summary comment would be posted."""
    _mock_pr_head_sha(httpx_mock, sha="abc123")  # pre-flight: matches
    _mock_pr_head_sha(httpx_mock, sha="newsha99")  # pre-output: head moved
    _mock_github_repo(httpx_mock)
    _mock_pr_checker_yml_absent(httpx_mock)
    _mock_pr_diff(httpx_mock)

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_stream_from_response(_submit_review_response())
    )

    with patch.object(StandardsDetector, "detect", return_value=ProjectStandards()):
        orchestrator = ReviewOrchestrator(
            github=github,
            config_manager=ConfigManager(),
            standards_detector=StandardsDetector(github),
            issue_resolver=IssueResolver(github, llm=None),
            model_manager=None,
            openai=mock_openai,
        )
        with pytest.raises(StalePRError):
            await orchestrator.run(_job(head_sha="abc123"))

    requests = httpx_mock.get_requests()
    urls = [str(r.url) for r in requests]
    assert not any("issues/1/comments" in u for u in urls)
    assert not any("pulls/1/reviews" in u for u in urls)
