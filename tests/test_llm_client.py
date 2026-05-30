import json
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import ChatCompletionMessageToolCallUnion
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice as ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from pr_checker.config import ServerConfig
from pr_checker.github_client import GitHubClient
from pr_checker.llm_client import LLMClient, _parse_review_result
from pr_checker.models import (
    DiffHunk,
    DiffLine,
    LinkedIssue,
    ProjectStandards,
    ReviewContext,
    Severity,
)


# ---------------------------------------------------------------------------
# Streaming mock infrastructure
# ---------------------------------------------------------------------------


class _MockStream:
    """Async context manager / async iterable backed by a list of chunks."""

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
    """Convert a non-streaming ChatCompletion into a minimal streaming mock."""
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
            # Chunk 1: call id + function name
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
                                            name=tc.function.name,
                                            arguments="",
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
            # Chunk 2: arguments
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
                                            arguments=tc.function.arguments,
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

    # Final chunk carries the finish_reason
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


# ---------------------------------------------------------------------------
# Non-streaming response builders (still useful for _stream_from_response)
# ---------------------------------------------------------------------------


def _context() -> ReviewContext:
    hunk = DiffHunk(
        file_path="main.py",
        header="@@ -1,3 +1,4 @@",
        old_start=1,
        old_count=3,
        new_start=1,
        new_count=4,
        lines=[DiffLine(kind="+", content="x = 1/0", old_lineno=None, new_lineno=4)],
    )
    return ReviewContext(
        hunks=[hunk],
        standards=ProjectStandards(),
        linked_issues=[],
    )


def _make_tool_call(
    name: str, args: dict[str, Any], call_id: str = "call_1"
) -> ChatCompletionMessageFunctionToolCall:
    return ChatCompletionMessageFunctionToolCall(
        id=call_id,
        type="function",
        function=Function(name=name, arguments=json.dumps(args)),
    )


def _make_response(
    tool_calls: list[ChatCompletionMessageFunctionToolCall] | None = None,
    content: str | None = None,
    finish_reason: Literal[
        "stop", "length", "tool_calls", "content_filter", "function_call"
    ] = "tool_calls",
) -> ChatCompletion:
    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        tool_calls=cast(list[ChatCompletionMessageToolCallUnion], tool_calls),
    )
    choice = Choice(finish_reason=finish_reason, index=0, message=message, logprobs=None)
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[choice],
        created=0,
        model="gpt-4o-mini",
        object="chat.completion",
    )


def _submit_response(
    findings: list[dict[str, Any]] | None = None,
    verdict: str = "request_changes",
    summary: str = "Issues found",
) -> ChatCompletion:
    args: dict[str, Any] = {
        "findings": findings
        or [
            {
                "severity": "high",
                "confidence": 0.9,
                "category": "logic",
                "message": "Division by zero",
                "file_path": "main.py",
                "line_number": 4,
            }
        ],
        "verdict": verdict,
        "summary": summary,
    }
    return _make_response(tool_calls=[_make_tool_call("submit_review", args)])


def _no_tool_calls_response(content: str = "Here is my analysis.") -> ChatCompletion:
    return _make_response(tool_calls=None, content=content, finish_reason="stop")


def _tool_call_response(name: str, args: dict[str, Any], call_id: str = "call_1") -> ChatCompletion:
    return _make_response(tool_calls=[_make_tool_call(name, args, call_id)])


@pytest.fixture
def mock_github() -> AsyncMock:
    return AsyncMock(spec=GitHubClient)


def _client(mock_openai: Any, mock_github: AsyncMock, max_turns: int = 10) -> LLMClient:
    return LLMClient(
        openai=mock_openai,
        model="gpt-4o-mini",
        github=mock_github,
        repo_full_name="owner/repo",
        head_sha="abc123",
        max_turns=max_turns,
    )


def _mock_openai(*responses: ChatCompletion) -> Any:
    """Return a mock openai client whose create yields streaming versions of responses."""
    streams = [_stream_from_response(r) for r in responses]
    mock = MagicMock()
    mock.chat.completions.create = AsyncMock(
        side_effect=streams if len(streams) > 1 else None,
        return_value=streams[0] if len(streams) == 1 else None,
    )
    return mock


# ---------------------------------------------------------------------------
# ServerConfig — llm_timeout
# ---------------------------------------------------------------------------


def test_llm_timeout_default() -> None:
    assert ServerConfig().llm_timeout == 600.0


def test_llm_timeout_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_TIMEOUT", "120")
    assert ServerConfig().llm_timeout == 120.0


# ---------------------------------------------------------------------------
# submit_review termination
# ---------------------------------------------------------------------------


async def test_submit_review_returns_result(mock_github: AsyncMock) -> None:
    result = await _client(_mock_openai(_submit_response()), mock_github).review(_context())

    assert len(result.findings) == 1
    assert result.findings[0].severity == Severity.HIGH
    assert result.findings[0].message == "Division by zero"
    assert result.verdict == "request_changes"
    assert result.summary == "Issues found"


async def test_submit_review_parses_all_finding_fields(mock_github: AsyncMock) -> None:
    findings = [
        {
            "severity": "critical",
            "confidence": 0.95,
            "category": "security",
            "message": "SQL injection",
            "suggestion": "Use parameterized queries",
            "file_path": "db.py",
            "line_number": 10,
        }
    ]
    result = await _client(_mock_openai(_submit_response(findings=findings)), mock_github).review(
        _context()
    )

    f = result.findings[0]
    assert f.severity == Severity.CRITICAL
    assert f.confidence == 0.95
    assert f.category == "security"
    assert f.suggestion == "Use parameterized queries"
    assert f.file_path == "db.py"
    assert f.line_number == 10


# ---------------------------------------------------------------------------
# get_code_snippet dispatch
# ---------------------------------------------------------------------------


async def test_get_code_snippet_resolved_and_loop_continues(mock_github: AsyncMock) -> None:
    mock_github.get_file_content.return_value = "line1\nline2\nline3"
    r1 = _tool_call_response("get_code_snippet", {"file_path": "main.py", "start_line": 1})
    mock_openai = _mock_openai(r1, _submit_response())

    result = await _client(mock_openai, mock_github).review(_context())

    mock_github.get_file_content.assert_called_once_with("owner/repo", "main.py", "abc123")
    assert mock_openai.chat.completions.create.call_count == 2
    assert len(result.findings) == 1


async def test_get_code_snippet_clamps_end_line(mock_github: AsyncMock) -> None:
    mock_github.get_file_content.return_value = "\n".join(f"line{i}" for i in range(1, 1001))
    r1 = _tool_call_response(
        "get_code_snippet", {"file_path": "big.py", "start_line": 1, "end_line": 9999}
    )
    mock_openai = _mock_openai(r1, _submit_response())

    await _client(mock_openai, mock_github).review(_context())

    # The tool result appended to messages should not be 9999 lines
    second_call_messages = mock_openai.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_result = next(m for m in second_call_messages if m.get("role") == "tool")
    snippet_lines = tool_result["content"].count("\n")
    assert snippet_lines <= 201  # 200 content lines + code fence markers


async def test_get_code_snippet_file_fetch_error_returns_error_text(
    mock_github: AsyncMock,
) -> None:
    mock_github.get_file_content.side_effect = Exception("network error")
    r1 = _tool_call_response("get_code_snippet", {"file_path": "missing.py"})
    mock_openai = _mock_openai(r1, _submit_response())

    await _client(mock_openai, mock_github).review(_context())

    second_call_messages = mock_openai.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_result = next(m for m in second_call_messages if m.get("role") == "tool")
    assert "Error" in tool_result["content"]


# ---------------------------------------------------------------------------
# search_code stub
# ---------------------------------------------------------------------------


async def test_search_code_returns_empty_stub(mock_github: AsyncMock) -> None:
    r1 = _tool_call_response("search_code", {"query": "database connection"})
    mock_openai = _mock_openai(r1, _submit_response())

    await _client(mock_openai, mock_github).review(_context())

    second_call_messages = mock_openai.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_result = next(m for m in second_call_messages if m.get("role") == "tool")
    assert tool_result["content"] == "[]"


# ---------------------------------------------------------------------------
# max turns
# ---------------------------------------------------------------------------


async def test_max_turns_enforced(mock_github: AsyncMock) -> None:
    mock_github.get_file_content.return_value = "content"
    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=lambda **_: _stream_from_response(
            _tool_call_response("get_code_snippet", {"file_path": "a.py"})
        )
    )
    result = await _client(mock_openai, mock_github, max_turns=3).review(_context())

    assert mock_openai.chat.completions.create.call_count == 3
    assert any("turn" in f.message.lower() or "limit" in f.message.lower() for f in result.findings)
    assert result.verdict == "comment"


# ---------------------------------------------------------------------------
# no-tool-calls fallback
# ---------------------------------------------------------------------------


async def test_no_tool_calls_returns_fallback_with_system_finding(mock_github: AsyncMock) -> None:
    result = await _client(_mock_openai(_no_tool_calls_response()), mock_github).review(_context())

    assert result.verdict == "comment"
    assert any(f.category == "system" for f in result.findings)


async def test_no_tool_calls_surfaces_assistant_content_as_summary(
    mock_github: AsyncMock,
) -> None:
    result = await _client(
        _mock_openai(_no_tool_calls_response(content="The code looks problematic here.")),
        mock_github,
    ).review(_context())

    assert result.summary == "The code looks problematic here."


# ---------------------------------------------------------------------------
# Streaming: multi-chunk argument accumulation
# ---------------------------------------------------------------------------


async def test_streaming_arguments_split_across_chunks(mock_github: AsyncMock) -> None:
    """Tool-call arguments that arrive across multiple chunks must be concatenated correctly."""
    args_dict: dict[str, Any] = {"findings": [], "verdict": "approve", "summary": "LGTM"}
    full_args = json.dumps(args_dict)
    mid = len(full_args) // 2

    chunks: list[ChatCompletionChunk] = [
        # Chunk 1: call id + function name
        ChatCompletionChunk(
            id="chunk-test",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_1",
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name="submit_review", arguments=""
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
        ),
        # Chunk 2: first half of arguments
        ChatCompletionChunk(
            id="chunk-test",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                function=ChoiceDeltaToolCallFunction(arguments=full_args[:mid]),
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
        ),
        # Chunk 3: second half + finish
        ChatCompletionChunk(
            id="chunk-test",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                function=ChoiceDeltaToolCallFunction(arguments=full_args[mid:]),
                            )
                        ]
                    ),
                    index=0,
                    finish_reason="tool_calls",
                    logprobs=None,
                )
            ],
            created=0,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        ),
    ]

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=_MockStream(chunks))
    result = await _client(mock_openai, mock_github).review(_context())

    assert result.verdict == "approve"
    assert result.summary == "LGTM"


async def test_streaming_missing_id_synthesised(mock_github: AsyncMock) -> None:
    """When a backend omits the tool-call id, a stable id is synthesised and the call proceeds."""
    args_dict: dict[str, Any] = {"findings": [], "verdict": "approve", "summary": "ok"}
    chunks: list[ChatCompletionChunk] = [
        # No id field on the first chunk (some LM Studio-compatible backends omit it)
        ChatCompletionChunk(
            id="chunk-test",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                function=ChoiceDeltaToolCallFunction(
                                    name="submit_review", arguments=json.dumps(args_dict)
                                ),
                            )
                        ]
                    ),
                    index=0,
                    finish_reason="tool_calls",
                    logprobs=None,
                )
            ],
            created=0,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        ),
    ]

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=_MockStream(chunks))
    result = await _client(mock_openai, mock_github).review(_context())

    assert result.verdict == "approve"
    assert result.summary == "ok"


# ---------------------------------------------------------------------------
# _parse_review_result
# ---------------------------------------------------------------------------


def test_parse_unknown_severity_defaults_to_info() -> None:
    result = _parse_review_result(
        {
            "findings": [{"severity": "bogus", "confidence": 0.5, "category": "x", "message": "m"}],
            "verdict": "comment",
            "summary": "s",
        }
    )
    assert result.findings[0].severity == Severity.INFO


def test_parse_confidence_clamped_above_1() -> None:
    result = _parse_review_result(
        {
            "findings": [{"severity": "info", "confidence": 2.5, "category": "x", "message": "m"}],
            "verdict": "comment",
            "summary": "s",
        }
    )
    assert result.findings[0].confidence == 1.0


def test_parse_confidence_clamped_below_0() -> None:
    result = _parse_review_result(
        {
            "findings": [{"severity": "info", "confidence": -0.5, "category": "x", "message": "m"}],
            "verdict": "comment",
            "summary": "s",
        }
    )
    assert result.findings[0].confidence == 0.0


def test_parse_invalid_verdict_defaults_to_comment() -> None:
    result = _parse_review_result({"findings": [], "verdict": "merge_immediately", "summary": "s"})
    assert result.verdict == "comment"


def test_parse_optional_suggestion_absent_is_none() -> None:
    result = _parse_review_result(
        {
            "findings": [{"severity": "low", "confidence": 0.5, "category": "x", "message": "m"}],
            "verdict": "approve",
            "summary": "s",
        }
    )
    assert result.findings[0].suggestion is None


def test_parse_empty_findings_list() -> None:
    result = _parse_review_result({"findings": [], "verdict": "approve", "summary": "LGTM"})
    assert result.findings == []
    assert result.verdict == "approve"
    assert result.summary == "LGTM"


# ---------------------------------------------------------------------------
# context building includes linked issues
# ---------------------------------------------------------------------------


async def test_user_message_includes_linked_issue(mock_github: AsyncMock) -> None:
    hunk = DiffHunk(
        file_path="f.py",
        header="@@ -1 +1 @@",
        old_start=1,
        old_count=1,
        new_start=1,
        new_count=1,
        lines=[DiffLine(kind="+", content="pass", old_lineno=None, new_lineno=1)],
    )
    context = ReviewContext(
        hunks=[hunk],
        standards=ProjectStandards(),
        linked_issues=[
            LinkedIssue(
                number=99, title="Fix race condition", body="Details here", labels=[], assignees=[]
            )
        ],
    )
    mock_openai = _mock_openai(_submit_response())
    await _client(mock_openai, mock_github).review(context)

    user_message = mock_openai.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "Issue #99" in user_message
    assert "Fix race condition" in user_message
