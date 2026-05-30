from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

from pr_checker.github_client import GitHubClient
from pr_checker.models import Finding, ReviewContext, ReviewResult, Severity

_log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a senior software engineer performing a pull request code review.
Analyse the PR diff and context provided. Identify bugs, security issues, code quality problems,
and non-conformance to the linked issues.

Use get_code_snippet to fetch surrounding code when you need more context.
Use search_code to find related code by concept or identifier.
When you have finished your analysis, call submit_review with all findings.
"""

_TOOLS: list[Any] = [
    {
        "type": "function",
        "function": {
            "name": "get_code_snippet",
            "description": "Fetch a range of lines from a file at the PR head commit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file"},
                    "start_line": {"type": "integer", "description": "First line (1-based)"},
                    "end_line": {"type": "integer", "description": "Last line (inclusive)"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the codebase for code related to a query. Returns snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language or keyword query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_review",
            "description": "Submit the completed code review. Call this once analysis is done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "severity": {
                                    "type": "string",
                                    "enum": ["critical", "high", "medium", "low", "info"],
                                },
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "category": {"type": "string"},
                                "message": {"type": "string"},
                                "suggestion": {"type": "string"},
                                "file_path": {"type": "string"},
                                "line_number": {"type": "integer"},
                            },
                            "required": ["severity", "confidence", "category", "message"],
                        },
                    },
                    "verdict": {
                        "type": "string",
                        "enum": ["approve", "request_changes", "comment"],
                    },
                    "summary": {"type": "string"},
                },
                "required": ["findings", "verdict", "summary"],
            },
        },
    },
]

_MAX_SNIPPET_LINES = 200
MAX_STATIC_FINDINGS = 20
_SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]
_LOG_TRUNCATE = 200


def _truncate(text: str, limit: int = _LOG_TRUNCATE) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


class LLMClient:
    # 25 turns gives models room to make several get_code_snippet / search_code
    # calls before submitting; raise if reviews consistently hit the limit.
    DEFAULT_MAX_TURNS = 25

    def __init__(
        self,
        openai: AsyncOpenAI,
        model: str,
        github: GitHubClient,
        repo_full_name: str,
        head_sha: str,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        self._openai = openai
        self._model = model
        self._github = github
        self._repo = repo_full_name
        self._sha = head_sha
        self._max_turns = max_turns

    async def review(self, context: ReviewContext) -> ReviewResult:
        messages: list[Any] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(context)},
        ]

        for turn in range(self._max_turns):
            _log.debug(
                "Turn %d/%d: waiting for %s (context=%d messages)",
                turn + 1,
                self._max_turns,
                self._model,
                len(messages),
            )

            t0 = time.monotonic()
            response = await self._openai.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
            )
            elapsed = time.monotonic() - t0

            choice = response.choices[0]

            # Build the tool-name list for the debug summary before appending to messages.
            tool_names = [
                tc.function.name
                for tc in (choice.message.tool_calls or [])
                if isinstance(tc, ChatCompletionMessageFunctionToolCall)
            ]
            if tool_names:
                _log.debug(
                    "Turn %d/%d responded in %.1fs: tools=[%s]",
                    turn + 1,
                    self._max_turns,
                    elapsed,
                    ", ".join(tool_names),
                )
            else:
                _log.debug(
                    "Turn %d/%d responded in %.1fs: no tools (finish=%s)",
                    turn + 1,
                    self._max_turns,
                    elapsed,
                    choice.finish_reason,
                )

            # Append assistant message so the next turn has full context
            msg: dict[str, Any] = {"role": "assistant"}
            if choice.message.content:
                msg["content"] = choice.message.content
            if choice.message.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                    if isinstance(tc, ChatCompletionMessageFunctionToolCall)
                ]
            messages.append(msg)

            if not choice.message.tool_calls:
                _log.warning(
                    "LLM did not call submit_review on turn %d (model=%s); fallback result",
                    turn + 1,
                    self._model,
                )
                assistant_content = choice.message.content or ""
                return _fallback_result("Model did not call submit_review.", assistant_content)

            for tool_call in choice.message.tool_calls:
                if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                    continue
                name = tool_call.function.name
                args: dict[str, Any] = json.loads(tool_call.function.arguments)

                if name == "submit_review":
                    _log.debug(
                        "Tool call: submit_review(verdict=%s, findings=%d)",
                        args.get("verdict", "?"),
                        len(args.get("findings", [])),
                    )
                    return _parse_review_result(args)

                _log.debug(
                    "Tool call: %s(%s)", name, _truncate(json.dumps(args, ensure_ascii=False))
                )
                t1 = time.monotonic()
                result_text = await self._dispatch_tool(name, args)
                elapsed_ms = int((time.monotonic() - t1) * 1000)
                _log.debug("Tool result: %s -> %d chars (%dms)", name, len(result_text), elapsed_ms)
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": result_text}
                )

        _log.warning(
            "LLM review exceeded max_turns=%d (model=%s); returning partial result",
            self._max_turns,
            self._model,
        )
        return _fallback_result(
            f"Review exceeded the maximum tool-call turn limit ({self._max_turns})."
        )

    async def _dispatch_tool(self, name: str, args: dict[str, Any]) -> str:
        if name == "get_code_snippet":
            return await self._get_code_snippet(args)
        if name == "search_code":
            # stub — wired to Qdrant in Phase 7
            return "[]"
        _log.warning("Unknown tool called: %s", name)
        return f"Unknown tool: {name}"

    async def _get_code_snippet(self, args: dict[str, Any]) -> str:
        file_path: str = args["file_path"]
        start: int = max(1, int(args.get("start_line", 1)))
        end: int = int(args.get("end_line", start + _MAX_SNIPPET_LINES - 1))
        end = max(start, min(end, start + _MAX_SNIPPET_LINES - 1))
        try:
            content = await self._github.get_file_content(self._repo, file_path, self._sha)
        except Exception:
            _log.warning("get_code_snippet failed for %s", file_path, exc_info=True)
            return f"Error: could not fetch {file_path}"
        if content is None:
            return f"File {file_path} is too large to fetch."
        lines = content.splitlines()
        snippet = "\n".join(lines[start - 1 : end])
        return f"```\n{snippet}\n```"


def _fallback_result(reason: str, assistant_content: str = "") -> ReviewResult:
    return ReviewResult(
        findings=[
            Finding(
                severity=Severity.INFO,
                confidence=1.0,
                category="system",
                message=reason,
            )
        ],
        verdict="comment",
        summary=assistant_content or "Automated review was incomplete.",
    )


def _parse_review_result(args: dict[str, Any]) -> ReviewResult:
    findings: list[Finding] = []
    for raw in args.get("findings", []):
        sev_str = raw.get("severity", "info")
        try:
            severity = Severity(sev_str)
        except ValueError:
            severity = Severity.INFO
        confidence = float(raw.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        findings.append(
            Finding(
                severity=severity,
                confidence=confidence,
                category=str(raw.get("category", "general")),
                message=str(raw.get("message", "")),
                suggestion=raw.get("suggestion") or None,
                file_path=raw.get("file_path") or None,
                line_number=raw.get("line_number"),
            )
        )

    verdict_str = str(args.get("verdict", "comment"))
    if verdict_str not in ("approve", "request_changes", "comment"):
        verdict_str = "comment"

    return ReviewResult(
        findings=findings,
        verdict=verdict_str,  # type: ignore[arg-type]
        summary=str(args.get("summary", "")),
    )


def _build_user_message(context: ReviewContext) -> str:
    parts: list[str] = ["## Pull Request Diff\n"]
    for hunk in context.hunks:
        parts.append(f"### File: {hunk.file_path}\n```diff")
        parts.append(hunk.header)
        for dl in hunk.lines:
            parts.append(f"{dl.kind}{dl.content}")
        parts.append("```\n")

    if context.linked_issues:
        parts.append("## Linked Issues\n")
        for issue in context.linked_issues:
            parts.append(f"**Issue #{issue.number}: {issue.title}**")
            if issue.body:
                # Truncate very long issue bodies to avoid bloating the prompt
                body = issue.body[:1000] + "..." if len(issue.body) > 1000 else issue.body
                parts.append(body)
            parts.append("")

    standards = context.standards
    std_parts: list[str] = []
    if standards.ruff:
        std_parts.append(f"- Ruff: `{json.dumps(standards.ruff)}`")
    if standards.mypy:
        std_parts.append(f"- Mypy: `{json.dumps(standards.mypy)}`")
    if standards.eslint:
        std_parts.append(f"- ESLint: `{json.dumps(standards.eslint)}`")
    if standards.prettier:
        std_parts.append(f"- Prettier: `{json.dumps(standards.prettier)}`")
    if std_parts:
        parts.append("## Project Standards\n")
        parts.extend(std_parts)
        parts.append("")

    if context.static_findings:
        shown = context.static_findings[:MAX_STATIC_FINDINGS]
        omitted = len(context.static_findings) - len(shown)
        parts.append("## Static Analysis Findings\n")
        parts.append(
            "The following issues were detected by automated tools before this review. "
            "Evaluate whether each is a real issue in this PR's context — some may be "
            "false positives or pre-existing issues outside the scope of this change.\n"
        )
        for f in shown:
            loc = f"{f.file_path}:{f.line}" if f.line is not None else f.file_path
            code_part = f" ({f.code})" if f.code else ""
            parts.append(f"- [{f.tool}] {loc}{code_part}: {f.message}")
        if omitted:
            parts.append(
                f"\n*{omitted} additional finding(s) omitted — address the listed ones first.*"
            )
        parts.append("")

    parts.append(
        "Analyse the diff. Use tools if needed, then call submit_review with your findings."
    )
    return "\n".join(parts)
