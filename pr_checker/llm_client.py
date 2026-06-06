from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import httpx
from openai import APITimeoutError, AsyncOpenAI

from pr_checker.code_chunker import LANG_EXT
from pr_checker.context_builder import ContextBuilder
from pr_checker.github_client import GitHubClient
from pr_checker.models import CodeSnippet, Finding, ReviewContext, ReviewResult, Severity

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

_SUBMIT_TOOL = [t for t in _TOOLS if t["function"]["name"] == "submit_review"]
_MAX_SNIPPET_LINES = 200
MAX_STATIC_FINDINGS = 20
_SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]
_LOG_TRUNCATE = 200
_STREAM_LOG_INTERVAL = 30.0


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
        pr_number: int,
        head_sha: str,
        max_turns: int = DEFAULT_MAX_TURNS,
        debug_dir: Path | None = None,
        context_builder: ContextBuilder | None = None,
        head_branch: str = "",  # required when context_builder is set
        max_search_chunks: int = 5,
    ) -> None:
        self._openai = openai
        self._model = model
        self._github = github
        self._repo = repo_full_name
        self._pr_number = pr_number
        self._sha = head_sha
        self._max_turns = max_turns
        self._debug_dir = debug_dir
        self._context_builder = context_builder
        self._head_branch = head_branch
        self._max_search_chunks = max_search_chunks

    async def review(self, context: ReviewContext) -> ReviewResult:
        messages: list[Any] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(context)},
        ]
        for m in messages:
            self._append_log(m)

        prev_prompt_chars = 0
        reminded = False
        force_submit = False
        total_prompt_tokens = 0
        total_completion_tokens = 0
        for turn in range(self._max_turns):
            prompt_chars = sum(
                len(m["content"]) for m in messages if isinstance(m.get("content"), str)
            )
            initial_content = str(messages[1].get("content", "")) if len(messages) > 1 else ""
            snippet = initial_content[:120].replace("\n", " ")
            _log.info(
                "Turn %d/%d: %s | %d chars (+%d), %d messages | %s",
                turn + 1,
                self._max_turns,
                self._model,
                prompt_chars,
                prompt_chars - prev_prompt_chars,
                len(messages),
                snippet,
            )
            prev_prompt_chars = prompt_chars

            if force_submit:
                tool_choice = "required"
                tools = _SUBMIT_TOOL
            else:
                tool_choice = "auto"
                tools = None
            force_submit = False
            try:
                (
                    content,
                    reasoning,
                    tool_calls,
                    finish_reason,
                    elapsed,
                    usage,
                ) = await self._stream_turn(messages, turn, tool_choice=tool_choice, tools=tools)
            except (httpx.TimeoutException, APITimeoutError) as exc:
                _log.warning(
                    "LLM stream timed out on turn %d/%d (model=%s): %s",
                    turn + 1,
                    self._max_turns,
                    self._model,
                    exc,
                )
                return _fallback_result(
                    "LLM request timed out; review incomplete.",
                    _collect_assistant_content(messages),
                )

            if usage:
                total_prompt_tokens += usage["prompt_tokens"]
                total_completion_tokens += usage["completion_tokens"]
                usage_str = (
                    f" | prompt={usage['prompt_tokens']}"
                    f" completion={usage['completion_tokens']}"
                    f" total={usage['total_tokens']}"
                )
            else:
                usage_str = ""

            if tool_calls:
                _log.info(
                    "Turn %d/%d completed in %.1fs: tools=[%s]%s",
                    turn + 1,
                    self._max_turns,
                    elapsed,
                    ", ".join(tc["function"]["name"] for tc in tool_calls),
                    usage_str,
                )
            else:
                _log.debug(
                    "Turn %d/%d completed in %.1fs: no tools (finish=%s)%s",
                    turn + 1,
                    self._max_turns,
                    elapsed,
                    finish_reason,
                    usage_str,
                )

            msg: dict[str, Any] = {"role": "assistant"}
            if content:
                msg["content"] = content
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)

            log_entry: dict[str, Any] = dict(msg)
            if reasoning:
                log_entry["reasoning"] = reasoning
            if usage:
                log_entry["usage"] = usage
            log_entry["elapsed"] = round(elapsed, 2)
            log_entry["turn"] = turn + 1
            self._append_log(log_entry)

            if not tool_calls:
                if not reminded and turn < self._max_turns - 1:
                    if content:
                        # Strip intermediate tool exchanges: keep only the system
                        # prompt, the initial PR diff, and the model's prose so it
                        # can convert its analysis into a submit_review call without
                        # being weighed down by the full history.
                        prose_msg = messages[-1]
                        dropped = len(messages) - 3  # system + initial_user + prose
                        reminder = (
                            "The intermediate tool-call history has been trimmed"
                            " to reduce context size. Your analysis above contains"
                            " your findings. Please call `submit_review` now with"
                            " those findings."
                        )
                        messages = [
                            messages[0],
                            messages[1],
                            prose_msg,
                            {"role": "user", "content": reminder},
                        ]
                        self._append_log(messages[-1])
                        kind = "prose"
                    else:
                        # Model returned a completely empty response — drop it and
                        # prompt directly without any intermediate history.
                        dropped = len(messages) - 2  # system + initial_user only
                        reminder = (
                            "You did not provide a response. Please call `submit_review`"
                            " now with your findings based on the diff."
                        )
                        messages = [messages[0], messages[1], {"role": "user", "content": reminder}]
                        self._append_log(messages[-1])
                        kind = "empty response"
                    _log.warning(
                        "Turn %d/%d: model gave %s but no tool call (model=%s); "
                        "trimmed %d intermediate messages and sending submit_review reminder",
                        turn + 1,
                        self._max_turns,
                        kind,
                        self._model,
                        dropped,
                    )
                    reminded = True
                    force_submit = True
                    continue
                _log.warning(
                    "LLM did not call submit_review on turn %d (model=%s); fallback result",
                    turn + 1,
                    self._model,
                )
                return _fallback_result("Model did not call submit_review.", content or "")

            for tool_call in tool_calls:
                name = tool_call["function"]["name"]
                raw_args = tool_call["function"]["arguments"]
                bad_args: str | None = None
                try:
                    parsed = json.loads(raw_args) if raw_args else {}
                    if not isinstance(parsed, dict):
                        bad_args = f"expected JSON object, got {type(parsed).__name__}"
                    else:
                        args: dict[str, Any] = parsed
                except json.JSONDecodeError as json_exc:
                    bad_args = str(json_exc)
                if bad_args is not None:
                    _log.warning(
                        "Bad arguments in tool call %r (model=%s): %s",
                        name,
                        self._model,
                        bad_args,
                    )
                    if name == "submit_review":
                        return _fallback_result("submit_review called with malformed arguments.")
                    error_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": f"Error: could not parse tool arguments — {bad_args}",
                    }
                    self._append_log(error_msg)
                    messages.append(error_msg)
                    continue

                if name == "submit_review":
                    if total_prompt_tokens:
                        _log.info(
                            "Tool call: submit_review(verdict=%s, findings=%d) | "
                            "cumulative tokens: prompt=%d completion=%d total=%d",
                            args.get("verdict", "?"),
                            len(args.get("findings", [])),
                            total_prompt_tokens,
                            total_completion_tokens,
                            total_prompt_tokens + total_completion_tokens,
                        )
                    else:
                        _log.info(
                            "Tool call: submit_review(verdict=%s, findings=%d)",
                            args.get("verdict", "?"),
                            len(args.get("findings", [])),
                        )
                    return _parse_review_result(args)

                _log.info(
                    "Tool call: %s(%s)", name, _truncate(json.dumps(args, ensure_ascii=False))
                )
                t1 = time.monotonic()
                result_text = await self._dispatch_tool(name, args)
                elapsed_ms = int((time.monotonic() - t1) * 1000)
                _log.info("Tool result: %s -> %d chars (%dms)", name, len(result_text), elapsed_ms)
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result_text,
                }
                self._append_log(tool_msg)
                messages.append(tool_msg)

        _log.warning(
            "LLM review exceeded max_turns=%d (model=%s); returning partial result",
            self._max_turns,
            self._model,
        )
        return _fallback_result(
            f"Review exceeded the maximum tool-call turn limit ({self._max_turns}).",
            _collect_assistant_content(messages),
        )

    def _log_path(self) -> Path | None:
        if self._debug_dir is None:
            return None
        owner, repo = self._repo.split("/", 1)
        return self._debug_dir / f"{owner}__{repo}__{self._pr_number}__{self._sha}.jsonl"

    def _append_log(self, entry: dict[str, Any]) -> None:
        path = self._log_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            _log.warning("Failed to append to debug log %s", path, exc_info=True)

    async def _stream_turn(
        self,
        messages: list[Any],
        turn: int,
        tool_choice: Any = "auto",
        tools: list[Any] | None = None,
    ) -> tuple[
        str | None, str | None, list[dict[str, Any]], str | None, float, dict[str, int] | None
    ]:
        """Stream one LLM turn.

        Returns (content, reasoning, tool_calls, finish_reason, elapsed, usage).

        LLM_TIMEOUT acts as the read timeout — max wait between any two received chunks.
        This covers slow time-to-first-token on CPU-only backends as well as stalls mid-stream.
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: dict[str, int] | None = None
        chunk_count = 0
        t0 = time.monotonic()
        t_last_log = t0

        async with await self._openai.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools if tools is not None else _TOOLS,
            tool_choice=tool_choice,
            stream=True,
        ) as stream:
            async for chunk in stream:
                chunk_count += 1
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = {
                        "prompt_tokens": getattr(chunk_usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(chunk_usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(chunk_usage, "total_tokens", 0) or 0,
                    }
                delta = choice.delta
                if delta.content:
                    content_parts.append(delta.content)
                # Qwen3 and other thinking models stream reasoning in a non-standard
                # reasoning_content field rather than content.  Capture it so we can
                # use it as a fallback when content is empty.
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    reasoning_parts.append(reasoning)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.id:
                            tool_calls_acc[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_acc[idx]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls_acc[idx]["arguments"] += tc.function.arguments

                now = time.monotonic()
                if now - t_last_log >= _STREAM_LOG_INTERVAL:
                    _log.info(
                        "Streaming turn %d/%d: %d chunks, %.0fs elapsed (model=%s)",
                        turn + 1,
                        self._max_turns,
                        chunk_count,
                        now - t0,
                        self._model,
                    )
                    t_last_log = now

        elapsed = time.monotonic() - t0
        content = "".join(content_parts) or None
        if not content and reasoning_parts:
            # Thinking model: all output went to reasoning_content, visible content is empty.
            # Promote reasoning so the reminder path can pass it back as context.
            content = "".join(reasoning_parts)
            _log.debug(
                "Turn %d/%d: promoted reasoning_content to content (%d chars, model=%s)",
                turn + 1,
                self._max_turns,
                len(content),
                self._model,
            )
        tool_calls: list[dict[str, Any]] = []
        for idx, acc in sorted(tool_calls_acc.items()):
            if not acc["name"]:
                _log.warning("Skipping streamed tool call at index %d: missing function name", idx)
                continue
            call_id = acc["id"] or f"call_{idx}"
            if not acc["id"]:
                # Some OpenAI-compatible backends (e.g. LM Studio) omit the id on delta chunks.
                _log.warning(
                    "Streamed tool call %r at index %d has no id; synthesizing %r",
                    acc["name"],
                    idx,
                    call_id,
                )
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": acc["name"], "arguments": acc["arguments"]},
                }
            )
        # Some model builds emit tool calls as <tool_call> markup in text rather
        # than structured tool_calls — parse and promote them transparently.
        if not tool_calls and content and "<tool_call>" in content:
            parsed = _parse_text_tool_calls(content)
            if parsed:
                _log.warning(
                    "Turn %d/%d: extracted %d text-based tool call(s) from content (model=%s); "
                    "model template does not emit structured tool_calls",
                    turn + 1,
                    self._max_turns,
                    len(parsed),
                    self._model,
                )
                tool_calls = parsed
                content = (
                    re.sub(r"\s*<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL).strip()
                    or None
                )

        reasoning = "".join(reasoning_parts) or None
        return content, reasoning, tool_calls, finish_reason, elapsed, usage

    async def _dispatch_tool(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name == "get_code_snippet":
                return await self._get_code_snippet(args)
            if name == "search_code":
                return await self._search_code(args)
            _log.warning("Unknown tool called: %s", name)
            return f"Unknown tool: {name}"
        except Exception as exc:
            _log.warning("Tool %s raised an unexpected error: %s", name, exc, exc_info=True)
            return f"Error: tool {name} failed ({exc})"

    async def _get_code_snippet(self, args: dict[str, Any]) -> str:
        file_path = args.get("file_path")
        if not file_path:
            return "Error: get_code_snippet requires a file_path argument."
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

    async def _search_code(self, args: dict[str, Any]) -> str:
        query = args.get("query")
        if not query:
            return "Error: search_code requires a query argument."
        if self._context_builder is None:
            return "No results found."
        # Searches the feature branch only; the LLM already has the diff
        # showing what changed from base, so base-branch results would be
        # redundant or stale.
        results = await self._context_builder.search(
            query=query,
            repo_full_name=self._repo,
            branch=self._head_branch,
            limit=self._max_search_chunks,
        )
        if not results:
            return "No results found."
        return _format_snippets(results)


def _parse_text_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse <tool_call> markup from text content into structured tool call dicts.

    Handles two formats emitted by models whose chat template lacks native tool-call
    support:

    Format 1 — JSON body:
        <tool_call>
        {"name": "fn", "arguments": {"k": "v"}}
        </tool_call>

    Format 2 — function/parameter XML (Qwen3 LM Studio template):
        <tool_call>
        <function=fn_name>
        <parameter>key>value</parameter>
        </function>
        </tool_call>
    """
    result: list[dict[str, Any]] = []
    for block in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL):
        inner = block.group(1).strip()
        if inner.startswith("{"):
            try:
                obj = json.loads(inner)
                name = obj.get("name") or (obj.get("function") or {}).get("name")
                args = obj.get("arguments") or obj.get("parameters") or {}
                if name:
                    result.append(
                        {
                            "id": f"text_{len(result)}",
                            "type": "function",
                            "function": {"name": str(name), "arguments": json.dumps(args)},
                        }
                    )
            except (json.JSONDecodeError, AttributeError):
                pass
        else:
            fn_match = re.search(r"<function=(\w+)>", inner)
            if not fn_match:
                continue
            fn_name = fn_match.group(1)
            params: dict[str, str] = {}
            for pm in re.finditer(r"<parameter>(\w+)>\s*(.*?)\s*</parameter>", inner, re.DOTALL):
                params[pm.group(1)] = pm.group(2).strip()
            result.append(
                {
                    "id": f"text_{len(result)}",
                    "type": "function",
                    "function": {"name": fn_name, "arguments": json.dumps(params)},
                }
            )
    return result


def _collect_assistant_content(messages: list[Any]) -> str:
    parts = [
        m["content"]
        for m in messages
        if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"]
    ]
    return "\n\n".join(parts)


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
        hunk_content = hunk.header + "\n" + "\n".join(f"{dl.kind}{dl.content}" for dl in hunk.lines)
        fence = _safe_fence(hunk_content)
        parts.append(f"### File: {hunk.file_path}\n{fence}diff")
        parts.append(hunk.header)
        for dl in hunk.lines:
            parts.append(f"{dl.kind}{dl.content}")
        parts.append(f"{fence}\n")

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

    if context.snippets:
        parts.append("## Related Code\n")
        parts.append(
            "The following code snippets were retrieved from the repository as context "
            "relevant to the changes in this PR.\n"
        )
        parts.append(_format_snippets(context.snippets))

    parts.append(
        "Analyse the diff. Use tools if needed, then call submit_review with your findings."
    )
    return "\n".join(parts)


def _guess_language(file_path: str) -> str:
    from pathlib import PurePosixPath

    suffix = PurePosixPath(file_path).suffix.lower()
    return LANG_EXT.get(suffix, "")


def _safe_fence(content: str) -> str:
    runs = re.findall(r"`+", content)
    longest = max((len(r) for r in runs), default=0)
    return "`" * max(3, longest + 1)


def _format_snippets(snippets: list[CodeSnippet]) -> str:
    parts: list[str] = []
    for snippet in snippets:
        lang = _guess_language(snippet.file_path)
        parts.append(
            f"### {snippet.file_path} ({snippet.chunk_type}: {snippet.chunk_name}) "
            f"[{snippet.branch}]"
        )
        parts.append(f"Lines {snippet.start_line}-{snippet.end_line}")
        fence = _safe_fence(snippet.content)
        parts.append(f"{fence}{lang}")
        parts.append(snippet.content)
        parts.append(f"{fence}\n")
    return "\n".join(parts)
