import asyncio
import logging
import re

from openai import AsyncOpenAI

from pr_checker.github_client import GitHubClient
from pr_checker.models import LinkedIssue

_CLOSING_KEYWORDS = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)",
    re.IGNORECASE,
)
_ISSUE_URL = re.compile(r"https://github\.com/[^/\s]+/[^/\s]+/issues/(\d+)")
_BRANCH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:^|/)issue-?(\d+)(?:-|$)"),  # issue-N  or  /issue-N
    re.compile(r"(?:^|/)[a-z][a-z0-9]*/(\d+)-[a-z]"),  # type/N-text-description
    re.compile(r"(?:^|/)[a-z][a-z0-9]*/(\d+)$"),  # type/N  (number at end)
]


class IssueResolver:
    def __init__(
        self,
        github: GitHubClient,
        llm: AsyncOpenAI | None = None,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self._github = github
        self._llm = llm
        self._llm_model = llm_model

    async def resolve(
        self,
        repo_full_name: str,
        pr_title: str,
        pr_body: str,
        head_branch: str,
    ) -> list[LinkedIssue]:
        numbers = _extract_closing_keywords(pr_body)
        if not numbers:
            numbers = _extract_issue_urls(pr_body)
        if not numbers:
            numbers = _extract_from_branch(head_branch)
        if not numbers and self._llm is not None:
            try:
                numbers = await self._infer_from_llm(self._llm, pr_title, pr_body)
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.warning("LLM issue inference failed; skipping", exc_info=True)

        issues: list[LinkedIssue] = []
        seen: set[int] = set()
        for n in numbers:
            if n in seen:
                continue
            seen.add(n)
            try:
                issue = await self._github.get_issue(repo_full_name, n)
                issues.append(issue)
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.warning(
                    "Failed to fetch issue #%d for %s", n, repo_full_name, exc_info=True
                )
        return issues

    async def _infer_from_llm(self, llm: AsyncOpenAI, title: str, body: str) -> list[int]:
        response = await llm.chat.completions.create(
            model=self._llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You identify GitHub issue numbers referenced by a pull request. "
                        "Reply with a comma-separated list of integers only, "
                        "e.g. '42' or '12, 34'. "
                        "Reply with an empty string if no issues are referenced."
                    ),
                },
                {"role": "user", "content": f"PR title: {title}\nPR description:\n{body}"},
            ],
            max_tokens=50,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return []
        return [int(m) for m in re.findall(r"\d+", text)]


def _extract_closing_keywords(body: str) -> list[int]:
    return [int(m) for m in _CLOSING_KEYWORDS.findall(body)]


def _extract_issue_urls(body: str) -> list[int]:
    return [int(m) for m in _ISSUE_URL.findall(body)]


def _extract_from_branch(branch: str) -> list[int]:
    for pattern in _BRANCH_PATTERNS:
        m = pattern.search(branch)
        if m:
            return [int(m.group(1))]
    return []
