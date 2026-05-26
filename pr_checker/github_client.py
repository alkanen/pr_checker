import base64
import logging
import re
from typing import Any

import httpx

from pr_checker.models import DiffHunk, DiffLine, LinkedIssue

_HUNK_HEADER = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)")


def _parse_patch(file_path: str, patch: str) -> list[DiffHunk]:
    hunks: list[DiffHunk] = []
    current: DiffHunk | None = None
    old_lineno = 0
    new_lineno = 0

    for line in patch.splitlines():
        m = _HUNK_HEADER.match(line)
        if m:
            if current is not None:
                hunks.append(current)
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) is not None else 1
            current = DiffHunk(
                file_path=file_path,
                header=line,
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                lines=[],
            )
            old_lineno = old_start
            new_lineno = new_start
            continue

        if current is None:
            continue

        if line.startswith("+"):
            current.lines.append(
                DiffLine(kind="+", content=line[1:], old_lineno=None, new_lineno=new_lineno)
            )
            new_lineno += 1
        elif line.startswith("-"):
            current.lines.append(
                DiffLine(kind="-", content=line[1:], old_lineno=old_lineno, new_lineno=None)
            )
            old_lineno += 1
        elif line.startswith(" "):
            current.lines.append(
                DiffLine(kind=" ", content=line[1:], old_lineno=old_lineno, new_lineno=new_lineno)
            )
            old_lineno += 1
            new_lineno += 1
        # skip "\ No newline at end of file" lines

    if current is not None:
        hunks.append(current)

    return hunks


class GitHubClient:
    _BASE_URL = "https://api.github.com"

    def __init__(self, token: str, http_client: httpx.AsyncClient | None = None) -> None:
        self._token = token
        self._http = http_client

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._BASE_URL,
                headers={
                    "Authorization": f"token {self._token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                timeout=30.0,
            )
        return self._http

    async def post_commit_status(
        self,
        repo_full_name: str,
        sha: str,
        state: str,
        description: str,
        context: str = "pr-checker",
    ) -> None:
        r = await self._client().post(
            f"/repos/{repo_full_name}/statuses/{sha}",
            json={"state": state, "description": description, "context": context},
        )
        r.raise_for_status()

    async def post_pr_comment(self, repo_full_name: str, pr_number: int, body: str) -> None:
        r = await self._client().post(
            f"/repos/{repo_full_name}/issues/{pr_number}/comments",
            json={"body": body},
        )
        r.raise_for_status()

    async def get_pr_diff(
        self,
        repo_full_name: str,
        pr_number: int,
        max_files: int = 3000,
    ) -> list[DiffHunk]:
        hunks: list[DiffHunk] = []
        per_page = 100
        page = 1
        files_processed = 0

        while True:
            r = await self._client().get(
                f"/repos/{repo_full_name}/pulls/{pr_number}/files",
                params={"per_page": per_page, "page": page},
            )
            r.raise_for_status()
            batch: list[dict[str, Any]] = r.json()
            if not batch:
                break

            for file_info in batch:
                if files_processed >= max_files:
                    logging.warning(
                        "get_pr_diff: %s #%d exceeds max_files=%d; truncating",
                        repo_full_name,
                        pr_number,
                        max_files,
                    )
                    return hunks
                patch: str | None = file_info.get("patch")
                if patch is not None:
                    hunks.extend(_parse_patch(file_info["filename"], patch))
                files_processed += 1

            if len(batch) < per_page:
                break
            page += 1

        return hunks

    async def get_file_content(
        self,
        repo_full_name: str,
        path: str,
        ref: str,
        size_threshold: int = 100 * 1024,
    ) -> str | None:
        r = await self._client().get(
            f"/repos/{repo_full_name}/contents/{path}",
            params={"ref": ref},
        )
        r.raise_for_status()
        data: dict[str, Any] = r.json()
        if data.get("size", 0) > size_threshold:
            return None
        content = data.get("content", "")
        if data.get("encoding") == "base64":
            return base64.b64decode(content).decode("utf-8", errors="replace")
        return str(content)

    async def get_issue(self, repo_full_name: str, issue_number: int) -> LinkedIssue:
        r = await self._client().get(f"/repos/{repo_full_name}/issues/{issue_number}")
        r.raise_for_status()
        data: dict[str, Any] = r.json()
        return LinkedIssue(
            number=data["number"],
            title=data["title"],
            body=data.get("body") or "",
            labels=[label["name"] for label in data.get("labels", [])],
            assignees=[a["login"] for a in data.get("assignees", [])],
        )

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
