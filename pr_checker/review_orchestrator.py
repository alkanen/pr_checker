from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, NoReturn

import httpx
from openai import AsyncOpenAI

from pr_checker.context_builder import ContextBuilder
from pr_checker.exceptions import StalePRError
from pr_checker.github_client import GitHubClient
from pr_checker.identifier_extractor import extract_identifiers
from pr_checker.issue_resolver import IssueResolver
from pr_checker.llm_client import MAX_STATIC_FINDINGS, LLMClient
from pr_checker.model_manager import ModelManager
from pr_checker.models import CodeSnippet, PRJob, ReviewContext, StaticFinding
from pr_checker.review_formatter import ReviewFormatter
from pr_checker.reviewer_config import ConfigManager
from pr_checker.standards_detector import StandardsDetector
from pr_checker.static_analyzer import StaticAnalyzer

_log = logging.getLogger(__name__)


class ReviewOrchestrator:
    def __init__(
        self,
        github: GitHubClient,
        config_manager: ConfigManager,
        standards_detector: StandardsDetector,
        issue_resolver: IssueResolver,
        model_manager: ModelManager | None,
        openai: AsyncOpenAI,
        model_override: str | None = None,
        static_analyzer: StaticAnalyzer | None = None,
        debug_dir: Path | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self._github = github
        self._config_manager = config_manager
        self._standards_detector = standards_detector
        self._issue_resolver = issue_resolver
        self._model_manager = model_manager
        self._openai = openai
        self._model_override = model_override
        self._static_analyzer = static_analyzer
        self._debug_dir = debug_dir
        self._context_builder = context_builder
        self._formatter = ReviewFormatter()

    async def run(self, job: PRJob) -> None:
        _log.info(
            "Review starting: %s #%d (sha=%s)", job.repo_full_name, job.pr_number, job.head_sha[:8]
        )

        current_sha = await self._github.get_pr_head_sha(job.repo_full_name, job.pr_number)
        if current_sha != job.head_sha:
            raise StalePRError(
                f"PR #{job.pr_number} head moved from {job.head_sha[:8]} to {current_sha[:8]}"
            )

        config = await self._config_manager.for_repo(job.repo_full_name, self._github)

        t0 = time.monotonic()
        hunks = await self._github.get_pr_diff(job.repo_full_name, job.pr_number)
        _log.info(
            "Diff fetched in %.1fs: %d hunks for %s #%d",
            time.monotonic() - t0,
            len(hunks),
            job.repo_full_name,
            job.pr_number,
        )

        t_std = time.monotonic()
        standards = await self._standards_detector.detect(job.repo_full_name, job.head_sha)
        _log.info(
            "Standards detected in %.1fs: ruff=%s mypy=%s eslint=%s prettier=%s",
            time.monotonic() - t_std,
            bool(standards.ruff),
            bool(standards.mypy),
            bool(standards.eslint),
            bool(standards.prettier),
        )

        t1 = time.monotonic()
        issues = await self._issue_resolver.resolve(
            job.repo_full_name, job.pr_title, job.pr_body, job.head_branch
        )
        _log.info(
            "Issues resolved in %.1fs: %d linked issues for %s #%d",
            time.monotonic() - t1,
            len(issues),
            job.repo_full_name,
            job.pr_number,
        )

        static_findings: list[StaticFinding] = []
        if self._static_analyzer is not None and (config.analyzers.ruff or config.analyzers.mypy):
            t_sa = time.monotonic()
            python_paths = sorted({h.file_path for h in hunks if h.file_path.endswith(".py")})
            file_contents: dict[str, str] = {}
            for path in python_paths:
                try:
                    content = await self._github.get_file_content(
                        job.repo_full_name, path, job.head_sha
                    )
                except Exception:
                    _log.debug("Could not fetch %s for static analysis; skipping", path)
                    continue
                if content is not None:
                    file_contents[path] = content
            try:
                static_findings = await self._static_analyzer.run(
                    file_contents,
                    run_ruff=config.analyzers.ruff,
                    run_mypy=config.analyzers.mypy,
                    ruff_config=standards.ruff or None,
                    mypy_config=standards.mypy or None,
                )
                _log.info(
                    "Static analysis in %.1fs: %d findings for %s #%d",
                    time.monotonic() - t_sa,
                    len(static_findings),
                    job.repo_full_name,
                    job.pr_number,
                )
            except Exception:
                _log.warning(
                    "Static analysis failed for %s #%d; continuing without findings",
                    job.repo_full_name,
                    job.pr_number,
                    exc_info=True,
                )

        snippets: list[CodeSnippet] = []
        if self._context_builder is not None:
            t_ctx = time.monotonic()
            identifiers = extract_identifiers(hunks)
            snippets = await self._context_builder.build(
                identifiers=identifiers,
                repo_full_name=job.repo_full_name,
                base_branch=job.base_branch,
                head_branch=job.head_branch,
                limit=config.retrieval.max_prefetch_chunks,
            )
            _log.info(
                "Context retrieval in %.1fs: %d snippets for %s #%d",
                time.monotonic() - t_ctx,
                len(snippets),
                job.repo_full_name,
                job.pr_number,
            )

        context = ReviewContext(
            hunks=hunks,
            standards=standards,
            linked_issues=issues,
            static_findings=static_findings,
            snippets=snippets,
        )
        estimated_tokens = _estimate_tokens(context)

        if self._model_override is not None:
            model_id = self._model_override
            model_reason = "override"
        elif self._model_manager is not None:
            model_id = await self._model_manager.get_model_for_task(
                "code_review", estimated_tokens, config
            )
            model_reason = "manager"
        else:
            model_id = config.models.tasks.code_review
            model_reason = "config"

        _log.info(
            "Model selected: %s (reason=%s, ~%d tokens) for %s #%d",
            model_id,
            model_reason,
            estimated_tokens,
            job.repo_full_name,
            job.pr_number,
        )

        llm = LLMClient(
            openai=self._openai,
            model=model_id,
            github=self._github,
            repo_full_name=job.repo_full_name,
            pr_number=job.pr_number,
            head_sha=job.head_sha,
            debug_dir=self._debug_dir,
            context_builder=self._context_builder,
            head_branch=job.head_branch,
            max_search_chunks=config.retrieval.max_search_chunks_per_call
            if config.retrieval.max_search_chunks_per_call is not None
            else 5,
        )

        t2 = time.monotonic()
        result = await llm.review(context)
        _log.info(
            "LLM review complete in %.1fs: verdict=%s, %d findings for %s #%d",
            time.monotonic() - t2,
            result.verdict,
            len(result.findings),
            job.repo_full_name,
            job.pr_number,
        )

        payload = self._formatter.format(result, config)

        # GitHub's review API returns 422 for any comment whose line is not part of
        # the diff context.  Filter to right-side lines that actually appear in the
        # diff so a single out-of-range finding doesn't abort the whole review.
        if payload.inline_comments:
            valid_diff_lines: set[tuple[str, int]] = {
                (hunk.file_path, line.new_lineno)
                for hunk in hunks
                for line in hunk.lines
                if line.new_lineno is not None
            }
            valid = [c for c in payload.inline_comments if (c.path, c.line) in valid_diff_lines]
            dropped = len(payload.inline_comments) - len(valid)
            if dropped:
                _log.info(
                    "Dropped %d inline comment(s) outside the diff for %s #%d",
                    dropped,
                    job.repo_full_name,
                    job.pr_number,
                )
            payload.inline_comments = valid

        has_output = (
            payload.summary_comment is not None
            or bool(payload.inline_comments)
            or payload.review_event is not None
        )
        if has_output:
            pre_post_sha = await self._github.get_pr_head_sha(job.repo_full_name, job.pr_number)
            if pre_post_sha != job.head_sha:
                raise StalePRError(
                    f"PR #{job.pr_number} head moved before posting: "
                    f"{job.head_sha[:8]} → {pre_post_sha[:8]}"
                )

        # Submit the commit-bound review before the unbound summary comment so that
        # a 422 (stale commit) aborts before any comment is posted to the PR.
        if payload.inline_comments or payload.review_event is not None:
            comments: list[dict[str, Any]] = [
                {"path": c.path, "line": c.line, "body": c.body, "side": "RIGHT"}
                for c in payload.inline_comments
            ]
            event = payload.review_event or "COMMENT"
            # GitHub requires a non-empty body for all non-APPROVE review events.
            # Guard against LLMs that omit or blank out the summary field.
            review_body = payload.review_body.strip()
            if event != "APPROVE" and not review_body:
                review_body = "Review complete — see inline comments and findings above."
            try:
                await self._github.submit_pr_review(
                    repo_full_name=job.repo_full_name,
                    pr_number=job.pr_number,
                    commit_sha=job.head_sha,
                    body=review_body,
                    event=event,
                    comments=comments if comments else None,
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 422:
                    raise
                _log.warning(
                    "422 from GitHub reviews API for %s #%d: %s",
                    job.repo_full_name,
                    job.pr_number,
                    exc.response.text,
                )
                # GitHub forbids APPROVE and REQUEST_CHANGES on your own PR.
                # Retry as COMMENT so findings are still posted when
                # reviewer == PR author.  Not a bug — just means the formal
                # approval/request-changes state can't be set.  The permanent
                # fix is to use a separate GitHub account for the bot.
                if (
                    event in ("REQUEST_CHANGES", "APPROVE")
                    and "own pull request" in exc.response.text
                ):
                    _log.info(
                        "Retrying review as COMMENT (reviewer is PR author, "
                        "cannot %s own PR) for %s #%d",
                        event,
                        job.repo_full_name,
                        job.pr_number,
                    )
                    verdict = event.replace("_", " ").lower()
                    own_pr_note = (
                        f"\n\n---\n*Note: the reviewer wanted to "
                        f"**{verdict}** this PR, but GitHub does not "
                        f"allow setting that status on your own pull "
                        f"request. Use a separate bot account to "
                        f"enable formal review statuses.*"
                    )
                    review_body = (review_body or "") + own_pr_note
                    event = "COMMENT"
                    try:
                        await self._github.submit_pr_review(
                            repo_full_name=job.repo_full_name,
                            pr_number=job.pr_number,
                            commit_sha=job.head_sha,
                            body=review_body,
                            event=event,
                            comments=comments if comments else None,
                        )
                    except httpx.HTTPStatusError as retry_exc:
                        if retry_exc.response.status_code == 422:
                            await self._raise_stale_or_reraise(retry_exc, job, "retry")
                        raise
                else:
                    await self._raise_stale_or_reraise(exc, job)
            _log.info(
                "Review submitted: %s, %d inline comments for %s #%d",
                event,
                len(payload.inline_comments),
                job.repo_full_name,
                job.pr_number,
            )

        if payload.summary_comment is not None:
            await self._github.post_pr_comment(
                job.repo_full_name, job.pr_number, payload.summary_comment
            )
            _log.info("Summary comment posted for %s #%d", job.repo_full_name, job.pr_number)

    async def _raise_stale_or_reraise(
        self,
        exc: httpx.HTTPStatusError,
        job: PRJob,
        context: str = "",
    ) -> NoReturn:
        """Check whether the PR head moved; raise StalePRError if so, else re-raise exc."""
        head_moved = False
        label = f" ({context})" if context else ""
        try:
            current_sha = await self._github.get_pr_head_sha(job.repo_full_name, job.pr_number)
            head_moved = current_sha != job.head_sha
        except Exception:
            _log.warning(
                "Could not re-check head SHA after 422%s for %s #%d; treating as validation error",
                label,
                job.repo_full_name,
                job.pr_number,
                exc_info=True,
            )
        if head_moved:
            raise StalePRError(
                f"PR #{job.pr_number} review rejected (422): "
                f"commit {job.head_sha[:8]} is no longer the head"
            ) from exc
        raise exc


def _estimate_tokens(context: ReviewContext) -> int:
    chars = sum(len(line.content) for hunk in context.hunks for line in hunk.lines)
    chars += sum(len(issue.body) for issue in context.linked_issues)
    shown_findings = context.static_findings[:MAX_STATIC_FINDINGS]
    chars += sum(len(f.file_path) + len(f.code) + len(f.message) for f in shown_findings)
    chars += sum(
        len(s.content) + len(s.file_path) + len(s.chunk_name) + 80 for s in context.snippets
    )
    return max(1000, chars // 4)
