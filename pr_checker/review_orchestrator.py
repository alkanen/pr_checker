from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from pr_checker.github_client import GitHubClient
from pr_checker.issue_resolver import IssueResolver
from pr_checker.llm_client import LLMClient
from pr_checker.model_manager import ModelManager
from pr_checker.models import PRJob, ReviewContext
from pr_checker.review_formatter import ReviewFormatter
from pr_checker.reviewer_config import ConfigManager
from pr_checker.standards_detector import StandardsDetector


class ReviewOrchestrator:
    def __init__(
        self,
        github: GitHubClient,
        config_manager: ConfigManager,
        standards_detector: StandardsDetector,
        issue_resolver: IssueResolver,
        model_manager: ModelManager | None,
        openai: AsyncOpenAI,
    ) -> None:
        self._github = github
        self._config_manager = config_manager
        self._standards_detector = standards_detector
        self._issue_resolver = issue_resolver
        self._model_manager = model_manager
        self._openai = openai
        self._formatter = ReviewFormatter()

    async def run(self, job: PRJob) -> None:
        logging.info("Starting review for %s #%d", job.repo_full_name, job.pr_number)

        config = await self._config_manager.for_repo(job.repo_full_name, self._github)

        hunks = await self._github.get_pr_diff(job.repo_full_name, job.pr_number)
        logging.info(
            "Fetched diff for %s #%d: %d hunks", job.repo_full_name, job.pr_number, len(hunks)
        )

        standards = await self._standards_detector.detect(job.repo_full_name, job.head_sha)
        issues = await self._issue_resolver.resolve(
            job.repo_full_name, job.pr_title, job.pr_body, job.head_branch
        )
        logging.info(
            "Context ready for %s #%d: %d linked issues",
            job.repo_full_name,
            job.pr_number,
            len(issues),
        )

        context = ReviewContext(hunks=hunks, standards=standards, linked_issues=issues)
        estimated_tokens = _estimate_tokens(context)

        if self._model_manager is not None:
            model_id = await self._model_manager.get_model_for_task(
                "code_review", estimated_tokens, config
            )
        else:
            model_id = config.models.tasks.code_review

        logging.info(
            "Sending %s #%d to model %s (~%d tokens)",
            job.repo_full_name,
            job.pr_number,
            model_id,
            estimated_tokens,
        )
        llm = LLMClient(
            openai=self._openai,
            model=model_id,
            github=self._github,
            repo_full_name=job.repo_full_name,
            head_sha=job.head_sha,
        )
        result = await llm.review(context)
        logging.info(
            "LLM review complete for %s #%d: verdict=%s, %d findings",
            job.repo_full_name,
            job.pr_number,
            result.verdict,
            len(result.findings),
        )

        payload = self._formatter.format(result, config)

        if payload.summary_comment is not None:
            await self._github.post_pr_comment(
                job.repo_full_name, job.pr_number, payload.summary_comment
            )

        # Submit a review whenever there is a formal verdict or inline comments to deliver.
        # Inline comments require a review submission regardless of the formal_review flag,
        # so they are gated on their own presence rather than on payload.review_event.
        if payload.inline_comments or payload.review_event is not None:
            comments: list[dict[str, Any]] = [
                {"path": c.path, "line": c.line, "body": c.body, "side": "RIGHT"}
                for c in payload.inline_comments
            ]
            event = payload.review_event or "COMMENT"
            await self._github.submit_pr_review(
                repo_full_name=job.repo_full_name,
                pr_number=job.pr_number,
                commit_sha=job.head_sha,
                body=payload.review_body,
                event=event,
                comments=comments if comments else None,
            )
            logging.info(
                "Submitted review for %s #%d: %s, %d findings",
                job.repo_full_name,
                job.pr_number,
                event,
                len(result.findings),
            )


def _estimate_tokens(context: ReviewContext) -> int:
    chars = sum(len(line.content) for hunk in context.hunks for line in hunk.lines)
    chars += sum(len(issue.body) for issue in context.linked_issues)
    return max(1000, chars // 4)
