import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from openai import AsyncOpenAI

from pr_checker.config import ServerConfig
from pr_checker.db import PersistenceLayer
from pr_checker.github_client import GitHubClient
from pr_checker.issue_resolver import IssueResolver
from pr_checker.logging_setup import configure_logging
from pr_checker.queue import ReviewQueue
from pr_checker.review_orchestrator import ReviewOrchestrator
from pr_checker.reviewer_config import ConfigManager
from pr_checker.standards_detector import StandardsDetector
from pr_checker.static_analyzer import StaticAnalyzer
from pr_checker.webhook import parse_pr_event, validate_signature

_log = logging.getLogger(__name__)


def create_app(config: ServerConfig | None = None) -> FastAPI:
    cfg = config or ServerConfig()
    configure_logging(cfg.log_level, cfg.log_format)

    persistence = PersistenceLayer(cfg.database_url)
    github = GitHubClient(cfg.github_token)
    config_manager = ConfigManager(Path(cfg.config_file) if cfg.config_file else None)
    standards_detector = StandardsDetector(github)

    openai_client = AsyncOpenAI(
        api_key=cfg.openai_api_key or "sk-placeholder",
        base_url=cfg.openai_base_url or None,
    )
    issue_resolver = IssueResolver(github, llm=openai_client)

    lm_studio = None
    model_manager = None
    if cfg.lm_studio_url:
        from pr_checker.lm_studio_client import LMStudioClient
        from pr_checker.model_manager import ModelManager

        lm_studio = LMStudioClient(cfg.lm_studio_url, cfg.lm_studio_api_key)
        model_manager = ModelManager(lm_studio)

    orchestrator = ReviewOrchestrator(
        github=github,
        config_manager=config_manager,
        standards_detector=standards_detector,
        issue_resolver=issue_resolver,
        model_manager=model_manager,
        openai=openai_client,
        model_override=cfg.llm_model or None,
        static_analyzer=StaticAnalyzer(),
    )
    queue = ReviewQueue(persistence, github, orchestrator)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        if not cfg.github_webhook_secret:
            raise RuntimeError("GITHUB_WEBHOOK_SECRET env var is required")
        if not cfg.github_token:
            _log.warning("GITHUB_TOKEN is not set; GitHub API calls will fail")
        await persistence.init()
        await queue.start()
        yield
        await queue.stop()
        await github.aclose()
        await openai_client.close()
        if lm_studio is not None:
            await lm_studio.aclose()
        await persistence.dispose()

    application = FastAPI(title="pr-checker", lifespan=lifespan)
    application.state.queue = queue

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/webhook")
    async def webhook(request: Request) -> Response:
        payload_bytes = await request.body()
        validate_signature(
            payload_bytes,
            request.headers.get("X-Hub-Signature-256"),
            cfg.github_webhook_secret,
        )

        event = request.headers.get("X-GitHub-Event", "")
        if event != "pull_request":
            return Response(status_code=204)

        try:
            payload: dict[str, Any] = json.loads(payload_bytes)
        except json.JSONDecodeError:
            return Response(status_code=400, content="Invalid JSON")
        job = parse_pr_event(payload)
        if job is None:
            return Response(status_code=204)

        await queue.enqueue(job)
        return Response(status_code=202)

    return application


app = create_app()
