import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response

from pr_checker.config import ServerConfig
from pr_checker.db import PersistenceLayer
from pr_checker.github_client import GitHubClient
from pr_checker.queue import ReviewQueue
from pr_checker.reviewer_config import ConfigManager
from pr_checker.webhook import parse_pr_event, validate_signature


def create_app(config: ServerConfig | None = None) -> FastAPI:
    cfg = config or ServerConfig()

    persistence = PersistenceLayer(cfg.database_url)
    github = GitHubClient(cfg.github_token)
    config_manager = ConfigManager(Path(cfg.config_file) if cfg.config_file else None)
    queue = ReviewQueue(persistence, github, config_manager)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        if not cfg.github_webhook_secret:
            raise RuntimeError("GITHUB_WEBHOOK_SECRET env var is required")
        if not cfg.github_token:
            logging.warning("GITHUB_TOKEN is not set; GitHub API calls will fail")
        await persistence.init()
        await queue.start()
        yield
        await queue.stop()
        await github.aclose()
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
