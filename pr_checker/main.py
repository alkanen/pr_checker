import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncGenerator, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Query, Request, Response
from openai import AsyncOpenAI

from pr_checker.code_indexer import CodeIndexer, aggregate_push_files
from pr_checker.config import ServerConfig
from pr_checker.context_builder import ContextBuilder
from pr_checker.db import PersistenceLayer
from pr_checker.embedding_service import EmbeddingService
from pr_checker.github_client import GitHubClient
from pr_checker.issue_resolver import IssueResolver
from pr_checker.logging_setup import configure_logging
from pr_checker.qdrant_gateway import QdrantGateway
from pr_checker.queue import ReviewQueue
from pr_checker.review_orchestrator import ReviewOrchestrator
from pr_checker.reviewer_config import ConfigManager
from pr_checker.standards_detector import StandardsDetector
from pr_checker.static_analyzer import StaticAnalyzer
from pr_checker.webhook import parse_pr_event, validate_signature

_log = logging.getLogger(__name__)

_ZERO_SHA = "0" * 40  # branch-deletion sentinel in GitHub push events


def create_app(config: ServerConfig | None = None) -> FastAPI:
    cfg = config or ServerConfig()
    configure_logging(cfg.log_level, cfg.log_format)

    persistence = PersistenceLayer(cfg.database_url)
    github = GitHubClient(cfg.github_token)
    config_manager = ConfigManager(
        config_file=Path(cfg.config_file) if cfg.config_file else None,
        max_prefetch_chunks=cfg.max_prefetch_chunks,
        max_search_chunks_per_call=cfg.max_search_chunks_per_call,
    )
    standards_detector = StandardsDetector(github)

    openai_client = AsyncOpenAI(
        api_key=cfg.openai_api_key or "sk-placeholder",
        base_url=cfg.openai_base_url or None,
        timeout=cfg.llm_timeout,
    )
    issue_resolver = IssueResolver(github, llm=openai_client)

    lm_studio = None
    model_manager = None
    if cfg.lm_studio_url:
        from pr_checker.lm_studio_client import LMStudioClient
        from pr_checker.model_manager import ModelManager

        lm_studio = LMStudioClient(cfg.lm_studio_url, cfg.lm_studio_api_key)
        model_manager = ModelManager(lm_studio)

    indexer: CodeIndexer | None = None
    qdrant_gateway: QdrantGateway | None = None
    context_builder: ContextBuilder | None = None
    if cfg.qdrant_url:
        from qdrant_client import AsyncQdrantClient

        qdrant_gateway = QdrantGateway(AsyncQdrantClient(url=cfg.qdrant_url))
        embedding_service = EmbeddingService(openai_client, cfg.embedding_model)
        indexer = CodeIndexer(
            github=github,
            embedding=embedding_service,
            qdrant=qdrant_gateway,
            persistence=persistence,
            max_chars=cfg.embedding_max_tokens * 4,
            max_concurrent_files=cfg.embedding_max_concurrent_files,
        )
        context_builder = ContextBuilder(
            gateway=qdrant_gateway,
            embedding_service=embedding_service,
            max_prefetch_chunks=cfg.max_prefetch_chunks,
        )

    orchestrator = ReviewOrchestrator(
        github=github,
        config_manager=config_manager,
        standards_detector=standards_detector,
        issue_resolver=issue_resolver,
        model_manager=model_manager,
        openai=openai_client,
        model_override=cfg.llm_model or None,
        static_analyzer=StaticAnalyzer(),
        debug_dir=Path(cfg.llm_debug_dir) if cfg.llm_debug_dir else None,
        context_builder=context_builder,
    )
    queue = ReviewQueue(persistence, github, orchestrator)

    bg_tasks: set[asyncio.Task[None]] = set()

    def _fire(coro: Coroutine[Any, Any, None]) -> None:
        """Schedule *coro* as a background task, keeping a strong reference to prevent GC."""
        task: asyncio.Task[None] = asyncio.create_task(coro)
        bg_tasks.add(task)
        task.add_done_callback(bg_tasks.discard)

    # Tracks (repo, branch) pairs for which a full re-index is currently in flight,
    # used by the admin endpoint to reject duplicate concurrent requests.
    in_flight_indexes: set[tuple[str, str]] = set()

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
        if bg_tasks:
            # Drain in-flight index tasks before closing resources. Cancel any that
            # don't finish within the timeout to avoid use-after-close crashes.
            _done, pending = await asyncio.wait(bg_tasks, timeout=30.0)
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        await github.aclose()
        await openai_client.close()
        if lm_studio is not None:
            await lm_studio.aclose()
        if qdrant_gateway is not None:
            await qdrant_gateway.aclose()
        await persistence.dispose()

    application = FastAPI(title="pr-checker", lifespan=lifespan)
    application.state.queue = queue
    application.state.indexer = indexer

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

        # Return 204 early for events we never process, preserving the previous
        # behaviour for non-standard webhook sources that may send non-JSON bodies.
        if event not in ("pull_request", "push", "delete"):
            return Response(status_code=204)

        try:
            payload: dict[str, Any] = json.loads(payload_bytes)
        except json.JSONDecodeError:
            return Response(status_code=400, content="Invalid JSON")

        if event == "pull_request":
            return await _handle_pull_request(payload)

        if event == "push" and indexer is not None:
            _handle_push_background(payload, indexer)
            return Response(status_code=204)

        if event == "delete" and indexer is not None:
            _handle_delete_background(payload, indexer)
            return Response(status_code=204)

        return Response(status_code=204)

    async def _handle_pull_request(payload: dict[str, Any]) -> Response:
        if payload.get("action") == "closed":
            try:
                repo_name = str(payload["repository"]["full_name"])
                pr_num = int(payload["pull_request"]["number"])
                await queue.cancel_pr(repo_name, pr_num)
            except (KeyError, TypeError, ValueError):
                pass
            return Response(status_code=204)

        job = parse_pr_event(payload)
        if job is None:
            return Response(status_code=204)

        await queue.enqueue(job)
        return Response(status_code=202)

    def _handle_push_background(payload: dict[str, Any], idx: CodeIndexer) -> None:
        # NOTE: concurrent pushes to the same branch are not serialized; rapid successive
        # pushes can race on delete_file/upsert_chunks for the same (branch, file_path).
        # Acceptable for single-user local use; add per-(repo,branch) locking if needed.
        try:
            repo = str(payload["repository"]["full_name"])
            ref: str = payload.get("ref") or ""
            if not ref.startswith("refs/heads/"):
                return
            branch = ref[len("refs/heads/") :]
            sha = str(payload["after"])
            # A push event with deleted:true or the zero SHA means a branch deletion;
            # skip indexing — the separate "delete" event handles cleanup.
            if payload.get("deleted") or sha == _ZERO_SHA:
                return
            forced: bool = bool(payload.get("forced", False))
            commits: list[dict[str, Any]] = payload.get("commits", [])
            # distinct_size > len(commits) means GitHub omitted commits (history rewrite
            # or push too large); fall back to a full re-index in that case.
            distinct_size: int = int(payload.get("distinct_size", len(commits)))
        except (KeyError, TypeError, ValueError):
            return

        # distinct_size=0 with no listed commits means the push advanced nothing
        # (e.g. a re-push of the current HEAD); skip to avoid creating a task at all.
        if distinct_size == 0 and not commits:
            return

        async def _run() -> None:
            try:
                if forced or not commits or distinct_size > len(commits):
                    key = (repo, branch)
                    if key in in_flight_indexes:
                        _log.info(
                            "Push-triggered full re-index skipped: %s@%s already in-flight",
                            repo,
                            branch,
                        )
                        return
                    in_flight_indexes.add(key)
                    try:
                        await idx.index_branch(repo, branch, sha)
                    finally:
                        in_flight_indexes.discard(key)
                else:
                    # Skip incremental index if a full re-index is already running for
                    # this branch; the full re-index will produce a complete, consistent
                    # snapshot and any incremental delta would race its stale-gen sweep.
                    if (repo, branch) in in_flight_indexes:
                        _log.info(
                            "Incremental index skipped: full re-index in-flight for %s@%s",
                            repo,
                            branch,
                        )
                        return
                    added_or_modified, removed = aggregate_push_files(commits)
                    await idx.index_files(repo, branch, sha, added_or_modified, removed)
            except Exception:
                _log.exception("Background index failed for %s@%s", repo, branch)

        _fire(_run())

    def _handle_delete_background(payload: dict[str, Any], idx: CodeIndexer) -> None:
        try:
            if payload.get("ref_type") != "branch":
                return
            repo = str(payload["repository"]["full_name"])
            branch = str(payload["ref"])
        except (KeyError, TypeError):
            return

        async def _run() -> None:
            try:
                await idx.delete_branch(repo, branch)
            except Exception:
                _log.exception("Background branch delete failed for %s@%s", repo, branch)

        _fire(_run())

    # /admin/index is intentionally unauthenticated: this server is designed for local
    # use by the repository owner and is not meant to be exposed to the public internet.
    @application.post("/admin/index")
    async def admin_index(
        repo: str = Query(..., description="Repository in owner/repo format"),
        branch: str = Query(..., description="Branch name"),
    ) -> Response:
        if indexer is None:
            return Response(
                status_code=503,
                content=json.dumps(
                    {"error": "Indexing not configured (QDRANT_URL and EMBEDDING_MODEL required)"}
                ),
                media_type="application/json",
            )
        key = (repo, branch)
        if key in in_flight_indexes:
            return Response(
                status_code=409,
                content=json.dumps({"error": f"Index already in progress for {repo}@{branch}"}),
                media_type="application/json",
            )
        # Add to in_flight_indexes before the first await so concurrent requests
        # for the same (repo, branch) see it immediately and get a 409.
        in_flight_indexes.add(key)
        try:
            sha = await github.get_branch_sha(repo, branch)
        except httpx.HTTPStatusError as exc:
            in_flight_indexes.discard(key)
            if exc.response.status_code == 404:
                return Response(
                    status_code=404,
                    content=json.dumps({"error": f"Branch '{branch}' not found in '{repo}'"}),
                    media_type="application/json",
                )
            raise
        except BaseException:
            in_flight_indexes.discard(key)
            raise

        idx = indexer  # capture non-None reference for the closure

        async def _run() -> None:
            try:
                await idx.index_branch(repo, branch, sha)
            except Exception:
                _log.exception("Admin index failed for %s@%s", repo, branch)
            finally:
                in_flight_indexes.discard(key)

        _fire(_run())
        return Response(
            status_code=202,
            content=json.dumps({"repo": repo, "branch": branch, "sha": sha, "status": "accepted"}),
            media_type="application/json",
        )

    return application


app = create_app()
