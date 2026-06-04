from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

from pr_checker.embedding_service import EmbeddingService
from pr_checker.identifier_extractor import ExtractedIdentifier
from pr_checker.models import CodeSnippet
from pr_checker.qdrant_gateway import QdrantGateway

_log = logging.getLogger(__name__)

_PER_QUERY_LIMIT = 3
_MAX_CONCURRENT_SEARCHES = 10
BRANCH_BOTH = "both"


def _rstrip_content(content: str) -> str:
    return "\n".join(line.rstrip() for line in content.splitlines())


def _deduplicate(snippets: list[CodeSnippet], base_branch: str) -> list[CodeSnippet]:
    best: dict[tuple[str, str, str], CodeSnippet] = {}
    for s in snippets:
        key = (s.file_path, s.chunk_name, s.branch)
        existing = best.get(key)
        if existing is None or s.score > existing.score:
            best[key] = s

    by_identity: dict[tuple[str, str], list[CodeSnippet]] = {}
    for s in best.values():
        k = (s.file_path, s.chunk_name)
        by_identity.setdefault(k, []).append(s)

    result: list[CodeSnippet] = []
    for group in by_identity.values():
        if len(group) == 1:
            result.append(group[0])
        elif len(group) == 2 and _rstrip_content(group[0].content) == _rstrip_content(
            group[1].content
        ):
            base = next((s for s in group if s.branch == base_branch), group[0])
            result.append(
                replace(
                    base,
                    branch=BRANCH_BOTH,
                    score=max(group[0].score, group[1].score),
                )
            )
        else:
            result.extend(group)

    return result


class ContextBuilder:
    def __init__(
        self,
        gateway: QdrantGateway,
        embedding_service: EmbeddingService,
        max_prefetch_chunks: int = 16,
    ) -> None:
        self._gateway = gateway
        self._embedding = embedding_service
        self._max_prefetch_chunks = max_prefetch_chunks
        self._sem = asyncio.Semaphore(_MAX_CONCURRENT_SEARCHES)

    async def build(
        self,
        identifiers: list[ExtractedIdentifier],
        repo_full_name: str,
        base_branch: str,
        head_branch: str,
    ) -> list[CodeSnippet]:
        if not identifiers:
            _log.info("No identifiers extracted; skipping RAG context")
            return []

        queries = [ident.name for ident in identifiers]
        try:
            vectors = await self._embedding.embed(queries)
        except Exception:
            _log.exception("Embedding failed; skipping RAG context")
            return []

        branches = [base_branch, head_branch] if base_branch != head_branch else [base_branch]
        coros = [
            self._throttled_search(vector, repo_full_name, branch)
            for vector in vectors
            for branch in branches
        ]
        search_results = await asyncio.gather(*coros, return_exceptions=True)

        raw: list[CodeSnippet] = []
        for r in search_results:
            if isinstance(r, BaseException):
                _log.warning("Search query failed: %s", r)
                continue
            raw.extend(r)

        if not raw:
            _log.info("No results from Qdrant for %s", repo_full_name)
            return []

        deduped = _deduplicate(raw, base_branch)
        deduped.sort(key=lambda s: s.score, reverse=True)
        return deduped[: self._max_prefetch_chunks]

    async def search(
        self,
        query: str,
        repo_full_name: str,
        branch: str,
        limit: int = 5,
    ) -> list[CodeSnippet]:
        # Single query, no fan-out — semaphore not needed.
        try:
            vectors = await self._embedding.embed([query])
        except Exception:
            _log.exception("Embedding failed for search query")
            return []

        if not vectors:
            return []

        return await self._gateway.search(vectors[0], repo_full_name, branch=branch, limit=limit)

    async def _throttled_search(
        self, vector: list[float], repo_full_name: str, branch: str
    ) -> list[CodeSnippet]:
        async with self._sem:
            return await self._gateway.search(
                vector, repo_full_name, branch=branch, limit=_PER_QUERY_LIMIT
            )
