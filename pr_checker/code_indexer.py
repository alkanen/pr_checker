import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from pr_checker.code_chunker import chunk_file, is_indexable
from pr_checker.db import PersistenceLayer
from pr_checker.embedding_service import EmbeddingService
from pr_checker.github_client import GitHubClient
from pr_checker.qdrant_gateway import QdrantGateway

_log = logging.getLogger(__name__)


def aggregate_push_files(
    commits: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Aggregate added/modified/removed paths across all commits in a push event."""
    added_or_modified: set[str] = set()
    removed: set[str] = set()
    for commit in commits:
        for path in commit.get("added", []) + commit.get("modified", []):
            removed.discard(path)
            added_or_modified.add(path)
        for path in commit.get("removed", []):
            added_or_modified.discard(path)
            removed.add(path)
    return sorted(added_or_modified), sorted(removed)


class CodeIndexer:
    def __init__(
        self,
        github: GitHubClient,
        embedding: EmbeddingService,
        qdrant: QdrantGateway,
        persistence: PersistenceLayer,
        max_chars: int = 2048,
        max_concurrent_files: int = 5,
    ) -> None:
        self._github = github
        self._embedding = embedding
        self._qdrant = qdrant
        self._persistence = persistence
        self._max_chars = max_chars
        self._sem = asyncio.Semaphore(max_concurrent_files)

    async def index_branch(self, repo_full_name: str, branch: str, sha: str) -> int:
        """Full re-index of branch. Returns chunk count.

        Indexes all files first tagged with *sha*, then removes stale-generation
        points so the collection is never empty mid-reindex.  The tree is fetched
        at *sha* (not the mutable branch name) so the file list and file contents
        are always from the same commit.
        """
        _log.info("Full index start: %s@%s (sha %.8s)", repo_full_name, branch, sha)

        try:
            paths = await self._github.get_file_tree(repo_full_name, sha)
        except RuntimeError:
            _log.error(
                "Full index aborted for %s@%s: repository tree is too large for the "
                "GitHub recursive tree API (>100 000 entries). "
                "Index it manually file-by-file or increase the GitHub tree API limit.",
                repo_full_name,
                branch,
            )
            return 0
        except httpx.HTTPStatusError as exc:
            _log.error(
                "Full index aborted for %s@%s: HTTP %d from tree API",
                repo_full_name,
                branch,
                exc.response.status_code,
            )
            return 0

        indexable = [p for p in paths if is_indexable(p)]
        _log.info(
            "Full index: %d/%d files are indexable in %s@%s",
            len(indexable),
            len(paths),
            repo_full_name,
            branch,
        )

        # pre_delete=False: stale-generation cleanup after the upsert loop handles
        # removal of old points, so we never briefly empty the collection.
        total, had_failures = await self._index_file_list(
            repo_full_name, branch, sha, indexable, pre_delete=False
        )
        if had_failures:
            _log.warning(
                "Full index for %s@%s completed with failures — stale-generation cleanup "
                "and SHA persistence skipped to avoid data loss",
                repo_full_name,
                branch,
            )
        elif total > 0:
            # Only sweep stale points when at least one chunk was written with the
            # current SHA; an all-oversized or all-empty tree would delete the entire
            # existing index because no points carry the new SHA.
            await self._qdrant.delete_stale_generations(repo_full_name, branch, sha)
            await self._persistence.upsert_indexed_branch(
                repo_full_name, branch, sha, datetime.now(timezone.utc)
            )
            _log.info("Full index done: %s@%s — %d chunks", repo_full_name, branch, total)
        else:
            # No indexable content found (all files non-indexable or oversized).
            # Advance SHA so the next push uses the correct base, but skip
            # delete_stale_generations — no new points carry the current SHA so
            # that call would wipe the entire existing index.
            # WARNING: any previously indexed vectors for this branch are now stale
            # and will remain in Qdrant until a future full re-index produces content.
            await self._persistence.upsert_indexed_branch(
                repo_full_name, branch, sha, datetime.now(timezone.utc)
            )
            _log.warning(
                "Full index for %s@%s: no indexable content found — SHA advanced "
                "but existing stale vectors were preserved",
                repo_full_name,
                branch,
            )
        return total

    async def index_files(
        self,
        repo_full_name: str,
        branch: str,
        sha: str,
        added_or_modified: list[str],
        removed: list[str],
    ) -> int:
        """Incremental update for a push event. Returns chunk count upserted.

        The SHA is recorded even when no indexable files changed, to track that
        the branch has advanced to this commit.
        """
        _log.info(
            "Incremental index: %s@%s (sha %.8s) +%d -%d files",
            repo_full_name,
            branch,
            sha,
            len(added_or_modified),
            len(removed),
        )
        indexable_removed = [p for p in removed if is_indexable(p)]
        del_results = await asyncio.gather(
            *(self._qdrant.delete_file(repo_full_name, branch, path) for path in indexable_removed),
            return_exceptions=True,
        )
        had_delete_failures = any(isinstance(r, BaseException) for r in del_results)
        if had_delete_failures:
            _log.warning(
                "Failed to delete one or more removed files from index for %s@%s — "
                "run /admin/index to clean up ghost vectors",
                repo_full_name,
                branch,
            )

        indexable = [p for p in added_or_modified if is_indexable(p)]
        total, had_failures = await self._index_file_list(repo_full_name, branch, sha, indexable)
        if not had_failures and not had_delete_failures:
            await self._persistence.upsert_indexed_branch(
                repo_full_name, branch, sha, datetime.now(timezone.utc)
            )
        return total

    async def delete_branch(self, repo_full_name: str, branch: str) -> None:
        await self._qdrant.delete_branch(repo_full_name, branch)
        await self._persistence.delete_indexed_branch(repo_full_name, branch)
        _log.info("Deleted index for %s@%s", repo_full_name, branch)

    async def _index_file_list(
        self,
        repo_full_name: str,
        branch: str,
        sha: str,
        paths: list[str],
        pre_delete: bool = True,
    ) -> tuple[int, bool]:
        """Index *paths* concurrently and return ``(chunk_count, had_failures)``.

        When *pre_delete* is True (incremental path), existing points for each file are
        deleted before upserting new ones.  This ensures files that become empty, oversized,
        or non-indexable lose their stale vectors immediately.  Callers that rely on
        stale-generation cleanup for bulk removal (full re-index) pass ``pre_delete=False``.

        ``had_failures=True`` is returned when any individual file raised an exception.
        Callers that gate SHA persistence and stale-generation cleanup on a clean run will
        keep retrying until the failing file succeeds or disappears — this is intentional
        to avoid recording a partial index as complete.
        """
        # Instantiate coroutines in batches to bound peak memory on large trees.
        total_chunks = 0
        had_failures = False
        for i in range(0, len(paths), 100):
            batch = paths[i : i + 100]
            batch_results: list[tuple[int, bool]] = await asyncio.gather(
                *(self._index_one(repo_full_name, branch, sha, p, pre_delete) for p in batch)
            )
            total_chunks += sum(c for c, _ in batch_results)
            had_failures = had_failures or any(f for _, f in batch_results)
        return total_chunks, had_failures

    async def _index_one(
        self,
        repo_full_name: str,
        branch: str,
        sha: str,
        path: str,
        pre_delete: bool,
    ) -> tuple[int, bool]:
        async with self._sem:
            try:
                # Pre-delete runs before fetch+embed so oversized and empty files also
                # lose their stale vectors immediately.  Trade-off: if any step from
                # get_file_content onward raises (network error, 429, embed timeout),
                # the file has zero vectors until the next push retries.
                # The alternative (post-upsert stale delete) was reverted because it
                # created a worse concurrent-push race that could also zero out a file.
                if pre_delete:
                    await self._qdrant.delete_file(repo_full_name, branch, path)
                source = await self._github.get_file_content(repo_full_name, path, sha)
                if source is None:
                    _log.debug("Skipping oversized file %s", path)
                    return 0, False
                chunks = chunk_file(repo_full_name, branch, path, source, self._max_chars)
                if not chunks:
                    return 0, False
                vectors = await self._embedding.embed([c.content for c in chunks])
                await self._qdrant.upsert_chunks(chunks, vectors, sha)
                return len(chunks), False
            except Exception:
                _log.warning(
                    "Failed to index %s@%s:%s — skipping",
                    repo_full_name,
                    branch,
                    path,
                    exc_info=True,
                )
                return 0, True
