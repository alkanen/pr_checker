import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from pr_checker.models import CodeChunk, CodeSnippet

_log = logging.getLogger(__name__)

_UPSERT_BATCH = 64


def collection_name(repo_full_name: str) -> str:
    return repo_full_name.replace("/", "__")


class QdrantGateway:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client
        self._known_collections: set[str] = set()

    async def upsert_chunks(
        self, chunks: list[CodeChunk], vectors: list[list[float]], sha: str
    ) -> None:
        if not chunks:
            return
        coll = collection_name(chunks[0].repo_full_name)
        dim = len(vectors[0])
        await self._ensure_collection(coll, dim)

        points = [
            PointStruct(
                id=chunk.point_id,
                vector=vector,
                payload={
                    "repo_full_name": chunk.repo_full_name,
                    "branch": chunk.branch,
                    "file_path": chunk.file_path,
                    "chunk_type": chunk.chunk_type,
                    "chunk_name": chunk.chunk_name,
                    "content": chunk.content,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "sha": sha,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        try:
            for i in range(0, len(points), _UPSERT_BATCH):
                await self._client.upsert(
                    collection_name=coll, points=points[i : i + _UPSERT_BATCH]
                )
        except UnexpectedResponse as exc:
            if exc.status_code != 404:
                raise
            # Collection deleted externally after our _ensure_collection call;
            # clear the cache, recreate, and retry once.
            self._known_collections.discard(coll)
            await self._ensure_collection(coll, dim)
            for i in range(0, len(points), _UPSERT_BATCH):
                await self._client.upsert(
                    collection_name=coll, points=points[i : i + _UPSERT_BATCH]
                )
        _log.debug("Upserted %d points into %s", len(points), coll)

    async def delete_file(self, repo_full_name: str, branch: str, file_path: str) -> None:
        """Delete all vectors for *file_path* on *branch* regardless of SHA."""
        coll = collection_name(repo_full_name)
        if coll not in self._known_collections and not await self._client.collection_exists(coll):
            return
        self._known_collections.add(coll)
        deleted = await self._safe_delete(
            coll,
            FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(key="branch", match=MatchValue(value=branch)),
                        FieldCondition(key="file_path", match=MatchValue(value=file_path)),
                    ]
                )
            ),
        )
        if deleted:
            _log.debug("Deleted points for %s@%s:%s", repo_full_name, branch, file_path)

    async def delete_stale_generations(
        self, repo_full_name: str, branch: str, current_sha: str
    ) -> None:
        """Delete all points for *branch* whose sha payload differs from *current_sha*."""
        coll = collection_name(repo_full_name)
        if coll not in self._known_collections and not await self._client.collection_exists(coll):
            return
        self._known_collections.add(coll)
        deleted = await self._safe_delete(
            coll,
            FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="branch", match=MatchValue(value=branch))],
                    must_not=[FieldCondition(key="sha", match=MatchValue(value=current_sha))],
                )
            ),
        )
        if deleted:
            _log.debug(
                "Deleted stale-generation points for %s@%s (current sha %.8s)",
                repo_full_name,
                branch,
                current_sha,
            )

    async def delete_branch(self, repo_full_name: str, branch: str) -> None:
        coll = collection_name(repo_full_name)
        if coll not in self._known_collections and not await self._client.collection_exists(coll):
            return
        self._known_collections.add(coll)
        deleted = await self._safe_delete(
            coll,
            FilterSelector(
                filter=Filter(must=[FieldCondition(key="branch", match=MatchValue(value=branch))])
            ),
        )
        if deleted:
            _log.info("Deleted all points for branch %s in %s", branch, repo_full_name)

    async def search(
        self,
        vector: list[float],
        repo_full_name: str,
        branch: str | None = None,
        limit: int = 10,
    ) -> list[CodeSnippet]:
        if not vector or limit < 1:
            return []

        coll = collection_name(repo_full_name)
        if coll not in self._known_collections and not await self._client.collection_exists(coll):
            return []
        self._known_collections.add(coll)

        query_filter: Filter | None = None
        if branch is not None:
            query_filter = Filter(
                must=[FieldCondition(key="branch", match=MatchValue(value=branch))]
            )

        try:
            response = await self._client.query_points(
                collection_name=coll,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        except UnexpectedResponse as exc:
            if exc.status_code != 404:
                raise
            self._known_collections.discard(coll)
            return []

        snippets: list[CodeSnippet] = []
        for pt in response.points:
            snippet = CodeSnippet.from_payload(pt.payload or {})
            snippet.score = pt.score
            snippets.append(snippet)
        return snippets

    async def aclose(self) -> None:
        await self._client.close()

    async def _safe_delete(self, coll: str, selector: FilterSelector) -> bool:
        """Run _client.delete; return False on 404 (collection deleted externally)."""
        try:
            await self._client.delete(collection_name=coll, points_selector=selector)
            return True
        except UnexpectedResponse as exc:
            if exc.status_code != 404:
                raise
            self._known_collections.discard(coll)
            return False

    async def _ensure_collection(self, coll: str, dim: int) -> None:
        if coll in self._known_collections:
            return
        if not await self._client.collection_exists(coll):
            try:
                await self._client.create_collection(
                    collection_name=coll,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
                _log.info("Created Qdrant collection %s (dim=%d)", coll, dim)
            except UnexpectedResponse as exc:
                # 400/409 = collection already exists, created by a concurrent coroutine
                # between our existence check and this call.  Any other status is a real error.
                if exc.status_code not in (400, 409):
                    raise
                if not await self._client.collection_exists(coll):
                    raise
        self._known_collections.add(coll)
