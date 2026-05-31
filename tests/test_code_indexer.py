import os
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from pr_checker.code_indexer import CodeIndexer, aggregate_push_files
from pr_checker.db import PersistenceLayer
from pr_checker.embedding_service import EmbeddingService
from pr_checker.github_client import GitHubClient
from pr_checker.qdrant_gateway import QdrantGateway

_REPO = "owner/repo"
_BRANCH = "main"
_SHA = "abc123def456"

_SIMPLE_PY = "def hello():\n    return 'hi'\n"
_SIMPLE_GO = "\n".join(f"// line {i}" for i in range(1, 10))


@pytest.fixture
async def persistence() -> AsyncGenerator[PersistenceLayer, None]:
    layer = PersistenceLayer("sqlite+aiosqlite:///:memory:")
    await layer.init()
    yield layer
    await layer.dispose()


@pytest.fixture
def mock_github() -> AsyncMock:
    mock = AsyncMock(spec=GitHubClient)
    mock.get_file_tree.return_value = ["pr_checker/queue.py", "README.md", "main.go"]
    mock.get_file_content.return_value = _SIMPLE_PY
    return mock


@pytest.fixture
def mock_embedding() -> AsyncMock:
    mock = AsyncMock(spec=EmbeddingService)
    mock.embed.return_value = [[0.1, 0.2, 0.3]]
    return mock


@pytest.fixture
def mock_qdrant() -> AsyncMock:
    return AsyncMock(spec=QdrantGateway)


@pytest.fixture
def indexer(
    mock_github: AsyncMock,
    mock_embedding: AsyncMock,
    mock_qdrant: AsyncMock,
    persistence: PersistenceLayer,
) -> CodeIndexer:
    return CodeIndexer(
        github=cast(GitHubClient, mock_github),
        embedding=cast(EmbeddingService, mock_embedding),
        qdrant=cast(QdrantGateway, mock_qdrant),
        persistence=persistence,
    )


# --- index_branch ---


async def test_index_branch_cleans_stale_generations(
    indexer: CodeIndexer, mock_qdrant: AsyncMock
) -> None:
    await indexer.index_branch(_REPO, _BRANCH, _SHA)
    mock_qdrant.delete_stale_generations.assert_awaited_once_with(_REPO, _BRANCH, _SHA)
    mock_qdrant.delete_branch.assert_not_awaited()


async def test_index_branch_skips_non_indexable_files(
    indexer: CodeIndexer, mock_github: AsyncMock, mock_qdrant: AsyncMock
) -> None:
    mock_github.get_file_tree.return_value = ["app.py", "README.md", "config.json"]
    mock_github.get_file_content.return_value = _SIMPLE_PY
    mock_qdrant.upsert_chunks = AsyncMock()

    await indexer.index_branch(_REPO, _BRANCH, _SHA)

    fetched_paths = [c.args[1] for c in mock_github.get_file_content.call_args_list]
    assert "README.md" not in fetched_paths
    assert "config.json" not in fetched_paths
    assert "app.py" in fetched_paths


async def test_index_branch_embeds_and_upserts(
    indexer: CodeIndexer, mock_embedding: AsyncMock, mock_qdrant: AsyncMock
) -> None:
    await indexer.index_branch(_REPO, _BRANCH, _SHA)
    assert mock_embedding.embed.await_count >= 1
    assert mock_qdrant.upsert_chunks.await_count >= 1


async def test_index_branch_updates_persistence(
    indexer: CodeIndexer, persistence: PersistenceLayer
) -> None:
    await indexer.index_branch(_REPO, _BRANCH, _SHA)
    record = await persistence.get_indexed_branch(_REPO, _BRANCH)
    assert record is not None
    assert record.last_indexed_sha == _SHA


async def test_index_branch_skips_oversized_file(
    indexer: CodeIndexer, mock_github: AsyncMock, mock_qdrant: AsyncMock
) -> None:
    mock_github.get_file_content.return_value = None
    await indexer.index_branch(_REPO, _BRANCH, _SHA)
    mock_qdrant.upsert_chunks.assert_not_awaited()


async def test_index_branch_swallows_per_file_errors(
    indexer: CodeIndexer, mock_github: AsyncMock, mock_qdrant: AsyncMock
) -> None:
    mock_github.get_file_content.side_effect = Exception("network error")
    count = await indexer.index_branch(_REPO, _BRANCH, _SHA)
    assert count == 0
    mock_qdrant.upsert_chunks.assert_not_awaited()


async def test_index_branch_skips_stale_cleanup_on_failure(
    indexer: CodeIndexer, mock_github: AsyncMock, mock_qdrant: AsyncMock
) -> None:
    mock_github.get_file_content.side_effect = Exception("network error")
    await indexer.index_branch(_REPO, _BRANCH, _SHA)
    mock_qdrant.delete_stale_generations.assert_not_awaited()


# --- incremental stale-vector cleanup ---


async def test_index_files_cleans_up_oversized_file(
    indexer: CodeIndexer, mock_github: AsyncMock, mock_qdrant: AsyncMock
) -> None:
    mock_github.get_file_content.return_value = None  # oversized
    await indexer.index_files(_REPO, _BRANCH, _SHA, ["big.py"], [])
    mock_qdrant.delete_file.assert_awaited_with(_REPO, _BRANCH, "big.py")


async def test_index_files_cleans_up_empty_chunk_file(
    indexer: CodeIndexer, mock_github: AsyncMock, mock_qdrant: AsyncMock
) -> None:
    mock_github.get_file_content.return_value = ""  # produces no chunks
    await indexer.index_files(_REPO, _BRANCH, _SHA, ["empty.py"], [])
    mock_qdrant.delete_file.assert_awaited_with(_REPO, _BRANCH, "empty.py")


async def test_index_files_pre_deletes_before_upsert(
    indexer: CodeIndexer, mock_github: AsyncMock, mock_qdrant: AsyncMock
) -> None:
    mock_github.get_file_content.return_value = _SIMPLE_PY
    order: list[str] = []

    async def record_delete(*_: Any, **__: Any) -> None:
        order.append("delete")

    async def record_upsert(*_: Any, **__: Any) -> None:
        order.append("upsert")

    mock_qdrant.delete_file.side_effect = record_delete
    mock_qdrant.upsert_chunks.side_effect = record_upsert
    await indexer.index_files(_REPO, _BRANCH, _SHA, ["app.py"], [])
    assert order == ["delete", "upsert"]


# --- index_files (incremental) ---


async def test_index_files_deletes_removed_files(
    indexer: CodeIndexer, mock_qdrant: AsyncMock
) -> None:
    await indexer.index_files(_REPO, _BRANCH, _SHA, [], ["old_file.py"])
    mock_qdrant.delete_file.assert_awaited_once_with(_REPO, _BRANCH, "old_file.py")


async def test_index_files_upserts_added_files(
    indexer: CodeIndexer, mock_github: AsyncMock, mock_qdrant: AsyncMock
) -> None:
    mock_github.get_file_content.return_value = _SIMPLE_PY
    await indexer.index_files(_REPO, _BRANCH, _SHA, ["new_file.py"], [])
    mock_qdrant.upsert_chunks.assert_awaited()


async def test_index_files_updates_persistence(
    indexer: CodeIndexer, persistence: PersistenceLayer
) -> None:
    await indexer.index_files(_REPO, _BRANCH, _SHA, [], [])
    record = await persistence.get_indexed_branch(_REPO, _BRANCH)
    assert record is not None
    assert record.last_indexed_sha == _SHA


# --- delete_branch ---


async def test_delete_branch_removes_qdrant_points(
    indexer: CodeIndexer, mock_qdrant: AsyncMock
) -> None:
    await indexer.delete_branch(_REPO, _BRANCH)
    mock_qdrant.delete_branch.assert_awaited_once_with(_REPO, _BRANCH)


async def test_delete_branch_removes_db_record(
    indexer: CodeIndexer, persistence: PersistenceLayer
) -> None:
    await persistence.upsert_indexed_branch(_REPO, _BRANCH, _SHA, datetime.now(timezone.utc))
    await indexer.delete_branch(_REPO, _BRANCH)
    assert await persistence.get_indexed_branch(_REPO, _BRANCH) is None


# --- _aggregate_push_files ---


def test_aggregate_push_files_merges_commits() -> None:
    commits: list[dict[str, Any]] = [
        {"added": ["a.py"], "modified": [], "removed": []},
        {"added": [], "modified": ["b.py"], "removed": ["c.py"]},
    ]
    added_or_modified, removed = aggregate_push_files(commits)
    assert "a.py" in added_or_modified
    assert "b.py" in added_or_modified
    assert "c.py" in removed
    assert "c.py" not in added_or_modified


def test_aggregate_push_files_file_readded_after_removal() -> None:
    commits: list[dict[str, Any]] = [
        {"added": [], "modified": [], "removed": ["a.py"]},
        {"added": ["a.py"], "modified": [], "removed": []},
    ]
    added_or_modified, removed = aggregate_push_files(commits)
    assert "a.py" in added_or_modified
    assert "a.py" not in removed


def test_aggregate_push_files_empty_commits() -> None:
    added_or_modified, removed = aggregate_push_files([])
    assert added_or_modified == []
    assert removed == []


# --- system test (skipped if QDRANT_URL absent) ---


@pytest.mark.skipif(
    not os.environ.get("QDRANT_URL"),
    reason="QDRANT_URL not set",
)
async def test_system_full_index_and_verify() -> None:
    from openai import AsyncOpenAI
    from qdrant_client import AsyncQdrantClient

    qdrant_url = os.environ["QDRANT_URL"]
    embedding_model = os.environ.get(
        "EMBEDDING_MODEL", "text-embedding-nomic-ai-nomic-embed-text-v2-moe"
    )
    openai_base = os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")
    openai_key = os.environ.get("OPENAI_API_KEY", "sk-placeholder")
    github_token = os.environ.get("GITHUB_TOKEN", "")

    openai_client = AsyncOpenAI(api_key=openai_key, base_url=openai_base)
    qdrant_client = AsyncQdrantClient(url=qdrant_url)
    qdrant = QdrantGateway(qdrant_client)
    github = GitHubClient(github_token)

    layer = PersistenceLayer("sqlite+aiosqlite:///:memory:")
    await layer.init()

    test_repo = os.environ.get("TEST_REPO", "alkanen/pr_checker")
    test_branch = os.environ.get("TEST_BRANCH", "main")

    idx = CodeIndexer(
        github=github,
        embedding=EmbeddingService(openai_client, embedding_model),
        qdrant=qdrant,
        persistence=layer,
    )
    try:
        head_sha = await github.get_branch_sha(test_repo, test_branch)
        count = await idx.index_branch(test_repo, test_branch, head_sha)
        assert count > 0, "Expected at least one chunk to be indexed"

        record = await layer.get_indexed_branch(test_repo, test_branch)
        assert record is not None

        from pr_checker.qdrant_gateway import collection_name

        info = await qdrant_client.get_collection(collection_name(test_repo))
        assert info.points_count is not None
        assert info.points_count > 0
    finally:
        await qdrant.delete_branch(test_repo, test_branch)
        await layer.dispose()
        await github.aclose()
        await openai_client.close()
        await qdrant_client.close()
