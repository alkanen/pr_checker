from unittest.mock import AsyncMock

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import QueryResponse, ScoredPoint

from pr_checker.models import CodeSnippet
from pr_checker.qdrant_gateway import QdrantGateway, collection_name


def _make_scored_point(
    file_path: str = "src/main.py",
    chunk_name: str = "do_stuff",
    chunk_type: str = "function",
    content: str = "def do_stuff(): ...",
    start_line: int = 1,
    end_line: int = 1,
    branch: str = "main",
    score: float = 0.95,
) -> ScoredPoint:
    return ScoredPoint(
        id=1,
        version=0,
        score=score,
        payload={
            "repo_full_name": "owner/repo",
            "branch": branch,
            "file_path": file_path,
            "chunk_type": chunk_type,
            "chunk_name": chunk_name,
            "content": content,
            "start_line": start_line,
            "end_line": end_line,
            "sha": "abc123",
        },
    )


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.collection_exists = AsyncMock(return_value=True)
    client.query_points = AsyncMock(return_value=QueryResponse(points=[]))
    return client


@pytest.fixture
def gateway(mock_client: AsyncMock) -> QdrantGateway:
    return QdrantGateway(mock_client)


class TestSearch:
    async def test_returns_snippets_from_payload(
        self, gateway: QdrantGateway, mock_client: AsyncMock
    ) -> None:
        point = _make_scored_point(
            file_path="src/utils.py",
            chunk_name="helper",
            chunk_type="function",
            content="def helper(): pass",
            start_line=10,
            end_line=15,
            branch="feature",
        )
        mock_client.query_points.return_value = QueryResponse(points=[point])

        result = await gateway.search([0.1, 0.2], "owner/repo", limit=5)

        assert len(result) == 1
        s = result[0]
        assert isinstance(s, CodeSnippet)
        assert s.file_path == "src/utils.py"
        assert s.chunk_name == "helper"
        assert s.chunk_type == "function"
        assert s.content == "def helper(): pass"
        assert s.start_line == 10
        assert s.end_line == 15
        assert s.branch == "feature"
        assert s.score == 0.95

    async def test_preserves_relevance_order(
        self, gateway: QdrantGateway, mock_client: AsyncMock
    ) -> None:
        points = [
            _make_scored_point(chunk_name="best", score=0.99),
            _make_scored_point(chunk_name="good", score=0.80),
            _make_scored_point(chunk_name="okay", score=0.60),
        ]
        mock_client.query_points.return_value = QueryResponse(points=points)

        result = await gateway.search([0.1], "owner/repo")

        assert [s.chunk_name for s in result] == ["best", "good", "okay"]

    async def test_branch_filter_passed_to_query(
        self, gateway: QdrantGateway, mock_client: AsyncMock
    ) -> None:
        mock_client.query_points.return_value = QueryResponse(points=[])

        await gateway.search([0.1], "owner/repo", branch="develop")

        call_kwargs = mock_client.query_points.call_args.kwargs
        qf = call_kwargs["query_filter"]
        assert qf is not None
        assert len(qf.must) == 1
        assert qf.must[0].key == "branch"
        assert qf.must[0].match.value == "develop"

    async def test_no_branch_filter_when_none(
        self, gateway: QdrantGateway, mock_client: AsyncMock
    ) -> None:
        mock_client.query_points.return_value = QueryResponse(points=[])

        await gateway.search([0.1], "owner/repo", branch=None)

        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is None

    async def test_empty_list_when_collection_missing(
        self, gateway: QdrantGateway, mock_client: AsyncMock
    ) -> None:
        mock_client.collection_exists.return_value = False

        result = await gateway.search([0.1], "owner/repo")

        assert result == []
        mock_client.query_points.assert_not_called()

    async def test_empty_list_on_404_unexpected_response(
        self, gateway: QdrantGateway, mock_client: AsyncMock
    ) -> None:
        gateway._known_collections.add(collection_name("owner/repo"))
        mock_client.query_points.side_effect = UnexpectedResponse(
            status_code=404, reason_phrase="Not Found", content=b"", headers=httpx.Headers()
        )

        result = await gateway.search([0.1], "owner/repo")

        assert result == []
        assert collection_name("owner/repo") not in gateway._known_collections

    async def test_non_404_unexpected_response_raises(
        self, gateway: QdrantGateway, mock_client: AsyncMock
    ) -> None:
        gateway._known_collections.add(collection_name("owner/repo"))
        mock_client.query_points.side_effect = UnexpectedResponse(
            status_code=500, reason_phrase="Internal", content=b"", headers=httpx.Headers()
        )

        with pytest.raises(UnexpectedResponse):
            await gateway.search([0.1], "owner/repo")

    async def test_limit_forwarded(self, gateway: QdrantGateway, mock_client: AsyncMock) -> None:
        mock_client.query_points.return_value = QueryResponse(points=[])

        await gateway.search([0.1], "owner/repo", limit=3)

        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["limit"] == 3

    async def test_uses_collection_name_helper(
        self, gateway: QdrantGateway, mock_client: AsyncMock
    ) -> None:
        mock_client.query_points.return_value = QueryResponse(points=[])

        await gateway.search([0.1], "org/my-repo")

        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["collection_name"] == "org__my-repo"

    async def test_skips_collection_exists_when_known(
        self, gateway: QdrantGateway, mock_client: AsyncMock
    ) -> None:
        gateway._known_collections.add(collection_name("owner/repo"))
        mock_client.query_points.return_value = QueryResponse(points=[])

        await gateway.search([0.1], "owner/repo")

        mock_client.collection_exists.assert_not_called()

    async def test_empty_vector_returns_early(
        self, gateway: QdrantGateway, mock_client: AsyncMock
    ) -> None:
        result = await gateway.search([], "owner/repo")

        assert result == []
        mock_client.query_points.assert_not_called()
        mock_client.collection_exists.assert_not_called()

    async def test_zero_limit_returns_early(
        self, gateway: QdrantGateway, mock_client: AsyncMock
    ) -> None:
        result = await gateway.search([0.1], "owner/repo", limit=0)

        assert result == []
        mock_client.query_points.assert_not_called()
        mock_client.collection_exists.assert_not_called()


class TestCodeSnippetFromPayload:
    def test_maps_all_fields(self) -> None:
        payload = {
            "file_path": "src/lib.py",
            "chunk_name": "parse",
            "chunk_type": "function",
            "content": "def parse(): ...",
            "start_line": 5,
            "end_line": 10,
            "branch": "main",
        }
        s = CodeSnippet.from_payload(payload)
        assert s.file_path == "src/lib.py"
        assert s.chunk_name == "parse"
        assert s.chunk_type == "function"
        assert s.content == "def parse(): ..."
        assert s.start_line == 5
        assert s.end_line == 10
        assert s.branch == "main"

    def test_defaults_on_empty_payload(self) -> None:
        s = CodeSnippet.from_payload({})
        assert s.file_path == ""
        assert s.chunk_name == ""
        assert s.chunk_type == "module_block"
        assert s.content == ""
        assert s.start_line == 1
        assert s.end_line == 1
        assert s.branch == ""

    def test_invalid_chunk_type_falls_back_to_default(self) -> None:
        s = CodeSnippet.from_payload({"chunk_type": "garbage"})
        assert s.chunk_type == "module_block"
