from unittest.mock import AsyncMock

import pytest

from pr_checker.context_builder import (
    BRANCH_BOTH,
    ContextBuilder,
    _deduplicate,
    _rstrip_content,
)
from pr_checker.identifier_extractor import ExtractedIdentifier
from pr_checker.models import CodeSnippet


def _snippet(
    file_path: str = "src/main.py",
    chunk_name: str = "do_stuff",
    content: str = "def do_stuff(): ...",
    branch: str = "main",
    score: float = 0.9,
    start_line: int = 1,
    end_line: int = 1,
) -> CodeSnippet:
    return CodeSnippet(
        file_path=file_path,
        chunk_name=chunk_name,
        chunk_type="function",
        content=content,
        start_line=start_line,
        end_line=end_line,
        branch=branch,
        score=score,
    )


def _ident(name: str, source_file: str = "a.py") -> ExtractedIdentifier:
    return ExtractedIdentifier(name=name, source_file=source_file, kind="function_call")


def _mock_gateway() -> AsyncMock:
    gw = AsyncMock()
    gw.search = AsyncMock(return_value=[])
    return gw


def _mock_embedding() -> AsyncMock:
    emb = AsyncMock()
    emb.embed = AsyncMock(return_value=[])
    return emb


@pytest.fixture
def gateway() -> AsyncMock:
    return _mock_gateway()


@pytest.fixture
def embedding() -> AsyncMock:
    return _mock_embedding()


@pytest.fixture
def builder(gateway: AsyncMock, embedding: AsyncMock) -> ContextBuilder:
    return ContextBuilder(gateway, embedding, max_prefetch_chunks=16)


class TestRstripContent:
    def test_strips_trailing_whitespace(self) -> None:
        assert _rstrip_content("foo  \nbar\t\n") == "foo\nbar"

    def test_preserves_leading_whitespace(self) -> None:
        assert _rstrip_content("  foo\n    bar") == "  foo\n    bar"

    def test_empty_string(self) -> None:
        assert _rstrip_content("") == ""


class TestDeduplicate:
    def test_same_branch_keeps_highest_score(self) -> None:
        snippets = [
            _snippet(chunk_name="fn", branch="main", score=0.7),
            _snippet(chunk_name="fn", branch="main", score=0.9),
        ]
        result = _deduplicate(snippets, "main")
        assert len(result) == 1
        assert result[0].score == 0.9

    def test_cross_branch_identical_content_merges(self) -> None:
        snippets = [
            _snippet(chunk_name="fn", branch="main", content="def fn(): pass", score=0.8),
            _snippet(chunk_name="fn", branch="feat", content="def fn(): pass", score=0.6),
        ]
        result = _deduplicate(snippets, "main")
        assert len(result) == 1
        assert result[0].branch == BRANCH_BOTH
        assert result[0].score == 0.8

    def test_cross_branch_different_content_keeps_both(self) -> None:
        snippets = [
            _snippet(chunk_name="fn", branch="main", content="def fn(): pass", score=0.8),
            _snippet(chunk_name="fn", branch="feat", content="def fn(): return 1", score=0.6),
        ]
        result = _deduplicate(snippets, "main")
        assert len(result) == 2
        branches = {s.branch for s in result}
        assert branches == {"main", "feat"}

    def test_trailing_whitespace_ignored_in_comparison(self) -> None:
        snippets = [
            _snippet(chunk_name="fn", branch="main", content="def fn():  \n  pass", score=0.8),
            _snippet(chunk_name="fn", branch="feat", content="def fn():\n  pass", score=0.6),
        ]
        result = _deduplicate(snippets, "main")
        assert len(result) == 1
        assert result[0].branch == BRANCH_BOTH

    def test_leading_whitespace_difference_keeps_both(self) -> None:
        snippets = [
            _snippet(chunk_name="fn", branch="main", content="  def fn(): pass", score=0.8),
            _snippet(chunk_name="fn", branch="feat", content="    def fn(): pass", score=0.6),
        ]
        result = _deduplicate(snippets, "main")
        assert len(result) == 2

    def test_merged_snippet_uses_max_score(self) -> None:
        snippets = [
            _snippet(chunk_name="fn", branch="main", content="x", score=0.5),
            _snippet(chunk_name="fn", branch="feat", content="x", score=0.9),
        ]
        result = _deduplicate(snippets, "main")
        assert len(result) == 1
        assert result[0].score == 0.9

    def test_merged_snippet_uses_base_branch_line_numbers(self) -> None:
        snippets = [
            _snippet(
                chunk_name="fn",
                branch="main",
                content="x",
                score=0.5,
                start_line=10,
                end_line=20,
            ),
            _snippet(
                chunk_name="fn",
                branch="feat",
                content="x",
                score=0.9,
                start_line=50,
                end_line=60,
            ),
        ]
        result = _deduplicate(snippets, "main")
        assert len(result) == 1
        assert result[0].start_line == 10
        assert result[0].end_line == 20

    def test_different_files_not_merged(self) -> None:
        snippets = [
            _snippet(file_path="a.py", chunk_name="fn", branch="main", content="x", score=0.8),
            _snippet(file_path="b.py", chunk_name="fn", branch="feat", content="x", score=0.6),
        ]
        result = _deduplicate(snippets, "main")
        assert len(result) == 2

    def test_different_chunk_names_not_merged(self) -> None:
        snippets = [
            _snippet(chunk_name="fn_a", branch="main", content="x", score=0.8),
            _snippet(chunk_name="fn_b", branch="feat", content="x", score=0.6),
        ]
        result = _deduplicate(snippets, "main")
        assert len(result) == 2


class TestBuild:
    async def test_empty_identifiers_returns_empty(self, builder: ContextBuilder) -> None:
        result = await builder.build([], "owner/repo", "main", "feat")
        assert result == []

    async def test_embedding_failure_returns_empty(
        self, gateway: AsyncMock, embedding: AsyncMock
    ) -> None:
        embedding.embed.side_effect = RuntimeError("api down")
        cb = ContextBuilder(gateway, embedding)
        result = await cb.build([_ident("process")], "owner/repo", "main", "feat")
        assert result == []

    async def test_unindexed_repo_returns_empty(
        self, gateway: AsyncMock, embedding: AsyncMock
    ) -> None:
        embedding.embed.return_value = [[0.1, 0.2]]
        gateway.search.return_value = []
        cb = ContextBuilder(gateway, embedding)
        result = await cb.build([_ident("process")], "owner/repo", "main", "feat")
        assert result == []

    async def test_searches_both_branches(self, gateway: AsyncMock, embedding: AsyncMock) -> None:
        embedding.embed.return_value = [[0.1, 0.2]]
        gateway.search.return_value = []
        cb = ContextBuilder(gateway, embedding)

        await cb.build([_ident("process")], "owner/repo", "main", "feat")

        branches_searched = [call.kwargs["branch"] for call in gateway.search.call_args_list]
        assert "main" in branches_searched
        assert "feat" in branches_searched

    async def test_same_branch_searched_once(
        self, gateway: AsyncMock, embedding: AsyncMock
    ) -> None:
        embedding.embed.return_value = [[0.1]]
        gateway.search.return_value = []
        cb = ContextBuilder(gateway, embedding)

        await cb.build([_ident("fn")], "owner/repo", "main", "main")

        assert gateway.search.call_count == 1

    async def test_returns_snippets_from_both_branches(
        self, gateway: AsyncMock, embedding: AsyncMock
    ) -> None:
        embedding.embed.return_value = [[0.1]]
        base_snippet = _snippet(chunk_name="base_fn", branch="main", score=0.8)
        head_snippet = _snippet(chunk_name="head_fn", branch="feat", score=0.7)

        async def search_side_effect(
            vector: list[float], repo: str, branch: str | None = None, limit: int = 10
        ) -> list[CodeSnippet]:
            if branch == "main":
                return [base_snippet]
            return [head_snippet]

        gateway.search.side_effect = search_side_effect
        cb = ContextBuilder(gateway, embedding)

        result = await cb.build([_ident("fn")], "owner/repo", "main", "feat")

        names = {s.chunk_name for s in result}
        assert "base_fn" in names
        assert "head_fn" in names

    async def test_deduplicates_across_branches(
        self, gateway: AsyncMock, embedding: AsyncMock
    ) -> None:
        embedding.embed.return_value = [[0.1]]

        async def search_side_effect(
            vector: list[float], repo: str, branch: str | None = None, limit: int = 10
        ) -> list[CodeSnippet]:
            return [
                _snippet(
                    chunk_name="shared",
                    content="def shared(): pass",
                    branch=branch or "",
                    score=0.8,
                )
            ]

        gateway.search.side_effect = search_side_effect
        cb = ContextBuilder(gateway, embedding)

        result = await cb.build([_ident("shared")], "owner/repo", "main", "feat")

        assert len(result) == 1
        assert result[0].branch == BRANCH_BOTH

    async def test_ranked_by_score_descending(
        self, gateway: AsyncMock, embedding: AsyncMock
    ) -> None:
        embedding.embed.return_value = [[0.1], [0.2]]

        async def search_side_effect(
            vector: list[float], repo: str, branch: str | None = None, limit: int = 10
        ) -> list[CodeSnippet]:
            if vector == [0.1] and branch == "main":
                return [_snippet(chunk_name="low", content="low", branch="main", score=0.3)]
            if vector == [0.2] and branch == "main":
                return [_snippet(chunk_name="high", content="high", branch="main", score=0.9)]
            return []

        gateway.search.side_effect = search_side_effect
        cb = ContextBuilder(gateway, embedding)

        result = await cb.build([_ident("low"), _ident("high")], "owner/repo", "main", "feat")

        assert len(result) == 2
        assert result[0].chunk_name == "high"
        assert result[1].chunk_name == "low"

    async def test_trims_to_budget(self, gateway: AsyncMock, embedding: AsyncMock) -> None:
        embedding.embed.return_value = [[0.1]]

        async def search_side_effect(
            vector: list[float], repo: str, branch: str | None = None, limit: int = 10
        ) -> list[CodeSnippet]:
            return [
                _snippet(
                    chunk_name=f"fn_{branch}_{i}",
                    content=f"content_{branch}_{i}",
                    branch=branch or "",
                    score=0.9 - i * 0.1,
                )
                for i in range(5)
            ]

        gateway.search.side_effect = search_side_effect
        cb = ContextBuilder(gateway, embedding, max_prefetch_chunks=3)

        result = await cb.build([_ident("fn")], "owner/repo", "main", "feat")

        assert len(result) <= 3

    async def test_limit_overrides_default_budget(
        self, gateway: AsyncMock, embedding: AsyncMock
    ) -> None:
        embedding.embed.return_value = [[0.1]]

        async def search_side_effect(
            vector: list[float], repo: str, branch: str | None = None, limit: int = 10
        ) -> list[CodeSnippet]:
            return [
                _snippet(
                    chunk_name=f"fn_{branch}_{i}",
                    content=f"content_{branch}_{i}",
                    branch=branch or "",
                    score=0.9 - i * 0.1,
                )
                for i in range(5)
            ]

        gateway.search.side_effect = search_side_effect
        cb = ContextBuilder(gateway, embedding, max_prefetch_chunks=16)

        result = await cb.build([_ident("fn")], "owner/repo", "main", "feat", limit=2)

        assert len(result) <= 2

    async def test_embeds_identifier_names(self, gateway: AsyncMock, embedding: AsyncMock) -> None:
        embedding.embed.return_value = [[0.1], [0.2]]
        gateway.search.return_value = []
        cb = ContextBuilder(gateway, embedding)

        await cb.build([_ident("process"), _ident("validate")], "owner/repo", "main", "feat")

        embedding.embed.assert_called_once_with(["process", "validate"])

    async def test_per_query_limit_forwarded(
        self, gateway: AsyncMock, embedding: AsyncMock
    ) -> None:
        embedding.embed.return_value = [[0.1]]
        gateway.search.return_value = []
        cb = ContextBuilder(gateway, embedding)

        await cb.build([_ident("fn")], "owner/repo", "main", "feat")

        for call in gateway.search.call_args_list:
            assert call.kwargs["limit"] == 3


class TestSearch:
    async def test_returns_gateway_results(self, gateway: AsyncMock, embedding: AsyncMock) -> None:
        embedding.embed.return_value = [[0.1, 0.2]]
        expected = [_snippet(chunk_name="result")]
        gateway.search.return_value = expected
        cb = ContextBuilder(gateway, embedding)

        result = await cb.search("find this", "owner/repo", "main", limit=5)

        assert result == expected

    async def test_passes_branch_and_limit(self, gateway: AsyncMock, embedding: AsyncMock) -> None:
        embedding.embed.return_value = [[0.1]]
        gateway.search.return_value = []
        cb = ContextBuilder(gateway, embedding)

        await cb.search("query", "owner/repo", "develop", limit=3)

        gateway.search.assert_called_once_with([0.1], "owner/repo", branch="develop", limit=3)

    async def test_embedding_failure_propagates(
        self, gateway: AsyncMock, embedding: AsyncMock
    ) -> None:
        embedding.embed.side_effect = RuntimeError("api down")
        cb = ContextBuilder(gateway, embedding)

        with pytest.raises(RuntimeError, match="api down"):
            await cb.search("query", "owner/repo", "main")

        gateway.search.assert_not_called()

    async def test_embeds_query_string(self, gateway: AsyncMock, embedding: AsyncMock) -> None:
        embedding.embed.return_value = [[0.5]]
        gateway.search.return_value = []
        cb = ContextBuilder(gateway, embedding)

        await cb.search("ReviewContext", "owner/repo", "main")

        embedding.embed.assert_called_once_with(["ReviewContext"])
