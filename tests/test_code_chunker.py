from pathlib import Path

import pytest

from pr_checker.code_chunker import chunk_file, is_indexable, split_to_fit
from pr_checker.models import CodeChunk

_REPO = "owner/repo"
_BRANCH = "main"
_FIXTURE = Path(__file__).parent / "fixtures" / "sample_module.py"


def _source() -> str:
    return _FIXTURE.read_text()


# --- is_indexable ---


def test_python_file_is_indexable() -> None:
    assert is_indexable("pr_checker/queue.py")


def test_typescript_file_is_indexable() -> None:
    assert is_indexable("src/App.tsx")


def test_markdown_is_not_indexable() -> None:
    assert not is_indexable("README.md")


def test_json_is_not_indexable() -> None:
    assert not is_indexable("package.json")


# --- Python AST chunker ---


def test_python_standalone_function_gets_own_chunk() -> None:
    chunks = chunk_file(_REPO, _BRANCH, "sample.py", _source())
    names = {c.chunk_name for c in chunks}
    assert "standalone_function" in names
    assert "async_function" in names


def test_python_class_gets_own_chunk() -> None:
    chunks = chunk_file(_REPO, _BRANCH, "sample.py", _source())
    names = {c.chunk_name for c in chunks}
    assert "MyClass" in names
    assert "AnotherClass" in names


def test_python_module_block_captures_imports_and_constants() -> None:
    chunks = chunk_file(_REPO, _BRANCH, "sample.py", _source())
    module_chunks = [c for c in chunks if c.chunk_type == "module_block"]
    # Two groups: top-level imports/constants and the trailing MODULE_LEVEL_DICT.
    assert len(module_chunks) == 2
    all_content = "\n".join(c.content for c in module_chunks)
    assert "import os" in all_content
    assert "CONSTANT" in all_content


def test_python_chunk_line_numbers_are_correct() -> None:
    chunks = chunk_file(_REPO, _BRANCH, "sample.py", _source())
    fn_chunk = next(c for c in chunks if c.chunk_name == "standalone_function")
    assert fn_chunk.start_line >= 1
    assert fn_chunk.end_line >= fn_chunk.start_line
    # Content must match the lines in the source
    source_lines = _source().splitlines()
    expected = "\n".join(source_lines[fn_chunk.start_line - 1 : fn_chunk.end_line])
    assert fn_chunk.content == expected


def test_python_chunk_metadata() -> None:
    chunks = chunk_file(_REPO, _BRANCH, "sample.py", _source())
    for chunk in chunks:
        assert chunk.repo_full_name == _REPO
        assert chunk.branch == _BRANCH
        assert chunk.file_path == "sample.py"


def test_python_chunk_point_ids_are_unique() -> None:
    chunks = chunk_file(_REPO, _BRANCH, "sample.py", _source())
    ids = [c.point_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_python_empty_file_returns_no_chunks() -> None:
    assert chunk_file(_REPO, _BRANCH, "empty.py", "") == []


def test_python_syntax_error_falls_back_to_heuristic() -> None:
    chunks = chunk_file(_REPO, _BRANCH, "bad.py", "def broken(\n  this is not python\n")
    assert len(chunks) >= 1
    assert all(c.chunk_type == "module_block" for c in chunks)


def test_python_only_functions_no_module_block() -> None:
    source = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
    chunks = chunk_file(_REPO, _BRANCH, "funcs.py", source)
    assert not any(c.chunk_type == "module_block" for c in chunks)
    assert {c.chunk_name for c in chunks} == {"foo", "bar"}


def test_python_decorator_included_in_function_chunk() -> None:
    source = "@staticmethod\ndef foo():\n    return 1\n"
    chunks = chunk_file(_REPO, _BRANCH, "dec.py", source)
    fn = next(c for c in chunks if c.chunk_name == "foo")
    assert fn.start_line == 1
    assert "@staticmethod" in fn.content


def test_python_decorator_not_in_module_block() -> None:
    source = "x = 1\n\n@staticmethod\ndef foo():\n    return 1\n"
    chunks = chunk_file(_REPO, _BRANCH, "dec2.py", source)
    module_chunks = [c for c in chunks if c.chunk_type == "module_block"]
    assert len(module_chunks) == 1
    assert "@staticmethod" not in module_chunks[0].content


# --- heuristic chunker ---


def test_heuristic_chunks_go_file() -> None:
    source = "\n".join(f"line {i}" for i in range(1, 101))
    chunks = chunk_file(_REPO, _BRANCH, "main.go", source)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.chunk_type == "module_block"
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line


def test_heuristic_chunk_overlap() -> None:
    source = "\n".join(f"line {i}" for i in range(1, 101))
    chunks = chunk_file(_REPO, _BRANCH, "main.go", source)
    # second chunk should start before end of first (overlap)
    assert chunks[1].start_line < chunks[0].end_line


def test_heuristic_empty_file_returns_no_chunks() -> None:
    assert chunk_file(_REPO, _BRANCH, "empty.go", "") == []


def test_unsupported_extension_returns_no_chunks() -> None:
    assert chunk_file(_REPO, _BRANCH, "data.csv", "a,b,c\n1,2,3\n") == []


# --- cross-repo / cross-branch isolation ---


@pytest.mark.parametrize(
    "repo,branch",
    [
        ("owner/repo-a", "main"),
        ("owner/repo-b", "main"),
        ("owner/repo-a", "feature"),
    ],
)
def test_point_ids_differ_across_repos_and_branches(repo: str, branch: str) -> None:
    chunks_a = chunk_file("owner/repo-a", "main", "f.py", "def foo():\n    pass\n")
    chunks_b = chunk_file(repo, branch, "f.py", "def foo():\n    pass\n")
    if repo == "owner/repo-a" and branch == "main":
        assert chunks_a[0].point_id == chunks_b[0].point_id
    else:
        assert chunks_a[0].point_id != chunks_b[0].point_id


# --- split_to_fit ---


def _big_chunk(lines: int = 200, chars_per_line: int = 80) -> CodeChunk:
    content = "\n".join("x" * chars_per_line for _ in range(lines))
    return CodeChunk(
        repo_full_name=_REPO,
        branch=_BRANCH,
        file_path="big.py",
        chunk_type="class",
        chunk_name="BigClass",
        content=content,
        start_line=1,
        end_line=lines,
    )


def test_split_to_fit_small_chunk_unchanged() -> None:
    chunk = _big_chunk(lines=5)
    result = split_to_fit(chunk, max_chars=10_000)
    assert result == [chunk]


def test_split_to_fit_large_chunk_produces_multiple() -> None:
    chunk = _big_chunk(lines=200, chars_per_line=80)
    result = split_to_fit(chunk, max_chars=800)  # ~10 lines each
    assert len(result) > 1


def test_split_to_fit_all_sub_chunks_fit() -> None:
    chunk = _big_chunk(lines=200, chars_per_line=80)
    max_chars = 1600  # ~20 lines
    result = split_to_fit(chunk, max_chars=max_chars)
    for sub in result:
        assert len(sub.content) <= max_chars


def test_split_to_fit_sub_chunks_overlap() -> None:
    chunk = _big_chunk(lines=100, chars_per_line=80)
    result = split_to_fit(chunk, max_chars=1600)
    assert len(result) > 1
    # Second chunk starts before first chunk ends (overlap)
    assert result[1].start_line < result[0].end_line


def test_split_to_fit_names_are_indexed() -> None:
    chunk = _big_chunk(lines=200, chars_per_line=80)
    result = split_to_fit(chunk, max_chars=800)
    for i, sub in enumerate(result, start=1):
        assert sub.chunk_name == f"BigClass[{i}]"


def test_split_to_fit_point_ids_are_unique() -> None:
    chunk = _big_chunk(lines=200, chars_per_line=80)
    result = split_to_fit(chunk, max_chars=800)
    ids = [c.point_id for c in result]
    assert len(ids) == len(set(ids))


def test_split_to_fit_line_numbers_are_correct() -> None:
    chunk = _big_chunk(lines=100, chars_per_line=80)
    result = split_to_fit(chunk, max_chars=1600)
    assert result[0].start_line == 1
    assert result[-1].end_line == 100
    for sub in result:
        assert sub.start_line <= sub.end_line


def test_split_to_fit_preserves_metadata() -> None:
    chunk = _big_chunk(lines=200, chars_per_line=80)
    result = split_to_fit(chunk, max_chars=800)
    for sub in result:
        assert sub.repo_full_name == _REPO
        assert sub.branch == _BRANCH
        assert sub.file_path == "big.py"
        assert sub.chunk_type == "class"


def test_split_to_fit_single_line_returned_as_is() -> None:
    chunk = CodeChunk(
        repo_full_name=_REPO,
        branch=_BRANCH,
        file_path="f.py",
        chunk_type="module_block",
        chunk_name="<module>",
        content="x" * 5000,
        start_line=1,
        end_line=1,
    )
    result = split_to_fit(chunk, max_chars=100)
    assert result == [chunk]


def test_split_to_fit_covers_all_content() -> None:
    """Every line of the original chunk appears in at least one sub-chunk."""
    lines = [f"line_{i}" for i in range(100)]
    content = "\n".join(lines)
    chunk = CodeChunk(
        repo_full_name=_REPO,
        branch=_BRANCH,
        file_path="f.go",
        chunk_type="module_block",
        chunk_name="lines_1_100",
        content=content,
        start_line=1,
        end_line=100,
    )
    result = split_to_fit(chunk, max_chars=len("line_xx\n") * 15)
    all_content = "\n".join(c.content for c in result)
    for line in lines:
        assert line in all_content
