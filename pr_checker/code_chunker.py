import ast
import logging
from pathlib import Path
from typing import Literal

from pr_checker.models import CodeChunk

_log = logging.getLogger(__name__)

_PYTHON_EXT = ".py"
_SOURCE_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".mjs",
    ".cjs",
    ".go",
    ".rs",
    ".java",
    ".rb",
    ".cs",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".kt",
    ".swift",
    ".scala",
    ".php",
    ".lua",
}

_CHARS_PER_LINE = 80  # assumed average for source code; used only for heuristic chunker sizing
_OVERLAP_FRACTION = 6  # overlap ≈ chunk_lines // _OVERLAP_FRACTION (~16%)

ChunkType = Literal["function", "class", "module_block"]

_DecorableNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def _end(node: ast.stmt) -> int:
    return node.end_lineno if node.end_lineno is not None else node.lineno


def is_indexable(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in _SOURCE_EXTS


def chunk_file(
    repo_full_name: str,
    branch: str,
    file_path: str,
    source: str,
    max_chars: int = 2048,
) -> list[CodeChunk]:
    """Chunk a source file into indexable units, each fitting within *max_chars*.

    Python files are split at top-level AST boundaries (functions, classes, and
    groups of consecutive module-level statements).  All other supported types use a
    sliding-window line split sized from *max_chars*.  Oversized Python AST chunks are
    then passed through :func:`split_to_fit` so the returned list uniformly respects
    the budget.

    Line endings are normalised to ``\\n`` (CRLF sources are stored with LF).
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == _PYTHON_EXT:
        try:
            raw = _chunk_python(repo_full_name, branch, file_path, source)
            return [sc for c in raw for sc in split_to_fit(c, max_chars)]
        except SyntaxError:
            _log.debug("SyntaxError in %s; falling back to heuristic chunker", file_path)
    if suffix in _SOURCE_EXTS:
        return [
            sc
            for c in _chunk_heuristic(repo_full_name, branch, file_path, source, max_chars)
            for sc in split_to_fit(c, max_chars)
        ]
    return []


def split_to_fit(chunk: CodeChunk, max_chars: int) -> list[CodeChunk]:
    """Split *chunk* into overlapping sub-chunks so each fits within *max_chars*.

    Character counts are accumulated line-by-line (including the '\\n' separator),
    so the guarantee holds even for files with highly variable line lengths.  A chunk
    that already fits is returned unchanged.  A single line longer than *max_chars*
    is emitted as-is with a warning; truncation is left to the embedding model.

    Sub-chunks are named ``original_name[1]``, ``original_name[2]``, … so their
    point IDs remain unique.
    """
    if len(chunk.content) <= max_chars:
        return [chunk]

    lines = chunk.content.splitlines()
    if len(lines) <= 1:
        _log.warning(
            "Single-line chunk in %s exceeds max_chars=%d; embedding model will truncate",
            chunk.file_path,
            max_chars,
        )
        return [chunk]

    result: list[CodeChunk] = []
    i = 0
    idx = 1

    while i < len(lines):
        sub_lines: list[str] = []
        char_count = 0
        j = i

        while j < len(lines):
            line = lines[j]
            # First line in a sub-chunk has no leading '\n'; subsequent lines do.
            extra = len(line) + (1 if sub_lines else 0)
            if sub_lines and char_count + extra > max_chars:
                break
            sub_lines.append(line)
            char_count += extra
            j += 1

        if len(sub_lines) == 1 and char_count > max_chars:
            _log.warning(
                "Sub-chunk line in %s exceeds max_chars=%d (%d chars); "
                "embedding model will truncate",
                chunk.file_path,
                max_chars,
                char_count,
            )

        end = i + len(sub_lines)
        result.append(
            CodeChunk(
                repo_full_name=chunk.repo_full_name,
                branch=chunk.branch,
                file_path=chunk.file_path,
                chunk_type=chunk.chunk_type,
                chunk_name=f"{chunk.chunk_name}[{idx}]",
                content="\n".join(sub_lines),
                start_line=chunk.start_line + i,
                end_line=chunk.start_line + end - 1,
            )
        )
        if j >= len(lines):
            break

        overlap = max(1, len(sub_lines) // _OVERLAP_FRACTION)
        i = max(i + 1, j - overlap)
        idx += 1

    return result


def _chunk_python(repo_full_name: str, branch: str, file_path: str, source: str) -> list[CodeChunk]:
    tree = ast.parse(source)
    lines = source.splitlines()
    chunks: list[CodeChunk] = []
    module_group: list[ast.stmt] = []

    def _flush_module_group() -> None:
        if not module_group:
            return
        start = module_group[0].lineno
        end = max(_end(n) for n in module_group)
        # Use the full contiguous slice so start_line/end_line match content exactly.
        # Blank lines between statements within the group are included.
        chunks.append(
            CodeChunk(
                repo_full_name=repo_full_name,
                branch=branch,
                file_path=file_path,
                chunk_type="module_block",
                chunk_name="<module>",
                content="\n".join(lines[start - 1 : end]),
                start_line=start,
                end_line=end,
            )
        )
        module_group.clear()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _flush_module_group()
            chunks.append(
                _make_chunk(repo_full_name, branch, file_path, "function", node.name, lines, node)
            )
        elif isinstance(node, ast.ClassDef):
            _flush_module_group()
            chunks.append(
                _make_chunk(repo_full_name, branch, file_path, "class", node.name, lines, node)
            )
        elif hasattr(node, "lineno"):
            module_group.append(node)

    _flush_module_group()
    return chunks


def _make_chunk(
    repo_full_name: str,
    branch: str,
    file_path: str,
    chunk_type: ChunkType,
    name: str,
    lines: list[str],
    node: _DecorableNode,
) -> CodeChunk:
    # Include decorators in the chunk; node.lineno points at the def/class keyword.
    start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    end = _end(node)
    return CodeChunk(
        repo_full_name=repo_full_name,
        branch=branch,
        file_path=file_path,
        chunk_type=chunk_type,
        chunk_name=name,
        content="\n".join(lines[start - 1 : end]),
        start_line=start,
        end_line=end,
    )


def _chunk_heuristic(
    repo_full_name: str,
    branch: str,
    file_path: str,
    source: str,
    max_chars: int,
) -> list[CodeChunk]:
    lines = source.splitlines()
    if not lines:
        return []

    max_lines = max(2, max_chars // _CHARS_PER_LINE)
    overlap = max(1, max_lines // _OVERLAP_FRACTION)
    step = max(1, max_lines - overlap)
    total = len(lines)
    chunks: list[CodeChunk] = []
    start = 0

    while start < total:
        end = min(start + max_lines, total)
        chunks.append(
            CodeChunk(
                repo_full_name=repo_full_name,
                branch=branch,
                file_path=file_path,
                chunk_type="module_block",
                chunk_name=f"lines_{start + 1}_{end}",
                content="\n".join(lines[start:end]),
                start_line=start + 1,
                end_line=end,
            )
        )
        if end == total:
            break
        start += step

    return chunks
