from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from pr_checker.models import DiffHunk

_log = logging.getLogger(__name__)

_PYTHON_KEYWORDS = frozenset(
    {
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    }
)

_CPP_KEYWORDS = frozenset(
    {
        "alignas",
        "alignof",
        "auto",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "class",
        "const",
        "constexpr",
        "continue",
        "decltype",
        "default",
        "delete",
        "do",
        "double",
        "else",
        "enum",
        "explicit",
        "extern",
        "false",
        "float",
        "for",
        "friend",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "nullptr",
        "operator",
        "private",
        "protected",
        "public",
        "register",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "struct",
        "switch",
        "template",
        "this",
        "throw",
        "true",
        "try",
        "typedef",
        "typeid",
        "typename",
        "union",
        "unsigned",
        "using",
        "virtual",
        "void",
        "volatile",
        "while",
    }
)

_PYTHON_BUILTINS = frozenset(
    {
        "abs",
        "all",
        "any",
        "bin",
        "bool",
        "bytes",
        "callable",
        "chr",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "globals",
        "hasattr",
        "hash",
        "hex",
        "id",
        "input",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "locals",
        "map",
        "max",
        "min",
        "next",
        "object",
        "oct",
        "open",
        "ord",
        "pow",
        "print",
        "property",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "setattr",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "vars",
        "zip",
    }
)

_CPP_BUILTINS = frozenset(
    {
        "std",
        "cout",
        "cin",
        "cerr",
        "endl",
        "string",
        "vector",
        "map",
        "set",
        "pair",
        "make_pair",
        "make_shared",
        "make_unique",
        "shared_ptr",
        "unique_ptr",
        "printf",
        "scanf",
        "malloc",
        "free",
        "memcpy",
        "memset",
    }
)

_STOP_WORDS = _PYTHON_KEYWORDS | _CPP_KEYWORDS | _PYTHON_BUILTINS | _CPP_BUILTINS

# Regex patterns applied to changed lines
_RE_FUNC_DEF_PY = re.compile(r"\bdef\s+(\w+)")
_RE_FUNC_DEF_CPP = re.compile(
    r"\b(?:void|int|float|double|bool|char|auto|string|size_t)\s+(\w+)\s*\("
)
_RE_CLASS_DEF = re.compile(r"\bclass\s+(\w+)")
_RE_FUNC_CALL = re.compile(r"\b(\w+)\s*\(")
_RE_METHOD_CALL = re.compile(r"\.(\w+)\s*\(")
_RE_SCOPE_RESOLUTION = re.compile(r"(\w+)::")
_RE_ISINSTANCE_ARGS = re.compile(r"isinstance\(\w+,\s*(.+?)\)\s*")
_RE_DOTTED_NAME = re.compile(r"[\w.]+")
_RE_IMPORT_PY = re.compile(r"(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))")
_RE_INCLUDE_CPP = re.compile(r'#include\s*[<"](.+?)[>"]')
_RE_TYPE_ANNOTATION_PY = re.compile(r"(?::\s*(\w+)|(?:->)\s*(\w+))")
_RE_DECORATOR = re.compile(r"@(\w+)")

IdentifierKind = str

_RANK: dict[str, int] = {
    "function_def": 0,
    "class_ref": 0,
    "function_call": 1,
    "method_call": 1,
    "import": 2,
    "type_annotation": 2,
    "decorator": 1,
    "scope_resolution": 1,
    "include": 2,
}


@dataclass
class ExtractedIdentifier:
    name: str
    source_file: str
    kind: IdentifierKind


def _dedup_and_rank(results: list[ExtractedIdentifier]) -> list[ExtractedIdentifier]:
    seen: dict[str, ExtractedIdentifier] = {}
    for ident in results:
        existing = seen.get(ident.name)
        if existing is None or _RANK.get(ident.kind, 99) < _RANK.get(existing.kind, 99):
            seen[ident.name] = ident
    deduped = list(seen.values())
    deduped.sort(key=lambda i: (_RANK.get(i.kind, 99), i.name))
    return deduped


def extract_identifiers(hunks: list[DiffHunk]) -> list[ExtractedIdentifier]:
    by_file: dict[str, list[str]] = {}
    for hunk in hunks:
        changed = [line.content for line in hunk.lines if line.kind in ("+", "-")]
        by_file.setdefault(hunk.file_path, []).extend(changed)

    results: list[ExtractedIdentifier] = []
    for file_path, lines in by_file.items():
        results.extend(_extract_regex(file_path, lines))

    return _dedup_and_rank(results)


def extract_identifiers_with_ast(
    hunks: list[DiffHunk],
    file_sources: dict[str, str],
) -> list[ExtractedIdentifier]:
    by_file: dict[str, list[str]] = {}
    changed_lines_by_file: dict[str, set[int]] = {}
    for hunk in hunks:
        for line in hunk.lines:
            if line.kind == "+" and line.new_lineno is not None:
                changed_lines_by_file.setdefault(hunk.file_path, set()).add(line.new_lineno)
        changed = [line.content for line in hunk.lines if line.kind in ("+", "-")]
        by_file.setdefault(hunk.file_path, []).extend(changed)

    results: list[ExtractedIdentifier] = []
    for file_path, lines in by_file.items():
        results.extend(_extract_regex(file_path, lines))
        if file_path.endswith(".py") and file_path in file_sources:
            ast_results = _extract_ast(
                file_path,
                file_sources[file_path],
                changed_lines_by_file.get(file_path, set()),
            )
            results.extend(ast_results)

    return _dedup_and_rank(results)


def _is_noise(name: str) -> bool:
    if len(name) <= 1:
        return True
    if name in _STOP_WORDS:
        return True
    if name.startswith("__") and name.endswith("__"):
        return True
    return False


def _add(
    out: list[ExtractedIdentifier],
    file_path: str,
    kind: str,
    name: str,
) -> None:
    if not _is_noise(name):
        out.append(ExtractedIdentifier(name=name, source_file=file_path, kind=kind))


_PY_EXTS = frozenset({".py", ".pyi"})
_CPP_EXTS = frozenset({".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"})


def _extract_regex(file_path: str, lines: list[str]) -> list[ExtractedIdentifier]:
    ext = PurePosixPath(file_path).suffix.lower()
    is_python = ext in _PY_EXTS
    is_cpp = ext in _CPP_EXTS

    out: list[ExtractedIdentifier] = []
    for line in lines:
        for m in _RE_CLASS_DEF.finditer(line):
            _add(out, file_path, "class_ref", m.group(1))
        for m in _RE_METHOD_CALL.finditer(line):
            _add(out, file_path, "method_call", m.group(1))
        for m in _RE_FUNC_CALL.finditer(line):
            name = m.group(1)
            if name not in ("def", "class"):
                _add(out, file_path, "function_call", name)

        if is_python:
            for m in _RE_FUNC_DEF_PY.finditer(line):
                _add(out, file_path, "function_def", m.group(1))
            for m in _RE_DECORATOR.finditer(line):
                _add(out, file_path, "decorator", m.group(1))
            for m in _RE_ISINSTANCE_ARGS.finditer(line):
                args_str = m.group(1).strip().strip("()")
                for name_match in _RE_DOTTED_NAME.findall(args_str):
                    _add(out, file_path, "class_ref", name_match)
            for m in _RE_IMPORT_PY.finditer(line):
                val = m.group(1) or m.group(2)
                if val:
                    _add(out, file_path, "import", val)
            for m in _RE_TYPE_ANNOTATION_PY.finditer(line):
                val = m.group(1) or m.group(2)
                if val:
                    _add(out, file_path, "type_annotation", val)

        if is_cpp:
            for m in _RE_FUNC_DEF_CPP.finditer(line):
                _add(out, file_path, "function_def", m.group(1))
            for m in _RE_SCOPE_RESOLUTION.finditer(line):
                _add(out, file_path, "scope_resolution", m.group(1))
            for m in _RE_INCLUDE_CPP.finditer(line):
                _add(out, file_path, "include", m.group(1))
    return out


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _extract_ast(
    file_path: str,
    source: str,
    changed_lines: set[int],
) -> list[ExtractedIdentifier]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        _log.debug("AST parse failed for %s; skipping AST refinement", file_path)
        return []

    out: list[ExtractedIdentifier] = []
    imports: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                local = alias.asname or alias.name
                imports[local] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name
                imports[local] = alias.name

    for node in ast.walk(tree):
        if not hasattr(node, "lineno"):
            continue
        if node.lineno not in changed_lines:
            continue

        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name and not _is_noise(name):
                if name in imports:
                    _add(out, file_path, "function_call", name)
                    mod = imports[name]
                    _add(out, file_path, "import", mod)
                else:
                    _add(out, file_path, "function_call", name)

    return out
