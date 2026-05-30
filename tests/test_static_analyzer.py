"""Tests for StaticAnalyzer: ruff + mypy pre-pass before LLM review."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from pr_checker.static_analyzer import StaticAnalyzer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNUSED_IMPORT = "import os\n\nx = 1\n"
_TYPE_ERROR = "def add(a: int, b: int) -> str:\n    return a + b\n"
_CLEAN = "def greet(name: str) -> str:\n    return f'Hello, {name}'\n"


def _fake_process(stdout: bytes, returncode: int = 0, stderr: bytes = b"") -> MagicMock:
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    proc.returncode = returncode
    return proc


# ---------------------------------------------------------------------------
# StaticAnalyzer.run() — input filtering
# ---------------------------------------------------------------------------


async def test_run_empty_dict_returns_empty() -> None:
    findings = await StaticAnalyzer().run({})
    assert findings == []


async def test_run_non_python_files_returns_empty() -> None:
    findings = await StaticAnalyzer().run({"index.ts": "const x = 1;\n"})
    assert findings == []


async def test_run_both_disabled_returns_empty() -> None:
    findings = await StaticAnalyzer().run({"foo.py": _CLEAN}, run_ruff=False, run_mypy=False)
    assert findings == []


async def test_run_skips_path_traversal() -> None:
    findings = await StaticAnalyzer().run(
        {"../escape.py": _UNUSED_IMPORT, "safe.py": _UNUSED_IMPORT},
        run_ruff=True,
        run_mypy=False,
    )
    paths = {f.file_path for f in findings}
    assert not any(".." in p for p in paths)
    assert any(p == "safe.py" for p in paths)


async def test_run_skips_absolute_paths() -> None:
    findings = await StaticAnalyzer().run(
        {"/etc/passwd.py": _UNUSED_IMPORT, "safe.py": _UNUSED_IMPORT},
        run_ruff=True,
        run_mypy=False,
    )
    paths = {f.file_path for f in findings}
    assert not any(p.startswith("/") for p in paths)
    assert any(p == "safe.py" for p in paths)


# ---------------------------------------------------------------------------
# ruff — real tool runs
# ---------------------------------------------------------------------------


async def test_ruff_detects_unused_import() -> None:
    findings = await StaticAnalyzer().run(
        {"check_me.py": _UNUSED_IMPORT}, run_ruff=True, run_mypy=False
    )
    assert any(f.tool == "ruff" and f.code == "F401" for f in findings)


async def test_ruff_clean_file_returns_no_findings() -> None:
    findings = await StaticAnalyzer().run({"clean.py": _CLEAN}, run_ruff=True, run_mypy=False)
    ruff_findings = [f for f in findings if f.tool == "ruff"]
    assert ruff_findings == []


async def test_ruff_finding_has_correct_relative_path() -> None:
    findings = await StaticAnalyzer().run(
        {"pkg/module.py": _UNUSED_IMPORT}, run_ruff=True, run_mypy=False
    )
    ruff = [f for f in findings if f.tool == "ruff"]
    assert ruff
    assert all(f.file_path == "pkg/module.py" for f in ruff)


async def test_ruff_finding_has_line_and_code() -> None:
    findings = await StaticAnalyzer().run({"x.py": _UNUSED_IMPORT}, run_ruff=True, run_mypy=False)
    f401 = next((f for f in findings if f.tool == "ruff" and f.code == "F401"), None)
    assert f401 is not None
    assert f401.line is not None
    assert f401.line >= 1
    assert f401.message != ""


# ---------------------------------------------------------------------------
# mypy — real tool runs
# ---------------------------------------------------------------------------


async def test_mypy_detects_return_type_error() -> None:
    findings = await StaticAnalyzer().run({"typed.py": _TYPE_ERROR}, run_ruff=False, run_mypy=True)
    mypy = [f for f in findings if f.tool == "mypy"]
    assert mypy, "expected mypy to find the return type mismatch"


async def test_mypy_clean_file_returns_no_findings() -> None:
    findings = await StaticAnalyzer().run({"clean.py": _CLEAN}, run_ruff=False, run_mypy=True)
    mypy = [f for f in findings if f.tool == "mypy"]
    assert mypy == []


async def test_mypy_finding_has_correct_relative_path() -> None:
    findings = await StaticAnalyzer().run(
        {"pkg/typed.py": _TYPE_ERROR}, run_ruff=False, run_mypy=True
    )
    mypy = [f for f in findings if f.tool == "mypy"]
    assert mypy
    assert all(f.file_path == "pkg/typed.py" for f in mypy)


# ---------------------------------------------------------------------------
# Config file injection
# ---------------------------------------------------------------------------


async def test_mypy_config_passed_as_config_file_flag() -> None:
    captured: list[tuple[str, ...]] = []

    async def fake_exec(*args: str, **kwargs: object) -> MagicMock:
        captured.append(args)
        stdout = b""
        return _fake_process(stdout)

    with patch("pr_checker.static_analyzer.asyncio.create_subprocess_exec", side_effect=fake_exec):
        await StaticAnalyzer().run(
            {"foo.py": _CLEAN},
            run_ruff=False,
            run_mypy=True,
            mypy_config={"strict": True},
        )

    assert captured, "mypy was not invoked"
    mypy_cmd = captured[0]
    assert "--config-file" in mypy_cmd
    assert "--ignore-missing-imports" in mypy_cmd


async def test_mypy_no_config_does_not_pass_config_file_flag() -> None:
    captured: list[tuple[str, ...]] = []

    async def fake_exec(*args: str, **kwargs: object) -> MagicMock:
        captured.append(args)
        return _fake_process(b"")

    with patch("pr_checker.static_analyzer.asyncio.create_subprocess_exec", side_effect=fake_exec):
        await StaticAnalyzer().run(
            {"foo.py": _CLEAN},
            run_ruff=False,
            run_mypy=True,
        )

    assert captured
    mypy_cmd = captured[0]
    assert "--config-file" not in mypy_cmd
    assert "--ignore-missing-imports" in mypy_cmd


async def test_mypy_config_with_list_of_dicts_does_not_raise() -> None:
    """mypy.overrides is a list[dict]; _toml_scalar must handle it without silently skipping."""
    captured: list[tuple[str, ...]] = []

    async def fake_exec(*args: str, **kwargs: object) -> MagicMock:
        captured.append(args)
        return _fake_process(b"")

    mypy_config = {
        "strict": True,
        "overrides": [{"module": "some_untyped_lib.*", "ignore_missing_imports": True}],
    }
    with patch("pr_checker.static_analyzer.asyncio.create_subprocess_exec", side_effect=fake_exec):
        await StaticAnalyzer().run(
            {"foo.py": _CLEAN},
            run_ruff=False,
            run_mypy=True,
            mypy_config=mypy_config,
        )

    assert captured, "mypy was not invoked"
    assert "--config-file" in captured[0]


# ---------------------------------------------------------------------------
# Error handling — ruff
# ---------------------------------------------------------------------------


async def test_ruff_not_found_returns_empty() -> None:
    with patch(
        "pr_checker.static_analyzer.asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError,
    ):
        findings = await StaticAnalyzer().run(
            {"foo.py": _UNUSED_IMPORT}, run_ruff=True, run_mypy=False
        )
    assert findings == []


async def test_ruff_not_installed_returns_empty() -> None:
    proc = _fake_process(b"", returncode=1, stderr=b"/path/python: No module named ruff\n")
    with patch("pr_checker.static_analyzer.asyncio.create_subprocess_exec", return_value=proc):
        findings = await StaticAnalyzer().run(
            {"foo.py": _UNUSED_IMPORT}, run_ruff=True, run_mypy=False
        )
    assert findings == []


async def test_ruff_timeout_returns_empty() -> None:
    proc = _fake_process(b"")
    proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)

    with patch("pr_checker.static_analyzer.asyncio.create_subprocess_exec", return_value=proc):
        findings = await StaticAnalyzer().run(
            {"foo.py": _UNUSED_IMPORT}, run_ruff=True, run_mypy=False
        )
    assert findings == []


async def test_ruff_invalid_json_returns_empty() -> None:
    proc = _fake_process(b"not json output")

    with patch("pr_checker.static_analyzer.asyncio.create_subprocess_exec", return_value=proc):
        findings = await StaticAnalyzer().run(
            {"foo.py": _UNUSED_IMPORT}, run_ruff=True, run_mypy=False
        )
    assert findings == []


# ---------------------------------------------------------------------------
# Error handling — mypy
# ---------------------------------------------------------------------------


async def test_mypy_not_found_returns_empty() -> None:
    with patch(
        "pr_checker.static_analyzer.asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError,
    ):
        findings = await StaticAnalyzer().run(
            {"foo.py": _TYPE_ERROR}, run_ruff=False, run_mypy=True
        )
    assert findings == []


async def test_mypy_not_installed_returns_empty() -> None:
    proc = _fake_process(b"", returncode=1, stderr=b"/path/python: No module named mypy\n")
    with patch("pr_checker.static_analyzer.asyncio.create_subprocess_exec", return_value=proc):
        findings = await StaticAnalyzer().run(
            {"foo.py": _TYPE_ERROR}, run_ruff=False, run_mypy=True
        )
    assert findings == []


async def test_mypy_timeout_returns_empty() -> None:
    proc = _fake_process(b"")
    proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)

    with patch("pr_checker.static_analyzer.asyncio.create_subprocess_exec", return_value=proc):
        findings = await StaticAnalyzer().run(
            {"foo.py": _TYPE_ERROR}, run_ruff=False, run_mypy=True
        )
    assert findings == []


# ---------------------------------------------------------------------------
# Both tools together
# ---------------------------------------------------------------------------


async def test_both_tools_combined_findings() -> None:
    findings = await StaticAnalyzer().run(
        {"combo.py": _UNUSED_IMPORT + _TYPE_ERROR},
        run_ruff=True,
        run_mypy=True,
    )
    tools = {f.tool for f in findings}
    assert "ruff" in tools
    assert "mypy" in tools
