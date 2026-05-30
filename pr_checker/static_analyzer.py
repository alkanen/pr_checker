from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from pr_checker.models import StaticFinding

_log = logging.getLogger(__name__)

_TOOL_TIMEOUT = 30.0
_MYPY_RE = re.compile(r"^(.+?):(\d+):\s*(error|warning):\s*(.+?)(?:\s+\[([^\]]+)\])?$")


def _toml_key(key: str) -> str:
    return key if re.match(r"^[A-Za-z0-9_-]+$", key) else json.dumps(key)


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, dict):
        inner = ", ".join(f"{_toml_key(k)} = {_toml_scalar(v)}" for k, v in value.items())
        return "{" + inner + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    raise TypeError(f"cannot serialise {type(value).__name__} to TOML")


def _append_toml_section(lines: list[str], data: dict[str, Any], prefix: str) -> None:
    nested: list[tuple[str, dict[str, Any]]] = []
    for key, value in data.items():
        if isinstance(value, dict):
            nested.append((key, value))
        else:
            try:
                lines.append(f"{_toml_key(key)} = {_toml_scalar(value)}")
            except TypeError:
                _log.debug("Skipping non-serialisable config key %r", key)
    for key, sub_dict in nested:
        section = f"{prefix}.{key}" if prefix else key
        lines.append(f"\n[{section}]")
        _append_toml_section(lines, sub_dict, section)


def _write_ruff_toml(config: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    _append_toml_section(lines, config, "")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_mypy_config(config: dict[str, Any], path: Path) -> None:
    lines: list[str] = ["[tool.mypy]"]
    _append_toml_section(lines, config, "tool.mypy")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class StaticAnalyzer:
    async def run(
        self,
        file_contents: dict[str, str],
        run_ruff: bool = True,
        run_mypy: bool = True,
        ruff_config: dict[str, Any] | None = None,
        mypy_config: dict[str, Any] | None = None,
    ) -> list[StaticFinding]:
        python_files = {p: c for p, c in file_contents.items() if p.endswith(".py")}
        if not python_files or (not run_ruff and not run_mypy):
            return []

        findings: list[StaticFinding] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            safe_paths: list[str] = []
            for rel_path, content in python_files.items():
                p = Path(rel_path)
                if p.is_absolute() or ".." in p.parts:
                    _log.warning("Skipping unsafe path from PR diff: %s", rel_path)
                    continue
                dest = tmp_path / p
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
                safe_paths.append(rel_path)

            if not safe_paths:
                return []

            if run_ruff and ruff_config:
                try:
                    _write_ruff_toml(ruff_config, tmp_path / "ruff.toml")
                except Exception:
                    _log.debug("Could not write ruff.toml from project config; using defaults")

            mypy_config_path: Path | None = None
            if run_mypy and mypy_config:
                try:
                    mypy_config_path = tmp_path / "pyproject.toml"
                    _write_mypy_config(mypy_config, mypy_config_path)
                except Exception:
                    _log.debug("Could not write pyproject.toml from mypy config; using defaults")
                    mypy_config_path = None

            abs_paths = [str(tmp_path / p) for p in safe_paths]

            if run_ruff:
                findings.extend(await _run_ruff(abs_paths, tmp_path))
            if run_mypy:
                findings.extend(await _run_mypy(safe_paths, tmp_path, mypy_config_path))

        return findings


async def _run_ruff(abs_paths: list[str], tmp_path: Path) -> list[StaticFinding]:
    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--output-format",
        "json",
        "--no-cache",
        "--quiet",
        "--",
        *abs_paths,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(tmp_path),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TOOL_TIMEOUT)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            _log.warning("ruff timed out after %.0fs; skipping", _TOOL_TIMEOUT)
            return []
    except FileNotFoundError:
        _log.warning("ruff not found; skipping ruff analysis")
        return []
    except Exception:
        _log.warning("ruff failed unexpectedly; skipping", exc_info=True)
        return []

    stderr_text = stderr.decode("utf-8", errors="replace")
    if "No module named" in stderr_text:
        _log.warning("ruff is not installed; skipping ruff analysis")
        return []

    try:
        data = json.loads(stdout.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, ValueError):
        _log.debug("ruff produced non-JSON output; skipping")
        return []

    findings: list[StaticFinding] = []
    for item in data:
        abs_path = item.get("filename", "")
        try:
            rel = Path(abs_path).relative_to(tmp_path).as_posix()
        except ValueError:
            rel = abs_path
        loc = item.get("location", {})
        findings.append(
            StaticFinding(
                tool="ruff",
                file_path=rel,
                line=loc.get("row"),
                col=loc.get("column"),
                code=item.get("code", ""),
                message=item.get("message", ""),
            )
        )
    return findings


async def _run_mypy(
    rel_paths: list[str], tmp_path: Path, config_file: Path | None = None
) -> list[StaticFinding]:
    # Use relative paths + cwd so mypy's error output is already repo-relative.
    # Always pass --ignore-missing-imports: we only materialise the changed files in the
    # temp dir, so all intra-repo imports that aren't part of the diff will be missing
    # on disk regardless of the repo's own mypy settings.
    config_args = ["--config-file", str(config_file)] if config_file is not None else []
    cmd = [
        sys.executable,
        "-m",
        "mypy",
        "--ignore-missing-imports",
        "--show-error-codes",
        "--no-error-summary",
        *config_args,
        "--",
        *rel_paths,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(tmp_path),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TOOL_TIMEOUT)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            _log.warning("mypy timed out after %.0fs; skipping", _TOOL_TIMEOUT)
            return []
    except FileNotFoundError:
        _log.warning("mypy not found; skipping mypy analysis")
        return []
    except Exception:
        _log.warning("mypy failed unexpectedly; skipping", exc_info=True)
        return []

    stderr_text = stderr.decode("utf-8", errors="replace")
    if "No module named" in stderr_text:
        _log.warning("mypy is not installed; skipping mypy analysis")
        return []

    findings: list[StaticFinding] = []
    for raw_line in stdout.decode("utf-8", errors="replace").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        m = _MYPY_RE.match(raw_line)
        if not m:
            continue
        file_path, lineno, _severity, message, code = m.groups()
        findings.append(
            StaticFinding(
                tool="mypy",
                file_path=Path(file_path).as_posix(),
                line=int(lineno),
                col=None,
                code=code or "",
                message=message.strip(),
            )
        )
    return findings
