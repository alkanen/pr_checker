from __future__ import annotations

import configparser
import json
import logging
import sys
from typing import Any

import httpx
import yaml

from pr_checker.github_client import GitHubClient
from pr_checker.models import ProjectStandards

_log = logging.getLogger(__name__)

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_ESLINT_FILENAMES = [
    ".eslintrc",
    ".eslintrc.json",
    ".eslintrc.yaml",
    ".eslintrc.yml",
]
_PRETTIER_FILENAMES = [
    ".prettierrc",
    ".prettierrc.json",
    ".prettierrc.yaml",
    ".prettierrc.yml",
]


class StandardsDetector:
    def __init__(self, client: GitHubClient) -> None:
        self._client = client

    async def detect(self, repo_full_name: str, ref: str) -> ProjectStandards:
        standards = ProjectStandards()

        pyproject = await self._fetch(repo_full_name, "pyproject.toml", ref)
        if pyproject is not None:
            try:
                data = tomllib.loads(pyproject)
                tool: dict[str, Any] = data.get("tool", {})
                if isinstance(tool.get("ruff"), dict):
                    standards.ruff = tool["ruff"]
                elif "ruff" in tool:
                    _log.warning(
                        "tool.ruff in pyproject.toml for %s is not a table; skipping",
                        repo_full_name,
                    )
                if isinstance(tool.get("mypy"), dict):
                    standards.mypy = tool["mypy"]
                elif "mypy" in tool:
                    _log.warning(
                        "tool.mypy in pyproject.toml for %s is not a table; skipping",
                        repo_full_name,
                    )
            except Exception:
                _log.warning("Failed to parse pyproject.toml for %s", repo_full_name, exc_info=True)

        if not standards.ruff:
            ruff_toml = await self._fetch(repo_full_name, "ruff.toml", ref)
            if ruff_toml is not None:
                try:
                    standards.ruff = tomllib.loads(ruff_toml)
                except Exception:
                    _log.warning("Failed to parse ruff.toml for %s", repo_full_name, exc_info=True)

        if not standards.mypy:
            mypy_ini = await self._fetch(repo_full_name, "mypy.ini", ref)
            if mypy_ini is not None:
                standards.mypy = _parse_ini_section(mypy_ini, "mypy")

        for filename in _ESLINT_FILENAMES:
            if standards.eslint:
                break
            content = await self._fetch(repo_full_name, filename, ref)
            if content is not None:
                standards.eslint = _parse_json_or_yaml(filename, content)

        for filename in _PRETTIER_FILENAMES:
            if standards.prettier:
                break
            content = await self._fetch(repo_full_name, filename, ref)
            if content is not None:
                standards.prettier = _parse_json_or_yaml(filename, content)

        return standards

    async def _fetch(self, repo_full_name: str, path: str, ref: str) -> str | None:
        try:
            return await self._client.get_file_content(repo_full_name, path, ref)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise


_INI_BOOL_TRUE = frozenset(("true", "yes", "on"))
_INI_BOOL_FALSE = frozenset(("false", "no", "off"))


def _coerce_ini_value(value: str) -> bool | int | str:
    lower = value.lower()
    if lower in _INI_BOOL_TRUE:
        return True
    if lower in _INI_BOOL_FALSE:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    return value


def _parse_ini_section(content: str, section: str) -> dict[str, Any]:
    try:
        parser = configparser.ConfigParser()
        parser.read_string(content)
        if section in parser:
            return {k: _coerce_ini_value(v) for k, v in parser[section].items()}
    except configparser.Error as exc:
        _log.warning("Failed to parse INI config for section [%s]: %s", section, exc)
    return {}


def _parse_json_or_yaml(filename: str, content: str) -> dict[str, Any]:
    if filename.endswith((".yaml", ".yml")):
        try:
            result = yaml.safe_load(content)
            return result if isinstance(result, dict) else {}
        except Exception as exc:
            _log.warning("Failed to parse %s as YAML: %s", filename, exc)
            return {}
    # Try JSON first; extensionless files (.eslintrc, .prettierrc) are often YAML
    try:
        result = json.loads(content)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        _log.debug("%s is not valid JSON; trying YAML", filename)
    try:
        result = yaml.safe_load(content)
        return result if isinstance(result, dict) else {}
    except Exception as exc:
        _log.warning("Failed to parse %s as YAML: %s", filename, exc)
        return {}
