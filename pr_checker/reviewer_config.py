from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import BaseModel, Field, ValidationError

from pr_checker.github_client import GitHubClient


class ModelTaskConfig(BaseModel):
    code_review: str = "gpt-4o-mini"
    synthesis: str = "gpt-4o-mini"


class ModelConfig(BaseModel):
    default: str = "gpt-4o-mini"
    tasks: ModelTaskConfig = Field(default_factory=ModelTaskConfig)


class EmbeddingConfig(BaseModel):
    model: str = "text-embedding-3-small"


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"


class ChecksConfig(BaseModel):
    code_quality: bool = True
    security: bool = True
    test_coverage: bool = True
    issue_conformance: bool = True
    ignore_patterns: list[str] = Field(default_factory=list)


class OutputConfig(BaseModel):
    inline_comments: bool = True
    summary_comment: bool = True
    formal_review: bool = True


class ReviewerConfig(BaseModel):
    models: ModelConfig = Field(default_factory=ModelConfig)
    vram_budget_gb: float = 16.0
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    checks: ChecksConfig = Field(default_factory=ChecksConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if value is None:
            continue  # null in repo config means "keep server default"
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _enforce_limits(merged: ReviewerConfig, server: ReviewerConfig) -> ReviewerConfig:
    """Clamp merged config so a repo cannot escalate beyond what the server permits."""
    checks_data = merged.checks.model_dump()
    for field_name, server_val in server.checks.model_dump().items():
        if isinstance(server_val, bool) and not server_val:
            checks_data[field_name] = False

    output_data = merged.output.model_dump()
    for field_name, server_val in server.output.model_dump().items():
        if isinstance(server_val, bool) and not server_val:
            output_data[field_name] = False

    return merged.model_copy(
        update={
            "checks": ChecksConfig(**checks_data),
            "output": OutputConfig(**output_data),
            "vram_budget_gb": min(merged.vram_budget_gb, server.vram_budget_gb),
            # Infrastructure settings are always server-controlled; repos cannot
            # redirect the Qdrant endpoint or swap the embedding model.
            "qdrant": server.qdrant,
            "embedding": server.embedding,
        }
    )


class ConfigManager:
    def __init__(self, config_file: Path | None = None) -> None:
        self._server_config = self._load_file(config_file)

    @property
    def server_config(self) -> ReviewerConfig:
        return self._server_config

    def _load_file(self, path: Path | None) -> ReviewerConfig:
        if path is None or not path.is_file():
            return ReviewerConfig()
        try:
            with path.open() as f:
                raw = yaml.safe_load(f)
            if not isinstance(raw, dict):
                logging.warning("Config file %s is not a YAML mapping; using defaults", path)
                return ReviewerConfig()
            return ReviewerConfig.model_validate(raw)
        except (OSError, yaml.YAMLError, ValidationError) as exc:
            logging.warning("Failed to load config file %s; using defaults: %s", path, exc)
            return ReviewerConfig()

    async def for_repo(self, repo_full_name: str, client: GitHubClient) -> ReviewerConfig:
        default_branch = await client.get_default_branch(repo_full_name)
        try:
            content = await client.get_file_content(
                repo_full_name, ".pr-checker.yml", default_branch
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return self._server_config
            raise

        if content is None:
            return self._server_config

        try:
            raw = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            logging.warning(
                ".pr-checker.yml in %s is invalid YAML; using server defaults: %s",
                repo_full_name,
                exc,
            )
            return self._server_config

        if not isinstance(raw, dict):
            logging.warning(
                ".pr-checker.yml in %s is not a mapping; using server defaults", repo_full_name
            )
            return self._server_config

        try:
            merged_data = _deep_merge(self._server_config.model_dump(), raw)
            merged = ReviewerConfig.model_validate(merged_data)
        except ValidationError as exc:
            logging.warning(
                ".pr-checker.yml in %s has invalid values; using server defaults: %s",
                repo_full_name,
                exc,
            )
            return self._server_config

        return _enforce_limits(merged, self._server_config)
