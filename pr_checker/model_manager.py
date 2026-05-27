from __future__ import annotations

import logging

from pr_checker.lm_studio_client import LMStudioClient, ModelInfo
from pr_checker.reviewer_config import ReviewerConfig


class NoModelFitsError(Exception):
    pass


class CannotFreeVramError(Exception):
    pass


class ModelManager:
    def __init__(self, client: LMStudioClient) -> None:
        self._client = client

    async def get_model_for_task(
        self, task_type: str, estimated_tokens: int, config: ReviewerConfig
    ) -> str:
        models = await self._client.list_models()
        target = self._select_model(models, task_type, estimated_tokens, config)

        if target.loaded:
            return target.id

        loaded_models = [m for m in models if m.loaded]
        vram_budget_bytes = int(config.vram_budget_gb * 1024**3)
        total_loaded_vram = sum(_model_vram(m, config) for m in loaded_models)
        target_vram = _model_vram(target, config)

        if target_vram > vram_budget_bytes:
            raise CannotFreeVramError(
                f"{target.id} requires {target_vram} bytes but budget is {vram_budget_bytes}"
            )

        freed = 0
        if total_loaded_vram + target_vram > vram_budget_bytes:
            to_unload = sorted(loaded_models, key=lambda m: _model_vram(m, config), reverse=True)
            for m in to_unload:
                if total_loaded_vram - freed + target_vram <= vram_budget_bytes:
                    break
                if m.instance_id is None:
                    raise CannotFreeVramError(f"Cannot free VRAM: {m.id} has no instance_id")
                logging.info("Unloading %s to free VRAM for %s", m.id, target.id)
                await self._client.unload_model(m.instance_id)
                freed += _model_vram(m, config)

        if total_loaded_vram - freed + target_vram > vram_budget_bytes:
            raise CannotFreeVramError("Could not free sufficient VRAM to load target model")

        await self._client.load_model(target.id)
        return target.id

    def _select_model(
        self,
        models: list[ModelInfo],
        task_type: str,
        estimated_tokens: int,
        config: ReviewerConfig,
    ) -> ModelInfo:
        configured_id = _configured_model_id(task_type, config)
        models_by_id = {m.id: m for m in models}

        configured = models_by_id.get(configured_id)
        if configured is not None and configured.context_length >= estimated_tokens:
            return configured

        candidates = sorted(
            [m for m in models if m.context_length >= estimated_tokens],
            key=lambda m: m.context_length,
        )
        if not candidates:
            raise NoModelFitsError(
                f"No available model has context length >= {estimated_tokens} tokens"
            )
        return candidates[0]


def _model_vram(model: ModelInfo, config: ReviewerConfig) -> int:
    if model.vram_bytes > 0:
        return model.vram_bytes
    if model.num_params > 0:
        return int(model.num_params * config.bytes_per_param)
    # Unknown size: assume worst case to avoid overcommitting VRAM
    return int(config.vram_budget_gb * 1024**3)


def _configured_model_id(task_type: str, config: ReviewerConfig) -> str:
    tasks = config.models.tasks
    task_map = {
        "code_review": tasks.code_review,
        "synthesis": tasks.synthesis,
    }
    return task_map.get(task_type, config.models.default)
