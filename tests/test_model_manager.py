from unittest.mock import AsyncMock

import pytest

from pr_checker.lm_studio_client import LMStudioClient, ModelInfo
from pr_checker.model_manager import CannotFreeVramError, ModelManager, NoModelFitsError
from pr_checker.reviewer_config import ModelConfig, ModelTaskConfig, ReviewerConfig


def _model(
    model_id: str,
    context_length: int,
    vram_gb: float = 0.0,
    num_params: int = 0,
    loaded: bool = False,
    instance_id: str | None = None,
) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        context_length=context_length,
        num_params=num_params,
        vram_bytes=int(vram_gb * 1024**3),
        loaded=loaded,
        instance_id=instance_id if loaded else None,
    )


def _config(
    code_review: str = "model-a",
    synthesis: str = "model-a",
    vram_budget_gb: float = 16.0,
    bytes_per_param: float = 2.0,
) -> ReviewerConfig:
    return ReviewerConfig(
        models=ModelConfig(tasks=ModelTaskConfig(code_review=code_review, synthesis=synthesis)),
        vram_budget_gb=vram_budget_gb,
        bytes_per_param=bytes_per_param,
    )


@pytest.fixture
def mock_client() -> AsyncMock:
    return AsyncMock(spec=LMStudioClient)


# --- task mapping ---


async def test_returns_configured_code_review_model(mock_client: AsyncMock) -> None:
    mock_client.list_models.return_value = [
        _model("model-a", 8192),
        _model("model-b", 4096),
    ]
    manager = ModelManager(mock_client)
    result = await manager.get_model_for_task("code_review", 1000, _config(code_review="model-a"))
    assert result == "model-a"
    mock_client.load_model.assert_called_once_with("model-a")


async def test_returns_configured_synthesis_model(mock_client: AsyncMock) -> None:
    mock_client.list_models.return_value = [
        _model("model-a", 8192),
        _model("model-b", 4096),
    ]
    manager = ModelManager(mock_client)
    result = await manager.get_model_for_task(
        "synthesis", 1000, _config(code_review="model-a", synthesis="model-b")
    )
    assert result == "model-b"
    mock_client.load_model.assert_called_once_with("model-b")


async def test_already_loaded_model_not_reloaded(mock_client: AsyncMock) -> None:
    mock_client.list_models.return_value = [
        _model("model-a", 8192, vram_gb=4.0, loaded=True, instance_id="inst-1")
    ]
    manager = ModelManager(mock_client)
    result = await manager.get_model_for_task("code_review", 1000, _config(code_review="model-a"))
    assert result == "model-a"
    mock_client.load_model.assert_not_called()
    mock_client.unload_model.assert_not_called()


# --- context-size fallback ---


async def test_context_fallback_picks_smallest_sufficient_model(mock_client: AsyncMock) -> None:
    mock_client.list_models.return_value = [
        _model("small", 4096),
        _model("medium", 16384),
        _model("large", 65536),
    ]
    manager = ModelManager(mock_client)
    # configured model "small" has 4096 context; need 8000 tokens
    result = await manager.get_model_for_task("code_review", 8000, _config(code_review="small"))
    assert result == "medium"
    mock_client.load_model.assert_called_once_with("medium")


async def test_context_fallback_when_configured_model_absent(mock_client: AsyncMock) -> None:
    mock_client.list_models.return_value = [
        _model("medium", 16384),
        _model("large", 65536),
    ]
    manager = ModelManager(mock_client)
    result = await manager.get_model_for_task(
        "code_review", 1000, _config(code_review="unknown-model")
    )
    assert result == "medium"  # smallest that fits


# --- VRAM management ---


async def test_vram_fits_both_does_not_unload(mock_client: AsyncMock) -> None:
    # currently loaded 5 GB; target 8 GB; budget 16 GB → 5+8=13 ≤ 16
    mock_client.list_models.return_value = [
        _model("current", 8192, vram_gb=5.0, loaded=True, instance_id="inst-current"),
        _model("target", 8192, vram_gb=8.0),
    ]
    manager = ModelManager(mock_client)
    result = await manager.get_model_for_task(
        "code_review", 1000, _config(code_review="target", vram_budget_gb=16.0)
    )
    assert result == "target"
    mock_client.unload_model.assert_not_called()
    mock_client.load_model.assert_called_once_with("target")


async def test_vram_must_swap_unloads_instance_first(mock_client: AsyncMock) -> None:
    # currently loaded 9 GB; target 8 GB; budget 16 GB → 9+8=17 > 16
    mock_client.list_models.return_value = [
        _model("current", 8192, vram_gb=9.0, loaded=True, instance_id="inst-current"),
        _model("target", 8192, vram_gb=8.0),
    ]
    manager = ModelManager(mock_client)
    result = await manager.get_model_for_task(
        "code_review", 1000, _config(code_review="target", vram_budget_gb=16.0)
    )
    assert result == "target"
    mock_client.unload_model.assert_called_once_with("inst-current")
    mock_client.load_model.assert_called_once_with("target")


async def test_no_currently_loaded_model_skips_vram_check(mock_client: AsyncMock) -> None:
    mock_client.list_models.return_value = [_model("target", 8192)]
    manager = ModelManager(mock_client)
    result = await manager.get_model_for_task("code_review", 1000, _config(code_review="target"))
    assert result == "target"
    mock_client.unload_model.assert_not_called()
    mock_client.load_model.assert_called_once_with("target")


# --- multiple loaded models ---


async def test_multiple_loaded_models_unloads_largest_first(mock_client: AsyncMock) -> None:
    # two loaded models: 6 GB + 4 GB = 10 GB total; target 8 GB; budget 16 GB → 18 > 16
    # must unload 6 GB model (largest) to fit: 10-6+8=12 ≤ 16
    mock_client.list_models.return_value = [
        _model("large-loaded", 8192, vram_gb=6.0, loaded=True, instance_id="inst-large"),
        _model("small-loaded", 8192, vram_gb=4.0, loaded=True, instance_id="inst-small"),
        _model("target", 8192, vram_gb=8.0),
    ]
    manager = ModelManager(mock_client)
    result = await manager.get_model_for_task(
        "code_review", 1000, _config(code_review="target", vram_budget_gb=16.0)
    )
    assert result == "target"
    mock_client.unload_model.assert_called_once_with("inst-large")
    mock_client.load_model.assert_called_once_with("target")


async def test_cannot_free_vram_raises_error(mock_client: AsyncMock) -> None:
    # loaded model has no instance_id and budget would be exceeded → raise
    mock_client.list_models.return_value = [
        _model("current", 8192, vram_gb=9.0, loaded=True, instance_id=None),
        _model("target", 8192, vram_gb=8.0),
    ]
    manager = ModelManager(mock_client)
    with pytest.raises(CannotFreeVramError):
        await manager.get_model_for_task(
            "code_review", 1000, _config(code_review="target", vram_budget_gb=16.0)
        )
    mock_client.unload_model.assert_not_called()
    mock_client.load_model.assert_not_called()


# --- VRAM estimation from num_params ---


async def test_vram_estimated_from_num_params_when_size_bytes_absent(
    mock_client: AsyncMock,
) -> None:
    # 7B params * 2.0 bytes/param = 14 GB; two such models exceed the 16 GB budget
    seven_b = 7_000_000_000
    mock_client.list_models.return_value = [
        _model("current", 8192, num_params=seven_b, loaded=True, instance_id="inst-current"),
        _model("target", 8192, num_params=seven_b),
    ]
    manager = ModelManager(mock_client)
    result = await manager.get_model_for_task(
        "code_review", 1000, _config(code_review="target", vram_budget_gb=16.0, bytes_per_param=2.0)
    )
    assert result == "target"
    mock_client.unload_model.assert_called_once_with("inst-current")


async def test_unknown_vram_treated_as_full_budget(mock_client: AsyncMock) -> None:
    # vram_bytes=0 and num_params=0 → each model costs the full budget
    # Two full budgets always exceed one budget, so unload must happen
    mock_client.list_models.return_value = [
        _model("current", 8192, loaded=True, instance_id="inst-current"),
        _model("target", 8192),
    ]
    manager = ModelManager(mock_client)
    result = await manager.get_model_for_task(
        "code_review", 1000, _config(code_review="target", vram_budget_gb=16.0)
    )
    assert result == "target"
    mock_client.unload_model.assert_called_once_with("inst-current")


# --- no model fits ---


async def test_no_model_fits_raises_error(mock_client: AsyncMock) -> None:
    mock_client.list_models.return_value = [
        _model("small", 1024),
        _model("medium", 2048),
    ]
    manager = ModelManager(mock_client)
    with pytest.raises(NoModelFitsError):
        await manager.get_model_for_task("code_review", 10000, _config(code_review="small"))
