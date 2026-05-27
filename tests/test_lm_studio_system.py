"""System tests for LMStudioClient against a real LM Studio instance.

Skipped automatically when LM_STUDIO_URL is not set. To run:
    export LM_STUDIO_URL=http://localhost:1234/api/v1
    export LM_STUDIO_API_KEY=...
    pytest tests/test_lm_studio_system.py -v

LM_STUDIO_URL must include the full base path (e.g. http://localhost:1234/api/v1).
Alternatively, populate LM_STUDIO_URL and LM_STUDIO_API_KEY in the local .env file before running.
"""

import logging

import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict

from pr_checker.lm_studio_client import LMStudioClient


class _Env(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    lm_studio_url: str = ""
    lm_studio_api_key: str = ""


_env = _Env()

pytestmark = pytest.mark.skipif(
    not _env.lm_studio_url,
    reason="LM_STUDIO_URL not set",
)

# Loading a large model can take several minutes.
_LOAD_TIMEOUT = 300.0


async def test_list_models_returns_nonempty_list() -> None:
    client = LMStudioClient(_env.lm_studio_url, _env.lm_studio_api_key, timeout=_LOAD_TIMEOUT)
    try:
        models = await client.list_models()
        assert models, "Expected at least one model in LM Studio"
        for m in models:
            assert m.id, "model.id should be non-empty"
            assert m.context_length > 0, f"{m.id}: context_length must be > 0"
            assert m.vram_bytes >= 0, f"{m.id}: vram_bytes (size_bytes) must be >= 0"
    finally:
        await client.aclose()


async def test_load_verify_unload_verify() -> None:
    client = LMStudioClient(_env.lm_studio_url, _env.lm_studio_api_key, timeout=_LOAD_TIMEOUT)
    instance_id: str | None = None
    try:
        models = await client.list_models()
        not_loaded = next((m for m in models if not m.loaded), None)
        if not_loaded is None:
            pytest.skip("No unloaded models available to test with")

        target_key = not_loaded.id

        # --- load ---
        await client.load_model(target_key)

        after_load = {m.id: m for m in await client.list_models()}
        loaded_model = after_load.get(target_key)
        assert loaded_model is not None, f"{target_key} disappeared after load"
        assert loaded_model.loaded, f"{target_key} should be loaded"
        assert loaded_model.vram_bytes >= 0, f"{target_key} should report vram_bytes >= 0"
        assert loaded_model.instance_id is not None, "loaded model must have an instance_id"

        instance_id = loaded_model.instance_id

        # --- unload ---
        await client.unload_model(instance_id)
        instance_id = None  # prevent double-unload in finally

        after_unload = {m.id: m for m in await client.list_models()}
        unloaded_model = after_unload.get(target_key)
        assert unloaded_model is None or not unloaded_model.loaded, (
            f"{target_key} should no longer be loaded"
        )
    finally:
        if instance_id is not None:
            try:
                await client.unload_model(instance_id)
            except Exception:
                logging.debug("Cleanup unload failed", exc_info=True)
        await client.aclose()
