"""Unit tests for LMStudioClient._parse_model and list_models filtering."""

import json
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from pr_checker.lm_studio_client import LMStudioClient, _parse_params_string

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "lm_studio" / "models.json").read_text()
)

BASE_URL = "http://lmstudio.test/api/v1"


@pytest.fixture
async def client() -> AsyncGenerator[LMStudioClient, None]:
    c = LMStudioClient(BASE_URL, api_key="")
    yield c
    await c.aclose()


# --- list_models: filtering and shape ---


async def test_list_models_excludes_embedding_models(
    client: LMStudioClient, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/models", json=_FIXTURE)
    models = await client.list_models()
    ids = {m.id for m in models}
    assert "text-embedding-nomic-ai-nomic-embed-text-v2-moe" not in ids
    assert "text-embedding-multilingual-e5-large" not in ids
    assert "text-embedding-nomic-embed-text-v1.5" not in ids
    assert "text-embedding-multilingual-e5-small" not in ids


async def test_list_models_includes_llm_models(
    client: LMStudioClient, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/models", json=_FIXTURE)
    models = await client.list_models()
    ids = {m.id for m in models}
    assert "qwen/qwen3.6-27b" in ids
    assert "google/gemma-4-26b-a4b" in ids


async def test_list_models_empty_response(client: LMStudioClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/models", json={"models": []})
    assert await client.list_models() == []


# --- loaded instance: id and context_length come from the instance ---


async def test_loaded_model_uses_instance_id(client: LMStudioClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/models", json=_FIXTURE)
    models = await client.list_models()
    qwen = next(m for m in models if m.id == "qwen/qwen3.6-27b")
    assert qwen.loaded is True
    assert qwen.instance_id == "qwen/qwen3.6-27b"


async def test_loaded_model_uses_instance_context_length_not_max(
    client: LMStudioClient, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/models", json=_FIXTURE)
    models = await client.list_models()
    qwen = next(m for m in models if m.id == "qwen/qwen3.6-27b")
    # Instance config says 130000; model max is 262144 — must use the lower instance value
    assert qwen.context_length == 130000


async def test_loaded_model_vram_bytes_from_size_bytes(
    client: LMStudioClient, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/models", json=_FIXTURE)
    models = await client.list_models()
    qwen = next(m for m in models if m.id == "qwen/qwen3.6-27b")
    assert qwen.vram_bytes == 17478732949


async def test_gemma_loaded_uses_instance_context_length(
    client: LMStudioClient, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/models", json=_FIXTURE)
    models = await client.list_models()
    gemma = next(m for m in models if m.id == "google/gemma-4-26b-a4b")
    assert gemma.loaded is True
    assert gemma.context_length == 80000  # instance config, not max_context_length 262144


async def test_ministral_loaded_uses_instance_context_length(
    client: LMStudioClient, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/models", json=_FIXTURE)
    models = await client.list_models()
    ministral = next(m for m in models if m.id == "mistralai/ministral-3-14b-reasoning")
    assert ministral.loaded is True
    assert ministral.context_length == 60000  # instance config, not max_context_length 262144


async def test_fixture_loaded_model_ids(client: LMStudioClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/models", json=_FIXTURE)
    models = await client.list_models()
    loaded_ids = {m.id for m in models if m.loaded}
    assert loaded_ids == {
        "qwen/qwen3.6-27b",
        "google/gemma-4-26b-a4b",
        "mistralai/ministral-3-14b-reasoning",
        "qwen/qwen3.5-9b_102564",
        "qwen/qwen3.5-9b_10256",
    }


# --- unloaded models: id and context_length come from the top-level entry ---


async def test_unloaded_model_uses_key_as_id(client: LMStudioClient, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/models", json=_FIXTURE)
    models = await client.list_models()
    devstral = next(m for m in models if m.id == "devstral-2-123b-instruct-2512")
    assert devstral.loaded is False
    assert devstral.instance_id is None


async def test_unloaded_model_uses_max_context_length(
    client: LMStudioClient, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/models", json=_FIXTURE)
    models = await client.list_models()
    devstral = next(m for m in models if m.id == "devstral-2-123b-instruct-2512")
    assert devstral.context_length == 262144


async def test_unloaded_model_with_small_context(
    client: LMStudioClient, httpx_mock: HTTPXMock
) -> None:
    # llava-phi-3-mini has max_context_length: 4096
    httpx_mock.add_response(url=f"{BASE_URL}/models", json=_FIXTURE)
    models = await client.list_models()
    llava = next(m for m in models if m.id == "llava-phi-3-mini")
    assert llava.context_length == 4096


# --- multiple instances from real fixture (qwen/qwen3.5-9b) ---


async def test_qwen35_9b_instance_large_context(
    client: LMStudioClient, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/models", json=_FIXTURE)
    models = await client.list_models()
    inst = next(m for m in models if m.id == "qwen/qwen3.5-9b_102564")
    assert inst.loaded is True
    assert inst.context_length == 102564
    assert inst.instance_id == "qwen/qwen3.5-9b_102564"


async def test_qwen35_9b_instance_small_context(
    client: LMStudioClient, httpx_mock: HTTPXMock
) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/models", json=_FIXTURE)
    models = await client.list_models()
    inst = next(m for m in models if m.id == "qwen/qwen3.5-9b_10256")
    assert inst.loaded is True
    assert inst.context_length == 10256
    assert inst.instance_id == "qwen/qwen3.5-9b_10256"


# --- multiple instances of the same model (synthetic) ---


async def test_multiple_loaded_instances_produce_separate_entries(
    client: LMStudioClient, httpx_mock: HTTPXMock
) -> None:
    payload = {
        "models": [
            {
                "type": "llm",
                "key": "qwen/qwen3.5-9b",
                "size_bytes": 6548927017,
                "params_string": "9B",
                "max_context_length": 262144,
                "loaded_instances": [
                    {"id": "qwen/qwen3.5-9b", "config": {"context_length": 32768}},
                    {"id": "qwen/qwen3.5-9b-long", "config": {"context_length": 65536}},
                ],
            }
        ]
    }
    httpx_mock.add_response(url=f"{BASE_URL}/models", json=payload)
    models = await client.list_models()
    assert len(models) == 2
    by_id = {m.id: m for m in models}
    assert by_id["qwen/qwen3.5-9b"].context_length == 32768
    assert by_id["qwen/qwen3.5-9b-long"].context_length == 65536
    assert all(m.loaded for m in models)
    assert all(m.vram_bytes == 6548927017 for m in models)


# --- params_string edge cases ---


def test_params_string_null_yields_zero() -> None:
    assert _parse_params_string("") == 0


def test_params_string_simple_billions() -> None:
    assert _parse_params_string("27B") == 27_000_000_000


def test_params_string_simple_millions() -> None:
    assert _parse_params_string("559M") == 559_000_000


def test_params_string_moe() -> None:
    assert _parse_params_string("8x277M") == 8 * 277_000_000


def test_params_string_fractional() -> None:
    assert _parse_params_string("8.2B") == 8_200_000_000


def test_params_string_moe_unparseable_returns_zero() -> None:
    # "26B-A4B" style (active params suffix) — not a recognised format
    assert _parse_params_string("26B-A4B") == 0
    assert _parse_params_string("35B-A3B") == 0
