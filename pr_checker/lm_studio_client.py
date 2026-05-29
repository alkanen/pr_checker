from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ModelInfo:
    id: str
    context_length: int
    num_params: int  # 0 if unknown; used as VRAM fallback when size_bytes absent
    vram_bytes: int  # size_bytes from API (file size ≈ VRAM for GGUF); 0 if unavailable
    loaded: bool
    instance_id: str | None  # id of the loaded instance; required for unload; None if not loaded


class LMStudioClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
    ) -> None:
        # Trailing slash is required: httpx resolves relative paths against the
        # base path, so "models" under "https://host/api/v1/" → correct URL.
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def list_models(self) -> list[ModelInfo]:
        response = await self._http.get("models")
        response.raise_for_status()
        result: list[ModelInfo] = []
        for raw in response.json().get("models", []):
            if raw.get("type") != "llm":
                continue
            result.extend(self._parse_model(raw))
        return result

    async def load_model(self, model_id: str) -> None:
        response = await self._http.post("models/load", json={"model": model_id})
        response.raise_for_status()

    async def unload_model(self, instance_id: str) -> None:
        response = await self._http.post("models/unload", json={"instance_id": instance_id})
        response.raise_for_status()

    def _parse_model(self, raw: dict[str, Any]) -> list[ModelInfo]:
        vram_bytes = int(raw.get("size_bytes") or 0)
        num_params = _parse_params_string(raw.get("params_string") or "")
        max_context = int(raw.get("max_context_length") or raw.get("context_length") or 4096)

        loaded_instances: list[dict[str, Any]] = raw.get("loaded_instances") or []
        if loaded_instances:
            # One ModelInfo per loaded instance — the instance id is how LM Studio (and
            # LLM_MODEL) addresses it, and its context_length is what the server will
            # actually accept (may be lower than the model's absolute max_context_length).
            return [
                ModelInfo(
                    id=inst["id"],
                    context_length=int(inst.get("config", {}).get("context_length") or max_context),
                    num_params=num_params,
                    vram_bytes=vram_bytes,
                    loaded=True,
                    instance_id=inst["id"],
                )
                for inst in loaded_instances
            ]

        return [
            ModelInfo(
                id=raw["key"],
                context_length=max_context,
                num_params=num_params,
                vram_bytes=vram_bytes,
                loaded=False,
                instance_id=None,
            )
        ]


def _parse_params_string(params_string: str) -> int:
    """Parse LM Studio's human-readable parameter count (e.g. '27B', '8x277M')."""
    # MoE format: "NxMB" or "NxMM"
    moe = re.fullmatch(r"(\d+)x(\d+(?:\.\d+)?)([BM])", params_string.strip())
    if moe:
        experts = int(moe.group(1))
        size = float(moe.group(2))
        multiplier = 1e9 if moe.group(3) == "B" else 1e6
        return round(experts * size * multiplier)
    # Simple format: "NB" or "NM"
    simple = re.fullmatch(r"(\d+(?:\.\d+)?)([BM])", params_string.strip())
    if simple:
        size = float(simple.group(1))
        multiplier = 1e9 if simple.group(2) == "B" else 1e6
        return round(size * multiplier)
    return 0
