import logging

from openai import AsyncOpenAI

_log = logging.getLogger(__name__)

_BATCH_SIZE = 100


class EmbeddingService:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches. Returns one vector per input text."""
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            response = await self._client.embeddings.create(model=self._model, input=batch)
            if len(response.data) != len(batch):
                raise ValueError(f"Expected {len(batch)} embeddings, got {len(response.data)}")
            results.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
            _log.debug("Embedded batch of %d texts (model=%s)", len(batch), self._model)
        return results
