"""Sentence embeddings, loaded lazily so importing the package doesn't require torch."""
from __future__ import annotations

import asyncio
from functools import cached_property
from typing import Any


class Embedder:
    """Multilingual by default: the documents are in French and Arabic."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @cached_property
    def _model(self) -> Any:
        from sentence_transformers import SentenceTransformer  # heavy import, only when needed

        return SentenceTransformer(self.model_name)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        vectors = self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)
