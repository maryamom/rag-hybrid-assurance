"""Hybrid retrieval: query Qdrant and Neo4j in parallel, merge with Reciprocal Rank Fusion.

RRF only uses ranks, so it can merge a cosine similarity and a Lucene score that are
not comparable. A passage that both sources rank high ends up first.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

logger = logging.getLogger(__name__)

RRF_K = 60  # standard constant from Cormack et al. (2009)


def _content_key(document: Document) -> str:
    normalized = " ".join(document.page_content.lower().split())[:300]
    return hashlib.sha1(normalized.encode()).hexdigest()


def reciprocal_rank_fusion(result_lists: list[list[Document]], top_n: int, k: int = RRF_K) -> list[Document]:
    scores: dict[str, float] = {}
    first_seen: dict[str, Document] = {}
    for results in result_lists:
        for rank, document in enumerate(results, start=1):
            key = _content_key(document)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(key, document)
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_n]
    fused = []
    for key in ranked:
        document = first_seen[key]
        fused.append(Document(page_content=document.page_content, metadata={**document.metadata, "rrf": scores[key]}))
    return fused


class HybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    retrievers: list[Any]  # BaseRetriever instances
    top_n: int = 8
    timeout_seconds: float = 15.0

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        outcomes = await asyncio.gather(
            *(asyncio.wait_for(retriever.ainvoke(query), self.timeout_seconds) for retriever in self.retrievers),
            return_exceptions=True,
        )
        result_lists = []
        for retriever, outcome in zip(self.retrievers, outcomes, strict=True):
            if isinstance(outcome, Exception):
                # One source down should not take the assistant down; /api/health reports it.
                logger.error("%s failed: %s", type(retriever).__name__, type(outcome).__name__)
                continue
            result_lists.append(outcome)
        return reciprocal_rank_fusion(result_lists, self.top_n)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        raise NotImplementedError("Use the async API (ainvoke).")
