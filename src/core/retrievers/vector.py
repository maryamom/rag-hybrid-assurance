"""Semantic search over the public knowledge base stored in Qdrant."""
from __future__ import annotations

from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from qdrant_client.models import FieldCondition, Filter, MatchAny

from ..graph_schema import PUBLIC_VECTOR_TYPES


class QdrantKnowledgeRetriever(BaseRetriever):
    """One filtered similarity search: results come back sorted by score.

    The filter restricts the search to public source types, so client records can
    never be retrieved here even if they were indexed by mistake.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any  # qdrant_client.AsyncQdrantClient
    embedder: Any  # src.core.embeddings.Embedder
    collection: str
    k: int = 8

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        vector = await self.embedder.aembed_query(query)
        response = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=Filter(
                must=[FieldCondition(key="source_type", match=MatchAny(any=list(PUBLIC_VECTOR_TYPES)))]
            ),
            limit=self.k,
            with_payload=True,
        )
        documents = []
        for point in response.points:
            payload = point.payload or {}
            text = (payload.get("text") or "").strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "id": f"vector:{point.id}",
                        "retriever": "vector",
                        "kind": payload.get("source_type", "document"),
                        "title": payload.get("title") or payload.get("filename") or "Document",
                        "score": float(point.score),
                    },
                )
            )
        return documents

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        raise NotImplementedError("Use the async API (ainvoke).")
