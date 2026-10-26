"""Search in the public part of the Neo4j knowledge graph (products, guarantees, FAQ).

Uses a full-text index (one query, ranked by Lucene score) instead of CONTAINS scans,
and never touches client nodes: only labels from PUBLIC_GRAPH_LABELS are read.
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from ..graph_schema import GRAPH_FULLTEXT_INDEX, PUBLIC_GRAPH_LABELS

_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')

SEARCH_QUERY = """
CALL db.index.fulltext.queryNodes($index, $text, {limit: $k}) YIELD node, score
WHERE any(label IN labels(node) WHERE label IN $public_labels)
OPTIONAL MATCH (node)-[rel]-(neighbour)
WHERE any(label IN labels(neighbour) WHERE label IN $public_labels)
WITH node, score, collect(DISTINCT type(rel) + ' : ' + neighbour.nom)[..6] AS links
RETURN [label IN labels(node) WHERE label IN $public_labels][0] AS label,
       node.nom AS title, node.texte AS text, links, score
ORDER BY score DESC
"""


def to_lucene_query(question: str) -> str:
    """Escape Lucene syntax so a user's question is always searched as plain words."""
    escaped = _LUCENE_SPECIAL.sub(r"\\\1", question)
    return " ".join(escaped.split())


class Neo4jKnowledgeRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    driver: Any  # neo4j.AsyncDriver
    k: int = 8

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        text = to_lucene_query(query)
        if not text:
            return []
        records, _, _ = await self.driver.execute_query(
            SEARCH_QUERY,
            index=GRAPH_FULLTEXT_INDEX,
            text=text,
            k=self.k,
            public_labels=list(PUBLIC_GRAPH_LABELS),
        )
        documents = []
        for record in records:
            body = (record["text"] or record["title"] or "").strip()
            if not body:
                continue
            links = [link for link in record["links"] if link]
            content = body + ("\nRelations : " + " ; ".join(links) if links else "")
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "id": f"graph:{record['label']}:{record['title']}",
                        "retriever": "graph",
                        "kind": record["label"] or "graphe",
                        "title": record["title"] or record["label"],
                        "score": float(record["score"]),
                    },
                )
            )
        return documents

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        raise NotImplementedError("Use the async API (ainvoke).")
