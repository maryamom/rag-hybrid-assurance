"""Step 4 – build the public vector index in Qdrant.

    python -m ingestion.index_qdrant

Indexes the cleaned PDF documents and the public reference tables (guarantee
descriptions, FAQ, product mapping, data dictionary). Client records (people,
contracts, claims, quotes) are never indexed: they stay in PostgreSQL and are
read only after authentication.
"""
from __future__ import annotations

import argparse
import asyncio
import uuid
from typing import Any, Iterable

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from src.config import get_settings
from src.core.embeddings import Embedder
from src.core.graph_schema import PUBLIC_VECTOR_TYPES

from .common import connect, quote_ident

# Public reference tables (names produced by load_tables.py) -> payload source_type.
# Matched on the start of the table name. Tables not listed here are never indexed.
PUBLIC_TABLES = {
    "description_garanties": "garantie",
    "exemples_questions_reponses": "faq",
    "mapping_produits_vs_profils_cibles": "produit",
    "description_des_colonnes": "dictionnaire",
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
_ID_NAMESPACE = uuid.UUID("6f1c2a52-5d1e-4d7a-9a51-0b6f3e0c9e11")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on paragraph boundaries, then hard-wrap paragraphs longer than `size`."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        while len(paragraph) > size:
            cut = paragraph.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            pieces.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        pieces.append(paragraph)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > size:
            chunks.append(current)
            current = current[-overlap:].split(" ", 1)[-1] if overlap else ""
        current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def row_to_text(row: dict[str, Any]) -> tuple[str, str]:
    """('col: value | col: value', title) for one table row."""
    values = [(k, v) for k, v in row.items() if v not in (None, "")]
    text = " | ".join(f"{k}: {v}" for k, v in values)
    title = next((str(v) for _, v in values if isinstance(v, str) and len(v) <= 120), "")
    return text, title


def source_type_of(table: str) -> str | None:
    for prefix, source_type in PUBLIC_TABLES.items():
        if table.startswith(prefix):
            assert source_type in PUBLIC_VECTOR_TYPES
            return source_type
    return None


async def collect_items(conn: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in await conn.fetch("SELECT filename, branche, text_cleaned FROM cleaned_docs ORDER BY filename"):
        for i, chunk in enumerate(chunk_text(row["text_cleaned"])):
            items.append({
                "key": f"doc:{row['filename']}:{i}", "text": chunk, "source_type": "document",
                "title": row["filename"], "branche": row["branche"],
            })

    tables = [r["table_name"] for r in await conn.fetch(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    )]
    for table in sorted(tables):
        source_type = source_type_of(table)
        if source_type is None:
            continue
        rows = await conn.fetch(f"SELECT * FROM {quote_ident(table)}")
        for i, record in enumerate(rows):
            text, title = row_to_text(dict(record))
            if text:
                items.append({
                    "key": f"{table}:{i}", "text": text, "source_type": source_type,
                    "title": title or table, "branche": "",
                })
        print(f"{table:<60} {len(rows):>6} rows -> {source_type}")
    return items


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


async def build_index(client: Any, embedder: Any, collection: str, items: list[dict[str, Any]]) -> int:
    if await client.collection_exists(collection):
        await client.delete_collection(collection)
    await client.create_collection(
        collection, vectors_config=VectorParams(size=embedder.dimension, distance=Distance.COSINE)
    )
    await client.create_payload_index(collection, "source_type", field_schema=PayloadSchemaType.KEYWORD)
    for batch in batched(items, 128):
        vectors = await asyncio.to_thread(embedder.embed_documents, [item["text"] for item in batch])
        await client.upsert(
            collection,
            points=[
                PointStruct(
                    id=str(uuid.uuid5(_ID_NAMESPACE, item["key"])),
                    vector=vector,
                    payload={k: item[k] for k in ("text", "source_type", "title", "branche")},
                )
                for item, vector in zip(batch, vectors, strict=True)
            ],
        )
    return len(items)


async def main() -> None:
    settings = get_settings()
    conn = await connect()
    try:
        items = await collect_items(conn)
    finally:
        await conn.close()
    client = AsyncQdrantClient(host=settings.qdrant.host, port=settings.qdrant.port, timeout=60)
    try:
        count = await build_index(client, Embedder(settings.embedding_model), settings.qdrant.collection, items)
    finally:
        await client.close()
    print(f"Qdrant collection '{settings.qdrant.collection}': {count} points")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    asyncio.run(main())
