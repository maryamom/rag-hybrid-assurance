import hashlib
import math

import pandas as pd
from qdrant_client import AsyncQdrantClient

from ingestion.clean_documents import branch_of, clean_document
from ingestion.common import slugify
from ingestion.index_qdrant import build_index, chunk_text, row_to_text, source_type_of
from ingestion.load_tables import _sql_type, read_tables
from src.core.retrievers.vector import QdrantKnowledgeRetriever


class HashEmbedder:
    """Deterministic 16-dim bag-of-words vectors: similar words -> similar vectors."""

    dimension = 16

    def _vector(self, text):
        vector = [0.0] * self.dimension
        for word in text.lower().split():
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    async def aembed_query(self, text):
        return self._vector(text)


def test_slugify():
    assert slugify("Données assurance S1.2") == "donnees_assurance_s1_2"
    assert slugify("2024 primes") == "t_2024_primes"


def test_read_tables_one_table_per_sheet(tmp_path):
    path = tmp_path / "Description garanties.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Nom Garantie": ["Vol", "Incendie"], "Prime (DT)": [120, 80]}).to_excel(writer, sheet_name="Sheet1", index=False)
        pd.DataFrame().to_excel(writer, sheet_name="Vide", index=False)
    tables = read_tables(path)
    assert list(tables) == ["description_garanties_sheet1"]
    frame = tables["description_garanties_sheet1"]
    assert list(frame.columns) == ["nom_garantie", "prime_dt"]
    assert _sql_type(frame["prime_dt"]) == "bigint" and _sql_type(frame["nom_garantie"]) == "text"


def test_branch_and_cleaning():
    assert branch_of("data/raw/pdfs/auto/cg.pdf") == "auto"
    assert branch_of("data/raw/pdfs/cg.pdf") == "general"
    pages = [f"BH ASSURANCE\nArticle {i} : texte   utile\n```\n{i}/3" for i in range(1, 4)]
    text = clean_document(pages)
    assert "BH ASSURANCE" not in text and "```" not in text and "1/3" not in text
    assert "Article 2 : texte utile" in text


def test_chunks_respect_size_and_keep_all_text():
    text = "\n\n".join(f"Paragraphe {i} " + "mot " * 60 for i in range(10))
    chunks = chunk_text(text, size=400, overlap=50)
    assert all(len(c) <= 400 for c in chunks)
    assert all(f"Paragraphe {i}" in " ".join(chunks) for i in range(10))


def test_only_reference_tables_are_public():
    assert source_type_of("description_garanties_sheet1") == "garantie"
    assert source_type_of("exemples_questions_reponses_qr") == "faq"
    assert source_type_of("donnees_assurance_s1_2_s2_2_personne_physique") is None
    assert source_type_of("donnees_assurance_s1_2_s2_2_contrats") is None


def test_row_to_text():
    text, title = row_to_text({"nom_garantie": "Vol", "description": None, "prime": 120})
    assert text == "nom_garantie: Vol | prime: 120" and title == "Vol"


async def test_index_then_retrieve_with_in_memory_qdrant():
    client = AsyncQdrantClient(location=":memory:")
    items = [
        {"key": "1", "text": "garantie vol du véhicule", "source_type": "garantie", "title": "Vol", "branche": ""},
        {"key": "2", "text": "incendie habitation dommages", "source_type": "document", "title": "CG", "branche": "habitation"},
        {"key": "3", "text": "garantie vol client Ben Ali", "source_type": "contrat", "title": "secret", "branche": ""},
    ]
    embedder = HashEmbedder()
    assert await build_index(client, embedder, "test", items) == 3

    retriever = QdrantKnowledgeRetriever(client=client, embedder=embedder, collection="test", k=5)
    docs = await retriever.ainvoke("garantie vol")
    assert docs[0].metadata["title"] == "Vol"
    assert all(d.metadata["kind"] != "contrat" for d in docs)  # non-public types are filtered out
    await client.close()
