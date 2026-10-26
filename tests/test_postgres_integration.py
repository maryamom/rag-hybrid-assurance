"""Integration tests against a real PostgreSQL.

Skipped unless TEST_POSTGRES_PORT is set (CI starts a PostgreSQL service and sets it).
Locally: docker compose up -d postgres, then
    TEST_POSTGRES_PORT=5432 POSTGRES_PASSWORD=... python -m pytest
The tests use a separate database, bhbot_test, never the application's database.
"""
import os

import asyncpg
import pandas as pd
import pytest

from ingestion import clean_documents, load_tables
from ingestion.common import connect
from src.auth.client_auth import PostgresClientRepository
from src.config import get_settings

pytestmark = pytest.mark.skipif("TEST_POSTGRES_PORT" not in os.environ, reason="no test PostgreSQL")


@pytest.fixture
async def conn(pg_env):
    connection = await connect()
    await connection.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
    yield connection
    await connection.close()


async def test_load_tables_creates_typed_tables(conn, tmp_path):
    frame = pd.DataFrame({
        "Nom Garantie": ["Vol", "Incendie"],
        "Prime (DT)": [120, 80],
        "Taux": [0.5, None],
        "Date effet": pd.to_datetime(["2024-01-01", "2024-06-30"]),
    })
    path = tmp_path / "Description garanties.xlsx"
    frame.to_excel(path, sheet_name="Sheet1", index=False)
    (table, data), = load_tables.read_tables(path).items()
    assert await load_tables.load_frame(conn, table, data) == 2
    assert await load_tables.load_frame(conn, table, data) == 2  # rerun is safe
    rows = await conn.fetch(f'SELECT * FROM "{table}" ORDER BY prime_dt')
    assert rows[0]["nom_garantie"] == "Incendie" and rows[0]["prime_dt"] == 80
    assert rows[0]["taux"] is None
    types = {r["column_name"]: r["data_type"] for r in await conn.fetch(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = $1", table)}
    assert types == {"nom_garantie": "text", "prime_dt": "bigint", "taux": "double precision",
                     "date_effet": "timestamp without time zone"}


async def test_clean_documents_from_ocr_tables(conn):
    await conn.execute("CREATE TABLE documents (id serial PRIMARY KEY, name text, filename text, file_path text, total_pages int)")
    await conn.execute("CREATE TABLE pages_content (id serial PRIMARY KEY, document_id int REFERENCES documents(id), page_number int, page_content text)")
    insert_doc = "INSERT INTO documents (name, filename, file_path, total_pages) VALUES ('cg', 'cg.pdf', $1, $2) RETURNING id"
    insert_page = "INSERT INTO pages_content (document_id, page_number, page_content) VALUES ($1, $2, $3)"

    first_run = await conn.fetchval(insert_doc, "data/raw/pdfs/auto/cg.pdf", 1)
    await conn.execute(insert_page, first_run, 1, "ancienne version")
    latest_run = await conn.fetchval(insert_doc, "data/raw/pdfs/auto/cg.pdf", 3)
    for n in (2, 1, 3):
        await conn.execute(insert_page, latest_run, n, f"EN-TÊTE BH\nArticle {n}")

    documents = await clean_documents.from_ocr_tables(conn)
    await clean_documents.save(conn, documents)
    row = await conn.fetchrow("SELECT * FROM cleaned_docs")
    assert row["branche"] == "auto" and row["pages"] == 3
    assert row["text_cleaned"] == "Article 1\n\nArticle 2\n\nArticle 3"  # latest run only, header removed


async def test_client_repository_queries(conn):
    await conn.execute("DROP TABLE IF EXISTS devis, contrats, clients CASCADE")
    await conn.execute("CREATE TABLE clients (id int PRIMARY KEY, nom text, prenom text, telephone text, email text)")
    await conn.execute("CREATE TABLE devis (id serial, client_id int, date_devis date, montant numeric)")
    await conn.execute("CREATE TABLE contrats (id serial, client_id int, date_effet date, produit text)")
    await conn.execute("INSERT INTO clients VALUES (7, 'Ben Ali', 'Mariem', '+216 98 765 432', 'm@example.com')")
    await conn.execute("INSERT INTO contrats (client_id, date_effet, produit) VALUES (7, '2024-01-01', 'Auto'), (8, '2024-01-01', 'Autre')")

    pool = await asyncpg.create_pool(get_settings().postgres.dsn, min_size=1, max_size=1)
    try:
        repo = PostgresClientRepository(pool)
        assert (await repo.find_client("ben ali", "MARIEM", "98765432"))["id"] == 7
        assert await repo.find_client("Ben Ali", "Mariem", "11111111") is None
        records = await repo.client_records(7)
        assert [c["produit"] for c in records["contrats"]] == ["Auto"]  # only this client's contracts
        assert "email" not in records["client"]
    finally:
        await pool.close()
