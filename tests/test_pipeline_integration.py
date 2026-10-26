"""Pipeline tests on hackathon-shaped tables (fictitious rows).

PostgreSQL tests need TEST_POSTGRES_PORT; the Neo4j round trip also needs TEST_NEO4J_URI.
Both are set in GitHub Actions (.github/workflows/tests.yml).
"""
import os

import asyncpg
import pandas as pd
import pytest

from ingestion import build_client_views, build_graph, check_data, load_tables
from ingestion.common import connect, table_columns
from src.auth.client_auth import PostgresClientRepository
from src.config import get_settings

pytestmark = pytest.mark.skipif("TEST_POSTGRES_PORT" not in os.environ, reason="no test PostgreSQL")

HACKATHON_FILES = {
    "Données assurance S1.xlsx": {
        "clients": pd.DataFrame({
            "id": [1, 2], "nom": ["Ben Ali", "Trabelsi"], "prénom": ["Mariem", "Sami"],
            "téléphone": ["+216 98 765 432", "22 333 444"], "email": ["m@example.com", "s@example.com"],
            "adresse": ["Tunis", "Sfax"],
        }),
        "devis": pd.DataFrame({"client_id": [1], "num_devis": ["D-1"], "montant": [450.0],
                               "created_at": pd.to_datetime(["2025-05-01"])}),
        "contrats": pd.DataFrame({"client_id": [1, 2], "num_contrat": ["C-1", "C-2"], "produit": ["Auto Confort", "Habitation"],
                                  "dateEffet": pd.to_datetime(["2025-01-01", "2024-03-01"])}),
    },
    "Données assurance S1.2 S2.2.xlsx": {
        "contrats": pd.DataFrame({
            "NUM_CONTRAT": ["A1", "A2", "H1"], "LIB_BRANCHE": ["IARD", "IARD", "IARD"],
            "LIB_SOUS_BRANCHE": ["Automobile", "Automobile", "Habitation"],
            "LIB_PRODUIT": ["Auto Confort", "Auto Confort", "Multirisque Habitation"],
        }),
        "garantie_contrat": pd.DataFrame({"NUM_CONTRAT": ["A1", "A2", "A1", "H1"],
                                          "LIB_GARANTIE": ["Bris de glace", "Bris de glace", "Vol", "Incendie"]}),
        "personne_physique": pd.DataFrame({"NOM_PRENOM": ["Personne Secrète"], "NUM_CONTRAT": ["A1"]}),
    },
    "Description garanties.xlsx": {
        "Sheet1": pd.DataFrame({"nom_garantie": ["Bris de glace", "vol"], "description": ["Pare-brise et vitres.", "Vol du véhicule."]}),
    },
    "Mapping produits vs profils cibles.xlsx": {
        "Sheet1": pd.DataFrame({"Produit": ["Auto Confort"], "Profil cible": ["Jeune conducteur; Famille"]}),
    },
    "Exemples questions réponses.xlsx": {
        "QR": pd.DataFrame({"Question": ["Comment déclarer un sinistre ?"], "Réponse": ["Dans les 5 jours ouvrés."]}),
    },
}


@pytest.fixture
async def loaded(pg_env, tmp_path):
    conn = await connect()
    await conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
    for file_name, sheets in HACKATHON_FILES.items():
        path = tmp_path / file_name
        with pd.ExcelWriter(path) as writer:
            for sheet, frame in sheets.items():
                frame.to_excel(writer, sheet_name=sheet, index=False)
        for table, frame in load_tables.read_tables(path).items():
            await load_tables.load_frame(conn, table, frame)
    yield conn
    await conn.close()


async def test_check_data_finds_everything(loaded):
    lines, all_ok = check_data.report(await table_columns(loaded))
    assert all_ok, "\n".join(lines)


async def test_client_views_and_repository(loaded):
    assert await build_client_views.build_views(loaded) == ["clients", "devis", "contrats"]
    pool = await asyncpg.create_pool(get_settings().postgres.dsn, min_size=1, max_size=1)
    try:
        repo = PostgresClientRepository(pool)
        client = await repo.find_client("Ben Ali", "Mariem", "98765432")
        assert client["id"] == "1" and client["email"] == "m@example.com"
        records = await repo.client_records(client["id"])
        assert [c["num_contrat"] for c in records["contrats"]] == ["C-1"]
        assert records["devis"][0]["num_devis"] == "D-1"
    finally:
        await pool.close()


async def test_graph_contains_only_public_catalogue(loaded):
    data = await build_graph.collect_graph(loaded)
    names = {label: {n["nom"] for n in nodes.values()} for label, nodes in data.nodes.items()}
    assert names["Produit"] == {"Auto Confort", "Multirisque Habitation"}
    assert names["Garantie"] == {"Bris de glace", "Vol", "Incendie"}  # 'vol' merged with 'Vol'
    assert names["ProfilCible"] == {"Jeune conducteur", "Famille"}
    assert ("Auto Confort", "Bris de glace") in data.edges[("Produit", "INCLUT", "Garantie")]
    assert data.edges[("Produit", "INCLUT", "Garantie")][("Auto Confort", "Bris de glace")] == {"nb_contrats": 2}
    all_text = str(dict(data.nodes)) + str(dict(data.edges))
    assert "Secrète" not in all_text and "A1" not in all_text  # no person, no contract number
    glass = next(n for n in data.nodes["Garantie"].values() if n["nom"] == "Bris de glace")
    assert "Pare-brise" in glass["texte"] and "Auto Confort" in glass["texte"]


@pytest.mark.skipif("TEST_NEO4J_URI" not in os.environ, reason="no test Neo4j")
async def test_graph_round_trip_in_neo4j(loaded):
    from neo4j import AsyncGraphDatabase, GraphDatabase

    from src.core.retrievers.graph import Neo4jKnowledgeRetriever

    data = await build_graph.collect_graph(loaded)
    auth = ("neo4j", os.environ["TEST_NEO4J_PASSWORD"])
    with GraphDatabase.driver(os.environ["TEST_NEO4J_URI"], auth=auth) as driver:
        counts = build_graph.write_graph(driver, data)
        driver.execute_query("CREATE (:PersonnePhysique {nom: 'Bris de glace secret', texte: 'Bris de glace'})")
    assert counts["Garantie"] == 3 and counts["INCLUT"] == 3

    async with AsyncGraphDatabase.driver(os.environ["TEST_NEO4J_URI"], auth=auth) as driver:
        docs = await Neo4jKnowledgeRetriever(driver=driver).ainvoke("bris de glace pare-brise")
    assert docs and docs[0].metadata["title"] == "Bris de glace"
    assert all("secret" not in d.page_content for d in docs)  # non-public labels are never returned
