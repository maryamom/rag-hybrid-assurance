"""Step 6 – build the public knowledge graph in Neo4j.

    python -m ingestion.build_graph

Nodes (all public, see src/core/graph_schema.py):
    Branche, SousBranche, Produit, Garantie, ProfilCible, QuestionReponse
Relations:
    (Produit)-[:APPARTIENT_A]->(SousBranche)-[:FAIT_PARTIE_DE]->(Branche)
    (Produit)-[:INCLUT {nb_contrats}]->(Garantie)
    (Produit)-[:CIBLE]->(ProfilCible)

Only distinct product, branch and guarantee names are read from the contract
portfolio: no client, contract number or amount ever reaches the graph.
Every node gets `nom` (title) and `texte` (searchable description); a French
full-text index covers both.
"""
from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from neo4j import GraphDatabase

from src.config import get_settings
from src.core.detectors import normalize
from src.core.graph_schema import GRAPH_FULLTEXT_INDEX, PUBLIC_GRAPH_LABELS

from .common import connect, quote_ident, resolve, table_columns
from .schema import CONTRACT_GUARANTEES, FAQ, GUARANTEE_DESCRIPTIONS, PORTFOLIO_CONTRACTS, PRODUCT_PROFILES


@dataclass
class GraphData:
    nodes: dict[str, dict[str, dict[str, Any]]] = field(default_factory=lambda: defaultdict(dict))
    edges: dict[tuple[str, str, str], dict[tuple[str, str], dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    def node(self, label: str, name: Any, **properties: Any) -> str | None:
        """Add or enrich a node; returns its display name (None for empty names)."""
        if name is None or not str(name).strip():
            return None
        display = " ".join(str(name).split())
        key = normalize(display)
        existing = self.nodes[label].get(key)
        if existing is None:
            self.nodes[label][key] = {"nom": display, **{k: v for k, v in properties.items() if v}}
            return display
        for k, v in properties.items():
            if v and not existing.get(k):
                existing[k] = v
        return existing["nom"]

    def edge(self, start: tuple[str, str], rel: str, end: tuple[str, str], **properties: Any) -> None:
        (start_label, start_name), (end_label, end_name) = start, end
        if start_name and end_name:
            self.edges[(start_label, rel, end_label)][(start_name, end_name)] = properties

    def neighbours(self, label: str, name: str, rel: str, outgoing: bool = True) -> list[str]:
        result = []
        for (start_label, r, end_label), pairs in self.edges.items():
            if r != rel:
                continue
            for start_name, end_name in pairs:
                if outgoing and start_label == label and start_name == name:
                    result.append(end_name)
                elif not outgoing and end_label == label and end_name == name:
                    result.append(start_name)
        return sorted(set(result))


def _split_profiles(value: Any) -> list[str]:
    return [p.strip() for p in re.split(r"[,;/\n]+", str(value or "")) if p.strip()]


async def collect_graph(conn: Any) -> GraphData:
    tables = await table_columns(conn)
    data = GraphData()

    portfolio = resolve(PORTFOLIO_CONTRACTS, tables)
    if portfolio and portfolio.usable:
        branche = portfolio.col("branche") or "NULL"
        sous_branche = portfolio.col("sous_branche") or "NULL"
        rows = await conn.fetch(
            f"SELECT DISTINCT {branche}::text AS branche, {sous_branche}::text AS sous_branche, "
            f"{portfolio.col('produit')}::text AS produit FROM {quote_ident(portfolio.table)}"
        )
        for row in rows:
            product = data.node("Produit", row["produit"])
            branch = data.node("Branche", row["branche"])
            sub_branch = data.node("SousBranche", row["sous_branche"])
            if sub_branch:
                data.edge(("Produit", product), "APPARTIENT_A", ("SousBranche", sub_branch))
                data.edge(("SousBranche", sub_branch), "FAIT_PARTIE_DE", ("Branche", branch))
            else:
                data.edge(("Produit", product), "APPARTIENT_A", ("Branche", branch))

        guarantees = resolve(CONTRACT_GUARANTEES, tables)
        if guarantees and guarantees.usable and portfolio.col("num_contrat"):
            rows = await conn.fetch(
                f"SELECT c.{portfolio.col('produit')}::text AS produit, g.{guarantees.col('garantie')}::text AS garantie, "
                f"count(*) AS nb FROM {quote_ident(guarantees.table)} g "
                f"JOIN {quote_ident(portfolio.table)} c "
                f"ON c.{portfolio.col('num_contrat')}::text = g.{guarantees.col('num_contrat')}::text "
                "GROUP BY 1, 2"
            )
            for row in rows:
                product = data.node("Produit", row["produit"])
                guarantee = data.node("Garantie", row["garantie"])
                data.edge(("Produit", product), "INCLUT", ("Garantie", guarantee), nb_contrats=int(row["nb"]))

    descriptions = resolve(GUARANTEE_DESCRIPTIONS, tables)
    if descriptions and descriptions.usable:
        description = descriptions.col("description") or "NULL"
        kind = descriptions.col("type") or "NULL"
        rows = await conn.fetch(
            f"SELECT {descriptions.col('nom')}::text AS nom, {description}::text AS description, "
            f"{kind}::text AS type FROM {quote_ident(descriptions.table)}"
        )
        for row in rows:
            data.node("Garantie", row["nom"], description=row["description"], type=row["type"])

    profiles = resolve(PRODUCT_PROFILES, tables)
    if profiles and profiles.usable:
        rows = await conn.fetch(
            f"SELECT {profiles.col('produit')}::text AS produit, {profiles.col('profil')}::text AS profil "
            f"FROM {quote_ident(profiles.table)}"
        )
        for row in rows:
            product = data.node("Produit", row["produit"])
            for profile in _split_profiles(row["profil"]):
                data.edge(("Produit", product), "CIBLE", ("ProfilCible", data.node("ProfilCible", profile)))

    faq = resolve(FAQ, tables)
    if faq and faq.usable:
        rows = await conn.fetch(
            f"SELECT {faq.col('question')}::text AS question, {faq.col('reponse')}::text AS reponse "
            f"FROM {quote_ident(faq.table)}"
        )
        for row in rows:
            if row["question"] and row["reponse"]:
                question = " ".join(row["question"].split())
                data.node("QuestionReponse", question[:200], texte=f"Question : {question}\nRéponse : {row['reponse'].strip()}")

    _write_descriptions(data)
    return data


def _write_descriptions(data: GraphData) -> None:
    """Fill `texte`, the searchable description of every node."""
    for product in data.nodes["Produit"].values():
        name = product["nom"]
        parts = [f"Produit d'assurance {name}."]
        if sub := data.neighbours("Produit", name, "APPARTIENT_A"):
            parts.append(f"Rattaché à : {', '.join(sub)}.")
        if guarantees := data.neighbours("Produit", name, "INCLUT"):
            parts.append(f"Garanties : {', '.join(guarantees)}.")
        if targets := data.neighbours("Produit", name, "CIBLE"):
            parts.append(f"Profils cibles : {', '.join(targets)}.")
        product["texte"] = " ".join(parts)
    for guarantee in data.nodes["Garantie"].values():
        name = guarantee["nom"]
        parts = [f"Garantie {name}."]
        if guarantee.get("description"):
            parts.append(str(guarantee["description"]).strip())
        if guarantee.get("type"):
            parts.append(f"Type : {guarantee['type']}.")
        if products := data.neighbours("Garantie", name, "INCLUT", outgoing=False):
            parts.append(f"Incluse dans : {', '.join(products)}.")
        guarantee["texte"] = " ".join(parts)
    for label, word in (("Branche", "Contient"), ("SousBranche", "Produits")):
        for node in data.nodes[label].values():
            children = sorted(
                set(data.neighbours(label, node["nom"], "FAIT_PARTIE_DE", outgoing=False))
                | set(data.neighbours(label, node["nom"], "APPARTIENT_A", outgoing=False))
            )
            node["texte"] = f"{label} {node['nom']}." + (f" {word} : {', '.join(children)}." if children else "")
    for profile in data.nodes["ProfilCible"].values():
        products = data.neighbours("ProfilCible", profile["nom"], "CIBLE", outgoing=False)
        profile["texte"] = f"Profil client {profile['nom']}. Produits adaptés : {', '.join(products)}."


def write_graph(driver: Any, data: GraphData) -> dict[str, int]:
    labels = list(PUBLIC_GRAPH_LABELS)
    driver.execute_query(
        "MATCH (n) WHERE any(label IN labels(n) WHERE label IN $labels) DETACH DELETE n", labels=labels
    )
    for label in labels:
        driver.execute_query(
            f"CREATE CONSTRAINT {label.lower()}_nom IF NOT EXISTS FOR (n:{label}) REQUIRE n.nom IS UNIQUE"
        )
    counts = {}
    for label in labels:
        rows = list(data.nodes.get(label, {}).values())
        if rows:
            driver.execute_query(f"UNWIND $rows AS row MERGE (n:{label} {{nom: row.nom}}) SET n += row", rows=rows)
        counts[label] = len(rows)
    for (start_label, rel, end_label), pairs in data.edges.items():
        assert start_label in labels and end_label in labels and rel.isupper()
        rows = [{"a": a, "b": b, "props": props} for (a, b), props in pairs.items()]
        driver.execute_query(
            f"UNWIND $rows AS row MATCH (a:{start_label} {{nom: row.a}}) MATCH (b:{end_label} {{nom: row.b}}) "
            f"MERGE (a)-[r:{rel}]->(b) SET r += row.props",
            rows=rows,
        )
        counts[rel] = counts.get(rel, 0) + len(rows)
    driver.execute_query(f"DROP INDEX {GRAPH_FULLTEXT_INDEX} IF EXISTS")
    driver.execute_query(
        f"CREATE FULLTEXT INDEX {GRAPH_FULLTEXT_INDEX} FOR (n:{'|'.join(labels)}) ON EACH [n.nom, n.texte] "
        "OPTIONS {indexConfig: {`fulltext.analyzer`: 'french'}}"
    )
    driver.execute_query("CALL db.awaitIndexes(300)")
    return counts


async def main() -> None:
    settings = get_settings()
    conn = await connect()
    try:
        data = await collect_graph(conn)
    finally:
        await conn.close()
    with GraphDatabase.driver(settings.neo4j.uri, auth=(settings.neo4j.user, settings.neo4j.password)) as driver:
        counts = write_graph(driver, data)
    for name, count in counts.items():
        print(f"{name:<16} {count:>6}")


if __name__ == "__main__":
    asyncio.run(main())
