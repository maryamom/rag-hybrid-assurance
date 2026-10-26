"""Measure retrieval quality and latency: vector only vs graph only vs hybrid.

    python -m eval.evaluate eval/questions.jsonl

Each line of the questions file is a JSON object:
    {"question": "Que couvre la garantie bris de glace ?", "expected": ["pare-brise", "vitres"]}
A question counts as answered at rank r when one of the `expected` phrases appears
(case and accents ignored) in the r-th retrieved passage.

Prints a Markdown table (hit rate @5, MRR, median and 95th-percentile latency) that
can be pasted into the README. Write the questions from the hackathon documents,
not from the FAQ table, which is itself indexed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from statistics import median
from typing import Any

from src.config import get_settings
from src.core.detectors import normalize


def first_hit_rank(passages: list[str], expected: list[str]) -> int | None:
    targets = [normalize(e) for e in expected]
    for rank, passage in enumerate(passages, start=1):
        text = normalize(passage)
        if any(t in text for t in targets):
            return rank
    return None


def percentile(values: list[float], share: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(share * (len(ordered) - 1))))
    return ordered[index]


def summarize(ranks: list[int | None], latencies: list[float], k: int = 5) -> dict[str, float]:
    n = len(ranks)
    return {
        "hit_rate": sum(1 for r in ranks if r is not None and r <= k) / n,
        "mrr": sum(1 / r for r in ranks if r is not None) / n,
        "p50_ms": median(latencies) * 1000,
        "p95_ms": percentile(latencies, 0.95) * 1000,
    }


async def evaluate(retrievers: dict[str, Any], questions: list[dict[str, Any]], k: int = 5) -> dict[str, dict[str, float]]:
    results = {}
    for name, retriever in retrievers.items():
        ranks, latencies = [], []
        for item in questions:
            started = time.perf_counter()
            documents = await retriever.ainvoke(item["question"])
            latencies.append(time.perf_counter() - started)
            ranks.append(first_hit_rank([d.page_content for d in documents[:k]], item["expected"]))
        results[name] = summarize(ranks, latencies, k)
    return results


def markdown_table(results: dict[str, dict[str, float]], n_questions: int, k: int = 5) -> str:
    lines = [
        f"| Retrieval ({n_questions} questions) | Hit rate @{k} | MRR | Latency p50 | Latency p95 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, m in results.items():
        lines.append(f"| {name} | {m['hit_rate']:.0%} | {m['mrr']:.2f} | {m['p50_ms']:.0f} ms | {m['p95_ms']:.0f} ms |")
    return "\n".join(lines)


async def main(questions_file: Path, k: int) -> None:
    from neo4j import AsyncGraphDatabase
    from qdrant_client import AsyncQdrantClient

    from src.core.embeddings import Embedder
    from src.core.retrievers.graph import Neo4jKnowledgeRetriever
    from src.core.retrievers.hybrid import HybridRetriever
    from src.core.retrievers.vector import QdrantKnowledgeRetriever

    settings = get_settings()
    questions = [json.loads(line) for line in questions_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    qdrant = AsyncQdrantClient(host=settings.qdrant.host, port=settings.qdrant.port)
    driver = AsyncGraphDatabase.driver(settings.neo4j.uri, auth=(settings.neo4j.user, settings.neo4j.password))
    try:
        vector = QdrantKnowledgeRetriever(client=qdrant, embedder=Embedder(settings.embedding_model),
                                          collection=settings.qdrant.collection, k=k)
        graph = Neo4jKnowledgeRetriever(driver=driver, k=k)
        await vector.ainvoke("échauffement")  # load the embedding model before timing
        results = await evaluate(
            {"Vector only (Qdrant)": vector, "Graph only (Neo4j)": graph,
             "Hybrid (RRF)": HybridRetriever(retrievers=[vector, graph], top_n=k)},
            questions, k,
        )
    finally:
        await qdrant.close()
        await driver.close()
    print(markdown_table(results, len(questions), k))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("questions", type=Path)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main(args.questions, args.k))
