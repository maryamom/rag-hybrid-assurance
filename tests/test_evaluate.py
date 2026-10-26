from langchain_core.documents import Document

from eval.evaluate import evaluate, first_hit_rank, markdown_table, summarize


def test_first_hit_rank_ignores_case_and_accents():
    passages = ["Rien ici", "Le PARE-BRISE est couvert", "vitres"]
    assert first_hit_rank(passages, ["pare-brise"]) == 2
    assert first_hit_rank(passages, ["Vîtres"]) == 3
    assert first_hit_rank(passages, ["incendie"]) is None


def test_summarize():
    metrics = summarize([1, 2, None, 6], [0.1, 0.2, 0.3, 0.4], k=5)
    assert metrics["hit_rate"] == 0.5
    assert round(metrics["mrr"], 3) == round((1 + 0.5 + 1 / 6) / 4, 3)
    assert metrics["p50_ms"] == 250


class Fixed:
    def __init__(self, texts):
        self.texts = texts

    async def ainvoke(self, query):
        return [Document(page_content=t) for t in self.texts]


async def test_evaluate_and_table():
    questions = [{"question": "q", "expected": ["vol"]}]
    results = await evaluate({"good": Fixed(["vol du véhicule"]), "bad": Fixed(["incendie"])}, questions)
    assert results["good"]["hit_rate"] == 1 and results["bad"]["hit_rate"] == 0
    table = markdown_table(results, 1)
    assert "| good | 100% | 1.00 |" in table
