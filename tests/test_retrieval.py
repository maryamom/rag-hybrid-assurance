from langchain_core.documents import Document

from src.core.graph_schema import GRAPH_FULLTEXT_INDEX, PUBLIC_GRAPH_LABELS
from src.core.retrievers.graph import Neo4jKnowledgeRetriever, to_lucene_query
from src.core.retrievers.hybrid import HybridRetriever, reciprocal_rank_fusion
from src.core.retrievers.vector import QdrantKnowledgeRetriever


def doc(text, retriever="vector"):
    return Document(page_content=text, metadata={"retriever": retriever, "title": text, "kind": "x"})


def test_rrf_puts_passages_found_by_both_sources_first():
    vector = [doc("A"), doc("B"), doc("C")]
    graph = [doc("C", "graph"), doc("D", "graph")]
    fused = reciprocal_rank_fusion([vector, graph], top_n=4)
    assert fused[0].page_content == "C"
    assert {d.page_content for d in fused} == {"A", "B", "C", "D"}


def test_rrf_deduplicates_identical_text():
    fused = reciprocal_rank_fusion([[doc("Même texte")], [doc("même   TEXTE", "graph")]], top_n=5)
    assert len(fused) == 1


class Failing:
    async def ainvoke(self, query):
        raise ConnectionError("neo4j down")


class Fixed:
    def __init__(self, docs):
        self.docs = docs

    async def ainvoke(self, query):
        return self.docs


async def test_hybrid_survives_one_source_failing():
    retriever = HybridRetriever(retrievers=[Fixed([doc("A")]), Failing()])
    result = await retriever.ainvoke("question")
    assert [d.page_content for d in result] == ["A"]


def test_lucene_syntax_is_escaped():
    assert to_lucene_query('garantie "vol" (auto)?') == 'garantie \\"vol\\" \\(auto\\)\\?'
    assert to_lucene_query("   ") == ""


class RecordingDriver:
    def __init__(self):
        self.calls = []

    async def execute_query(self, query, **params):
        self.calls.append((query, params))
        return [{"label": "Garantie", "title": "Vol", "text": "Couvre le vol du véhicule.", "links": ["INCLUSE_DANS : Auto"], "score": 2.0}], None, None


async def test_graph_retriever_only_reads_public_labels():
    driver = RecordingDriver()
    docs = await Neo4jKnowledgeRetriever(driver=driver).ainvoke("vol de voiture")
    query, params = driver.calls[0]
    assert params["public_labels"] == list(PUBLIC_GRAPH_LABELS)
    assert params["index"] == GRAPH_FULLTEXT_INDEX
    assert "Client" not in PUBLIC_GRAPH_LABELS and "Contrat" not in PUBLIC_GRAPH_LABELS
    assert "CONTAINS" not in query
    assert docs[0].page_content.startswith("Couvre le vol")
    assert set(docs[0].metadata) == {"id", "retriever", "kind", "title", "score"}


class Point:
    def __init__(self, pid, score, payload):
        self.id, self.score, self.payload = pid, score, payload


class FakeQdrant:
    def __init__(self):
        self.kwargs = None

    async def query_points(self, **kwargs):
        self.kwargs = kwargs

        class Response:
            points = [Point(1, 0.9, {"text": "Texte", "source_type": "garantie", "title": "Vol", "secret": "x"})]

        return Response()


class FakeEmbedder:
    async def aembed_query(self, text):
        return [0.1, 0.2]


async def test_vector_retriever_filters_on_public_types_and_hides_payload():
    qdrant = FakeQdrant()
    docs = await QdrantKnowledgeRetriever(client=qdrant, embedder=FakeEmbedder(), collection="c").ainvoke("vol")
    condition = qdrant.kwargs["query_filter"].must[0]
    assert condition.key == "source_type"
    assert "contrat" not in condition.match.any and "personne" not in condition.match.any
    assert "secret" not in docs[0].metadata


class Slow:
    async def ainvoke(self, query):
        import asyncio

        await asyncio.sleep(5)
        return [doc("trop tard")]


async def test_hybrid_does_not_wait_forever_for_a_slow_source():
    retriever = HybridRetriever(retrievers=[Fixed([doc("A")]), Slow()], timeout_seconds=0.2)
    result = await retriever.ainvoke("question")
    assert [d.page_content for d in result] == ["A"]
