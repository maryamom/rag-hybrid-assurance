"""Test doubles: no database, no network, no real LLM."""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.documents import Document

from src.auth.client_auth import ClientAuthService
from src.core.orchestrator import HybridRAGOrchestrator
from src.core.quotation import VehicleInfo
from src.core.sessions import SessionStore

CLIENT = {"id": 7, "nom": "Ben Ali", "prenom": "Mariem", "email": "mariem@example.com", "telephone": "98765432"}


class FakeRepository:
    async def find_client(self, nom: str, prenom: str, telephone: str) -> Optional[dict[str, Any]]:
        same = (nom.lower(), prenom.lower(), telephone) == (CLIENT["nom"].lower(), CLIENT["prenom"].lower(), CLIENT["telephone"])
        return dict(CLIENT) if same else None

    async def client_records(self, client_id: Any) -> dict[str, Any]:
        assert client_id == CLIENT["id"]
        return {
            "client": {"id": 7, "nom": "Ben Ali", "prenom": "Mariem"},
            "contrats": [{"id": 1, "client_id": 7, "produit": "Auto", "statut": "Actif"}],
            "devis": [],
        }


class CapturingSender:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send_code(self, email: str, code: str, prenom: str) -> None:
        self.sent.append((email, code))

    @property
    def last_code(self) -> str:
        return self.sent[-1][1]


class FakeRetriever:
    def __init__(self, documents: Optional[list[Document]] = None) -> None:
        self.documents = documents if documents is not None else [
            Document(
                page_content="La garantie bris de glace couvre le pare-brise et les vitres latérales.",
                metadata={"retriever": "vector", "kind": "garantie", "title": "Bris de glace", "score": 0.82},
            ),
            Document(
                page_content="Bris de glace\nRelations : INCLUSE_DANS : Auto Confort",
                metadata={"retriever": "graph", "kind": "Garantie", "title": "Bris de glace", "score": 3.1},
            ),
        ]
        self.queries: list[str] = []

    async def ainvoke(self, query: str) -> list[Document]:
        self.queries.append(query)
        return self.documents


class FakeQuotationClient:
    def __init__(self) -> None:
        self.calls: list[VehicleInfo] = []

    async def quote(self, vehicle: VehicleInfo) -> dict[str, Any]:
        self.calls.append(vehicle)
        return {"prime": 812.5, "pack": "Confort"}


def make_orchestrator(llm: Any = None, retriever: Any = None) -> tuple[HybridRAGOrchestrator, CapturingSender]:
    sender = CapturingSender()
    orchestrator = HybridRAGOrchestrator(
        retriever=retriever or FakeRetriever(),
        llm=llm,
        auth=ClientAuthService(FakeRepository(), sender),
        quotation_client=FakeQuotationClient(),
        sessions=SessionStore(),
    )
    return orchestrator, sender
