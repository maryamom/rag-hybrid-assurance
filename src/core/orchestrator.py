"""BHbot orchestrator: routes each question, then answers it.

Routing is rule-based and runs before any LLM call:
1. an authentication in progress for this session  -> identity / code steps
2. a question about the user's own data             -> authentication, then their records
3. a car-insurance quote request                    -> extraction + quote API
4. anything else                                    -> hybrid RAG (Qdrant + Neo4j, RRF fusion)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Awaitable, Callable, Optional

from langchain_core.output_parsers import StrOutputParser
from pydantic import ValidationError

from ..auth.client_auth import ClientAuthService
from .detectors import (
    PersonalDataDetector,
    QuotationDetector,
    normalize,
    parse_identity,
    parse_verification_code,
)
from .prompts import ANSWER_PROMPT, CLIENT_DATA_PROMPT, REPHRASE_PROMPT, format_context
from .quotation import (
    FIELD_LABELS,
    PartialVehicleInfo,
    QuotationClient,
    QuotationError,
    VehicleInfo,
    extract_vehicle_info,
    missing_fields_message,
)
from .schemas import QueryResult, Source
from .sessions import SessionState, SessionStore

logger = logging.getLogger(__name__)

CANCEL_WORDS = ("annuler", "annule", "stop", "laisse tomber", "abandonner")

MSG_ASK_IDENTITY = (
    "Pour accéder à vos informations personnelles, je dois d'abord vérifier votre identité.\n"
    "Indiquez votre prénom, votre nom et votre numéro de téléphone "
    "(par exemple : Mariem Ben Ali 98 765 432).\n"
    "Écrivez « annuler » pour revenir aux questions générales."
)
MSG_IDENTITY_INCOMPLETE = (
    "Il me faut votre prénom, votre nom et votre numéro de téléphone (8 chiffres), "
    "par exemple : Mariem Ben Ali 98 765 432. Ou écrivez « annuler »."
)
MSG_CODE_SENT = (
    "Merci. Si ces informations correspondent à un dossier client, un code à 6 chiffres vient "
    "d'être envoyé à l'adresse e-mail enregistrée. Il est valable 10 minutes : écrivez-le ici."
)
MSG_NO_DOCUMENTS = (
    "Je n'ai trouvé aucune information sur ce sujet dans la documentation. "
    "Essayez de reformuler, ou contactez une agence."
)

_CIN_LIKE = re.compile(r"(?<!\d)\d{8}(?!\d)")


def redact(text: str) -> str:
    """Remove identifiers (CIN, phone numbers) before a message is kept in history."""
    return _CIN_LIKE.sub("[numéro masqué]", text)


def format_records(records: dict[str, Any]) -> str:
    """Plain-text summary of a client's records, used when the LLM is disabled."""
    lines = []
    for title, key in (("Vos contrats", "contrats"), ("Vos devis", "devis")):
        rows = records.get(key) or []
        lines.append(f"{title} ({len(rows)}) :")
        for i, row in enumerate(rows, start=1):
            details = ", ".join(
                f"{k} : {v}" for k, v in row.items() if v not in (None, "") and k not in {"id", "client_id"}
            )
            lines.append(f"{i}. {details}")
    return "\n".join(lines)


def _invalid_fields(error: ValidationError) -> list[str]:
    return [str(item["loc"][0]) for item in error.errors() if item.get("loc")]


class HybridRAGOrchestrator:
    def __init__(
        self,
        *,
        retriever: Any,
        llm: Any,
        auth: Optional[ClientAuthService],
        quotation_client: QuotationClient,
        sessions: SessionStore,
        health_checks: Optional[dict[str, Callable[[], Awaitable[Any]]]] = None,
        closers: Optional[list[Callable[[], Awaitable[Any]]]] = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.auth = auth
        self.quotation_client = quotation_client
        self.sessions = sessions
        self.personal_detector = PersonalDataDetector()
        self.quote_detector = QuotationDetector()
        self._health_checks = health_checks or {}
        self._closers = closers or []

    # ---------------------------------------------------------------- lifecycle

    @classmethod
    async def create(cls, settings: Any) -> "HybridRAGOrchestrator":
        """Build the real clients from settings. Connections are opened lazily where possible."""
        import asyncpg
        from neo4j import AsyncGraphDatabase
        from qdrant_client import AsyncQdrantClient

        from ..auth.client_auth import (
            ConsoleCodeSender,
            DisabledCodeSender,
            PostgresClientRepository,
            SmtpCodeSender,
        )
        from .embeddings import Embedder
        from .retrievers.graph import Neo4jKnowledgeRetriever
        from .retrievers.hybrid import HybridRetriever
        from .retrievers.vector import QdrantKnowledgeRetriever

        qdrant = AsyncQdrantClient(host=settings.qdrant.host, port=settings.qdrant.port, timeout=10)
        neo4j_driver = AsyncGraphDatabase.driver(
            settings.neo4j.uri,
            auth=(settings.neo4j.user, settings.neo4j.password),
            connection_timeout=5,
            max_transaction_retry_time=5,
        )
        embedder = Embedder(settings.embedding_model)
        retriever = HybridRetriever(
            retrievers=[
                QdrantKnowledgeRetriever(client=qdrant, embedder=embedder, collection=settings.qdrant.collection),
                Neo4jKnowledgeRetriever(driver=neo4j_driver),
            ]
        )

        llm = None
        if settings.llm.enabled:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=settings.llm.model,
                api_key=settings.llm.api_key,
                base_url=settings.llm.base_url,
                temperature=settings.llm.temperature,
                timeout=30,
                max_retries=2,
                default_headers={"X-Title": "BHbot"},
            )
        else:
            logger.warning("OPENROUTER_API_KEY is empty: answers will list passages without the LLM")

        closers: list[Callable[[], Awaitable[Any]]] = [qdrant.close, neo4j_driver.close]
        checks: dict[str, Callable[[], Awaitable[Any]]] = {
            "qdrant": lambda: qdrant.get_collection(settings.qdrant.collection),
            "neo4j": neo4j_driver.verify_connectivity,
        }

        auth = None
        try:
            pool = await asyncpg.create_pool(dsn=settings.postgres.dsn, min_size=1, max_size=5)
        except (OSError, asyncpg.PostgresError) as error:
            logger.error("PostgreSQL unavailable, personal-data access disabled: %s", type(error).__name__)

            async def postgres_down() -> None:
                raise ConnectionError("PostgreSQL was unavailable at startup")

            checks["postgres"] = postgres_down
        else:
            if settings.auth_demo_mode:
                sender = ConsoleCodeSender()
                logger.warning("AUTH_DEMO_MODE is on: verification codes are printed in this console")
            elif settings.smtp.configured:
                s = settings.smtp
                sender = SmtpCodeSender(s.host, s.port, s.user, s.password, s.sender)
            else:
                sender = DisabledCodeSender()
            auth = ClientAuthService(PostgresClientRepository(pool), sender)
            closers.append(pool.close)
            checks["postgres"] = lambda: pool.fetchval("SELECT 1")

        return cls(
            retriever=retriever,
            llm=llm,
            auth=auth,
            quotation_client=QuotationClient(settings.quotation_api_url),
            sessions=SessionStore(settings.session_ttl_seconds),
            health_checks=checks,
            closers=closers,
        )

    async def close(self) -> None:
        for close in self._closers:
            try:
                await close()
            except Exception as error:  # noqa: BLE001 - best effort on shutdown
                logger.warning("Error while closing a connection: %s", type(error).__name__)

    async def health(self) -> dict[str, Any]:
        services: dict[str, bool] = {}
        for name, check in self._health_checks.items():
            try:
                await asyncio.wait_for(check(), timeout=3)
                services[name] = True
            except Exception:  # noqa: BLE001 - any failure means "down"
                services[name] = False
        services["llm"] = self.llm is not None
        core = [services.get(name, False) for name in ("qdrant", "neo4j")]
        return {"status": "healthy" if all(services.values()) else ("degraded" if any(core) else "down"),
                "services": services}

    def reset(self, session_id: str) -> None:
        state = self.sessions.reset(session_id)
        if self.auth:
            self.auth.forget_session(session_id)
            if state:
                self.auth.logout(state.client_token)

    # ---------------------------------------------------------------- routing

    async def answer(self, session_id: str, question: str) -> QueryResult:
        started = time.perf_counter()
        state = self.sessions.get(session_id)
        result = await self._route(session_id, state, question.strip())
        result.processing_time = round(time.perf_counter() - started, 3)
        # No question text, no personal data: only what is needed to monitor the service.
        logger.info("intent=%s sources=%d time=%.2fs", result.intent, len(result.sources), result.processing_time)
        return result

    async def _route(self, session_id: str, state: SessionState, question: str) -> QueryResult:
        if state.auth_step:
            return await self._continue_auth(session_id, state, question)
        if self.personal_detector.needs_personal_data(question):
            return await self._personal_data(session_id, state, question)
        if state.pending_quote is not None or self.quote_detector.needs_quotation(question):
            result = await self._quotation(state, question)
            if result is not None:
                return result
        return await self._rag(state, question)

    # ---------------------------------------------------------------- general questions

    async def _rag(self, state: SessionState, question: str) -> QueryResult:
        history = list(state.history)
        search_query = question
        if self.llm is not None and history:
            search_query = await (REPHRASE_PROMPT | self.llm | StrOutputParser()).ainvoke(
                {"history": history, "question": question}
            )

        documents = await self.retriever.ainvoke(search_query)
        sources = [
            Source(
                title=str(doc.metadata.get("title", "Document")),
                kind=str(doc.metadata.get("kind", "document")),
                score=round(float(doc.metadata.get("rrf", doc.metadata.get("score", 0.0))), 4),
            )
            for doc in documents
        ]

        if not documents:
            response = MSG_NO_DOCUMENTS
        elif self.llm is None:
            excerpts = "\n\n".join(
                f"[{i}] {doc.metadata.get('title', '')} : {doc.page_content[:300]}…"
                for i, doc in enumerate(documents[:3], start=1)
            )
            response = (
                "Le modèle de langage n'est pas configuré sur ce serveur. "
                f"Voici les passages les plus pertinents :\n\n{excerpts}"
            )
        else:
            response = await (ANSWER_PROMPT | self.llm | StrOutputParser()).ainvoke(
                {"context": format_context(documents), "history": history, "question": question}
            )

        state.remember(redact(question), response)
        return QueryResult(
            intent="general",
            response=response,
            sources=sources,
            vector_results_count=sum(1 for d in documents if d.metadata.get("retriever") == "vector"),
            graph_results_count=sum(1 for d in documents if d.metadata.get("retriever") == "graph"),
        )

    # ---------------------------------------------------------------- quotes

    async def _quotation(self, state: SessionState, question: str) -> Optional[QueryResult]:
        product = self.quote_detector.product_type(question)
        if state.pending_quote is None and product not in (None, "auto"):
            return QueryResult(
                intent="quotation",
                response=(
                    "Pour le moment, je ne calcule que des devis d'assurance auto. "
                    "Pour une autre assurance, une agence peut vous établir un devis."
                ),
            )

        found = await extract_vehicle_info(question, self.llm)
        new_fields = {k: v for k, v in found.model_dump().items() if v is not None}
        if state.pending_quote is not None and not new_fields and not self.quote_detector.needs_quotation(question):
            state.pending_quote = None  # the user moved on to another question
            return None

        merged = {**(state.pending_quote or {}), **new_fields}
        missing = PartialVehicleInfo(**merged).missing()
        if missing:
            state.pending_quote = merged
            return QueryResult(intent="quotation", response=missing_fields_message(missing))

        try:
            vehicle = VehicleInfo(**merged)
        except ValidationError as error:
            invalid = _invalid_fields(error)
            state.pending_quote = {k: v for k, v in merged.items() if k not in invalid}
            labels = "\n".join(f"- {FIELD_LABELS.get(name, name)}" for name in invalid)
            return QueryResult(
                intent="quotation",
                response=f"Ces informations ne semblent pas valides, pouvez-vous les vérifier ?\n{labels}",
            )

        state.pending_quote = None
        try:
            quote = await self.quotation_client.quote(vehicle)
        except QuotationError as error:
            return QueryResult(intent="quotation", response=str(error))
        response = "Voici votre devis auto indicatif, calculé par le service de tarification."
        state.remember(redact(question), response)
        return QueryResult(intent="quotation", response=response, quotation=quote)

    # ---------------------------------------------------------------- personal data

    async def _personal_data(self, session_id: str, state: SessionState, question: str) -> QueryResult:
        if self.auth is None:
            return QueryResult(
                intent="personal_data",
                response="L'accès aux données personnelles n'est pas disponible sur ce serveur pour le moment.",
            )
        records = await self.auth.client_records(state.client_token)
        if records is not None:
            return await self._answer_from_records(state, question, records)

        state.auth_step = "identity"
        state.pending_question = question
        identity = parse_identity(question)
        if identity.complete:
            return await self._start_auth(session_id, state, identity)
        return QueryResult(intent="authentication", response=MSG_ASK_IDENTITY, needs_auth=True, auth_step="identity")

    async def _start_auth(self, session_id: str, state: SessionState, identity: Any) -> QueryResult:
        assert self.auth is not None
        if self.auth.identity_attempts_exceeded(session_id):
            self._clear_auth(session_id, state)
            return QueryResult(
                intent="authentication",
                response="Trop de tentatives d'identification. Réessayez dans 15 minutes.",
            )
        await self.auth.start(session_id, identity.nom, identity.prenom, identity.telephone)
        state.auth_step = "code"
        return QueryResult(intent="authentication", response=MSG_CODE_SENT, needs_auth=True, auth_step="code")

    async def _continue_auth(self, session_id: str, state: SessionState, question: str) -> QueryResult:
        if self.auth is None:
            self._clear_auth(session_id, state)
            return await self._personal_data(session_id, state, question)
        if any(word in normalize(question) for word in CANCEL_WORDS):
            self._clear_auth(session_id, state)
            return QueryResult(
                intent="general",
                response="D'accord, j'ai annulé la vérification. Que puis-je faire pour vous ?",
            )

        if state.auth_step == "identity":
            identity = parse_identity(question)
            if not identity.complete:
                return QueryResult(
                    intent="authentication", response=MSG_IDENTITY_INCOMPLETE, needs_auth=True, auth_step="identity"
                )
            return await self._start_auth(session_id, state, identity)

        code = parse_verification_code(question)
        if code is None:
            return QueryResult(
                intent="authentication",
                response="Écrivez le code à 6 chiffres reçu par e-mail, ou « annuler ».",
                needs_auth=True,
                auth_step="code",
            )
        verdict = self.auth.verify(session_id, code)
        if verdict.ok:
            pending = state.pending_question or "mes contrats"
            self._clear_auth(session_id, state)
            state.client_token = verdict.token
            records = await self.auth.client_records(verdict.token) or {}
            return await self._answer_from_records(state, pending, records, prefix="Identité vérifiée.\n\n")
        if verdict.expired or verdict.attempts_left == 0:
            self._clear_auth(session_id, state)
            return QueryResult(
                intent="authentication",
                response="Le code a expiré ou le nombre d'essais est dépassé. Écrivez « mes contrats » pour recommencer.",
            )
        return QueryResult(
            intent="authentication",
            response=f"Code incorrect. Il vous reste {verdict.attempts_left} essai(s).",
            needs_auth=True,
            auth_step="code",
        )

    async def _answer_from_records(
        self, state: SessionState, question: str, records: dict[str, Any], prefix: str = ""
    ) -> QueryResult:
        if self.llm is None:
            body = format_records(records)
        else:
            body = await (CLIENT_DATA_PROMPT | self.llm | StrOutputParser()).ainvoke(
                {"records": json.dumps(records, default=str, ensure_ascii=False), "question": question}
            )
        # The client's data stays out of the conversation history that later prompts reuse.
        state.remember(redact(question), "[réponse basée sur les données du client]")
        return QueryResult(intent="personal_data", response=prefix + body)

    @staticmethod
    def _clear_auth_state(state: SessionState) -> None:
        state.auth_step = None
        state.pending_question = None

    def _clear_auth(self, session_id: str, state: SessionState) -> None:
        self._clear_auth_state(state)
        if self.auth:
            self.auth.forget_session(session_id)
