"""FastAPI application: chat page, chat API, quote form API and health check.

Run from the repository root:
    python -m uvicorn src.api.main:app --port 8005
"""
from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import get_settings
from ..core.orchestrator import HybridRAGOrchestrator
from ..core.quotation import QuotationError, VehicleInfo
from ..core.schemas import QueryResult
from ..core.sessions import SessionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
SESSION_COOKIE = "bhbot_session"
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{32}$")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if not hasattr(app.state, "orchestrator"):  # tests inject their own orchestrator
        settings = get_settings()
        app.state.session_ttl = settings.session_ttl_seconds
        app.state.orchestrator = await HybridRAGOrchestrator.create(settings)
    yield
    await app.state.orchestrator.close()


app = FastAPI(
    title="BHbot – RAG hybride pour l'assurance",
    description="Questions sur les garanties (Qdrant + Neo4j), devis auto et accès authentifié aux contrats.",
    version="3.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def session_cookie(request: Request, call_next: Any) -> Any:
    """Give every browser its own session id, stored in an HttpOnly cookie.

    Only ids issued by this server are accepted: an unknown id (forged, or expired)
    is replaced, so nobody can impose a session id on someone else's browser.
    """
    session_id = request.cookies.get(SESSION_COOKIE, "")
    orchestrator = getattr(request.app.state, "orchestrator", None)
    known = bool(_SESSION_ID_RE.match(session_id)) and orchestrator is not None and orchestrator.sessions.exists(session_id)
    is_new = not known
    if is_new:
        session_id = SessionStore.new_id()
        if orchestrator is not None:
            orchestrator.sessions.get(session_id)  # register it
    request.state.session_id = session_id
    response = await call_next(request)
    if is_new:
        response.set_cookie(
            SESSION_COOKIE,
            session_id,
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",
            max_age=getattr(request.app.state, "session_ttl", 7200),
        )
    return response


def orchestrator_of(request: Request) -> HybridRAGOrchestrator:
    return request.app.state.orchestrator


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/query", response_model=QueryResult)
async def query(body: QueryRequest, request: Request) -> QueryResult:
    return await orchestrator_of(request).answer(request.state.session_id, body.query)


@app.post("/api/session/reset")
async def reset_session(request: Request) -> dict[str, bool]:
    """Forget this browser's conversation and sign the client out."""
    orchestrator_of(request).reset(request.state.session_id)
    return {"success": True}


@app.post("/api/quotation")
async def quotation(vehicle: VehicleInfo, request: Request) -> JSONResponse:
    try:
        result = await orchestrator_of(request).quotation_client.quote(vehicle)
    except QuotationError as error:
        return JSONResponse(status_code=502, content={"error": True, "message": str(error)})
    return JSONResponse(content={"error": False, "quotation": result})


@app.get("/api/health")
async def health(request: Request) -> dict[str, Any]:
    return await orchestrator_of(request).health()


@app.exception_handler(Exception)
async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
    # Details go to the server log, not to the browser.
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Erreur interne, veuillez réessayer."})
