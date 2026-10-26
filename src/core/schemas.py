"""Data shapes shared by the orchestrator and the API."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Intent = Literal["general", "quotation", "personal_data", "authentication"]


class Source(BaseModel):
    """What the user may see about a retrieved passage: never the raw record."""

    title: str
    kind: str  # e.g. "document", "garantie", "faq", "graphe"
    score: float = 0.0


class QueryResult(BaseModel):
    response: str
    intent: Intent = "general"
    sources: list[Source] = Field(default_factory=list)
    quotation: Optional[dict[str, Any]] = None
    needs_auth: bool = False
    auth_step: Optional[Literal["identity", "code"]] = None
    vector_results_count: int = 0
    graph_results_count: int = 0
    processing_time: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
