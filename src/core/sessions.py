"""Per-browser conversation state.

Each browser gets its own session id (an HttpOnly cookie set by the API), so
conversation history and authentication progress are never shared between users.
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

MAX_HISTORY_MESSAGES = 12  # last 6 exchanges are enough to resolve follow-up questions


@dataclass
class SessionState:
    history: list[BaseMessage] = field(default_factory=list)
    auth_step: Optional[str] = None  # None, "identity" or "code"
    pending_question: Optional[str] = None  # the personal-data question asked before authenticating
    client_token: Optional[str] = None  # set once the client has entered a valid code
    pending_quote: Optional[dict] = None  # vehicle details collected so far for a quote
    last_seen: float = field(default_factory=time.monotonic)

    def remember(self, question: str, answer: str) -> None:
        self.history.extend([HumanMessage(question), AIMessage(answer)])
        del self.history[:-MAX_HISTORY_MESSAGES]


class SessionStore:
    """In-memory store, one entry per session id, expired after `ttl_seconds` of inactivity.

    In-memory is fine for a single-process demo. Behind several workers, move this
    to Redis so that every worker sees the same sessions.
    """

    def __init__(self, ttl_seconds: int = 7200) -> None:
        self._ttl = ttl_seconds
        self._sessions: dict[str, SessionState] = {}

    @staticmethod
    def new_id() -> str:
        return secrets.token_urlsafe(24)

    def get(self, session_id: str) -> SessionState:
        self._expire_old()
        state = self._sessions.get(session_id)
        if state is None:
            state = SessionState()
            self._sessions[session_id] = state
        state.last_seen = time.monotonic()
        return state

    def exists(self, session_id: str) -> bool:
        self._expire_old()
        return session_id in self._sessions

    def reset(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._sessions)

    def _expire_old(self) -> None:
        now = time.monotonic()
        expired = [sid for sid, s in self._sessions.items() if now - s.last_seen > self._ttl]
        for sid in expired:
            del self._sessions[sid]
