"""Client authentication for personal-data questions: identity, then a code sent by e-mail.

Security choices:
- codes come from `secrets` (cryptographic), are stored hashed, and compared in constant time;
- a code is bound to the browser session that asked for it, expires after 10 minutes,
  and is locked after 3 wrong attempts;
- the answer is the same whether or not the client exists, and never shows the e-mail
  address, so the chat can't be used to discover who is a client;
- nothing personal is written to the logs.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import secrets
import smtplib
import time
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

CODE_TTL_SECONDS = 600
TOKEN_TTL_SECONDS = 3600
MAX_CODE_ATTEMPTS = 3
MAX_IDENTITY_ATTEMPTS = 5


class ClientRepository(Protocol):
    async def find_client(self, nom: str, prenom: str, telephone: str) -> Optional[dict[str, Any]]: ...
    async def client_records(self, client_id: Any) -> dict[str, Any]: ...


class CodeSender(Protocol):
    async def send_code(self, email: str, code: str, prenom: str) -> None: ...


@dataclass
class _Challenge:
    code_hash: str
    salt: str
    client_id: Optional[Any]  # None when the identity matched no client
    expires_at: float
    attempts: int = 0


@dataclass
class _ClientSession:
    client_id: Any
    expires_at: float


@dataclass
class VerifyResult:
    ok: bool
    token: Optional[str] = None
    attempts_left: int = 0
    expired: bool = False


@dataclass
class ClientAuthService:
    repository: ClientRepository
    sender: CodeSender
    _challenges: dict[str, _Challenge] = field(default_factory=dict)
    _identity_attempts: dict[str, list[float]] = field(default_factory=dict)
    _tokens: dict[str, _ClientSession] = field(default_factory=dict)

    @staticmethod
    def _hash(code: str, salt: str) -> str:
        return hashlib.sha256(f"{salt}:{code}".encode()).hexdigest()

    def identity_attempts_exceeded(self, session_id: str) -> bool:
        window_start = time.monotonic() - 900
        recent = [t for t in self._identity_attempts.get(session_id, []) if t > window_start]
        self._identity_attempts[session_id] = recent
        return len(recent) >= MAX_IDENTITY_ATTEMPTS

    async def start(self, session_id: str, nom: str, prenom: str, telephone: str) -> None:
        """Look the client up and, if found, e-mail a code. Behaves the same either way."""
        self._identity_attempts.setdefault(session_id, []).append(time.monotonic())
        client = await self.repository.find_client(nom, prenom, telephone)
        code = f"{secrets.randbelow(10**6):06d}"
        salt = secrets.token_hex(8)
        self._challenges[session_id] = _Challenge(
            code_hash=self._hash(code, salt),
            salt=salt,
            client_id=client["id"] if client else None,
            expires_at=time.monotonic() + CODE_TTL_SECONDS,
        )
        if client and client.get("email"):
            try:
                await self.sender.send_code(client["email"], code, client.get("prenom") or prenom)
            except Exception as error:  # noqa: BLE001 - must not reveal that the client exists
                logger.error("Could not send the verification code: %s", type(error).__name__)

    def verify(self, session_id: str, code: str) -> VerifyResult:
        challenge = self._challenges.get(session_id)
        if challenge is None:
            return VerifyResult(ok=False, attempts_left=0)
        if time.monotonic() > challenge.expires_at:
            del self._challenges[session_id]
            return VerifyResult(ok=False, expired=True)
        challenge.attempts += 1
        matches = hmac.compare_digest(challenge.code_hash, self._hash(code, challenge.salt))
        if matches and challenge.client_id is not None:
            del self._challenges[session_id]
            token = secrets.token_urlsafe(32)
            self._tokens[token] = _ClientSession(challenge.client_id, time.monotonic() + TOKEN_TTL_SECONDS)
            return VerifyResult(ok=True, token=token)
        attempts_left = MAX_CODE_ATTEMPTS - challenge.attempts
        if attempts_left <= 0:
            del self._challenges[session_id]
        return VerifyResult(ok=False, attempts_left=max(attempts_left, 0))

    def client_id_for(self, token: Optional[str]) -> Optional[Any]:
        if not token:
            return None
        session = self._tokens.get(token)
        if session is None or time.monotonic() > session.expires_at:
            self._tokens.pop(token, None)
            return None
        return session.client_id

    async def client_records(self, token: Optional[str]) -> Optional[dict[str, Any]]:
        client_id = self.client_id_for(token)
        if client_id is None:
            return None
        return await self.repository.client_records(client_id)

    def logout(self, token: Optional[str]) -> None:
        if token:
            self._tokens.pop(token, None)

    def forget_session(self, session_id: str) -> None:
        """Drop the pending code. Identity attempts are kept: they expire on their own after 15 min."""
        self._challenges.pop(session_id, None)


class PostgresClientRepository:
    """Reads clients, quotes and contracts from PostgreSQL with an asyncpg pool.

    Table names match ingestion/load_postgres.py.
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def find_client(self, nom: str, prenom: str, telephone: str) -> Optional[dict[str, Any]]:
        row = await self._pool.fetchrow(
            """
            SELECT id, nom, prenom, email
            FROM clients
            WHERE lower(nom) = lower($1)
              AND lower(prenom) = lower($2)
              AND right(regexp_replace(telephone, '\\D', '', 'g'), 8) = $3
            """,
            nom.strip(), prenom.strip(), telephone,
        )
        return dict(row) if row else None

    async def client_records(self, client_id: Any) -> dict[str, Any]:
        client = await self._pool.fetchrow(
            "SELECT id, nom, prenom FROM clients WHERE id = $1", client_id
        )
        quotes = await self._pool.fetch(
            "SELECT * FROM devis WHERE client_id = $1 ORDER BY date_devis DESC LIMIT 10", client_id
        )
        contracts = await self._pool.fetch(
            "SELECT * FROM contrats WHERE client_id = $1 ORDER BY date_effet DESC LIMIT 10", client_id
        )
        return {
            "client": dict(client) if client else {},
            "devis": [dict(r) for r in quotes],
            "contrats": [dict(r) for r in contracts],
        }


class SmtpCodeSender:
    """Sends the code by e-mail. smtplib is blocking, so it runs in a worker thread."""

    def __init__(self, host: str, port: int, user: str, password: str, sender: str) -> None:
        self._host, self._port = host, port
        self._user, self._password, self._sender = user, password, sender

    def _send(self, email: str, code: str, prenom: str) -> None:
        message = EmailMessage()
        message["Subject"] = "Votre code de vérification BHbot"
        message["From"] = self._sender
        message["To"] = email
        message.set_content(
            f"Bonjour {prenom},\n\n"
            f"Votre code de vérification est : {code}\n"
            "Il est valable 10 minutes.\n\n"
            "Si vous n'êtes pas à l'origine de cette demande, ignorez ce message.\n"
        )
        with smtplib.SMTP(self._host, self._port, timeout=20) as server:
            server.starttls()
            server.login(self._user, self._password)
            server.send_message(message)

    async def send_code(self, email: str, code: str, prenom: str) -> None:
        await asyncio.to_thread(self._send, email, code, prenom)
        logger.info("Verification code e-mailed")


class DisabledCodeSender:
    """Used when neither SMTP nor demo mode is configured: authentication can't complete."""

    async def send_code(self, email: str, code: str, prenom: str) -> None:
        logger.warning("No e-mail sender configured (set SMTP_* or AUTH_DEMO_MODE=true)")


class ConsoleCodeSender:
    """Demo mode (AUTH_DEMO_MODE=true): prints the code in the server console instead of e-mailing it.

    For local demos with the hackathon's fictitious clients only. Never enable it in production.
    """

    async def send_code(self, email: str, code: str, prenom: str) -> None:
        logger.warning("[DEMO MODE] verification code: %s", code)
