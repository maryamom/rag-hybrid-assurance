"""Application settings, read from environment variables (see .env.example).

Nothing is read at import time: call get_settings() where a value is needed.
That keeps the modules importable in tests without databases or secrets.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


class MissingSetting(RuntimeError):
    """A required environment variable is not set."""


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise MissingSetting(
            f"Missing environment variable: {name} (copy .env.example to .env and fill it in)"
        )
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PostgresSettings:
    host: str
    port: int
    database: str
    user: str
    password: str

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    user: str
    password: str


@dataclass(frozen=True)
class QdrantSettings:
    host: str
    port: int
    collection: str


@dataclass(frozen=True)
class LLMSettings:
    api_key: str  # empty string = LLM disabled, the API still starts
    model: str
    base_url: str
    temperature: float

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class SmtpSettings:
    host: str
    port: int
    user: str
    password: str
    sender: str

    @property
    def configured(self) -> bool:
        return bool(self.host and self.user and self.password)


@dataclass(frozen=True)
class Settings:
    postgres: PostgresSettings
    neo4j: Neo4jSettings
    qdrant: QdrantSettings
    llm: LLMSettings
    smtp: SmtpSettings
    embedding_model: str
    quotation_api_url: str
    auth_demo_mode: bool
    session_ttl_seconds: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        postgres=PostgresSettings(
            host=_env("POSTGRES_HOST", "localhost"),
            port=int(_env("POSTGRES_PORT", "5432")),
            database=_env("POSTGRES_DB", "data"),
            user=_env("POSTGRES_USER", "postgres"),
            password=_env("POSTGRES_PASSWORD"),
        ),
        neo4j=Neo4jSettings(
            uri=_env("NEO4J_URI", "bolt://localhost:7687"),
            user=_env("NEO4J_USER", "neo4j"),
            password=_env("NEO4J_PASSWORD"),
        ),
        qdrant=QdrantSettings(
            host=_env("QDRANT_HOST", "localhost"),
            port=int(_env("QDRANT_PORT", "6333")),
            collection=_env("QDRANT_COLLECTION", "insurance_knowledge"),
        ),
        llm=LLMSettings(
            api_key=_env("OPENROUTER_API_KEY", ""),
            model=_env("LLM_MODEL", "deepseek/deepseek-chat-v3.1"),
            base_url=_env("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=float(_env("LLM_TEMPERATURE", "0.1")),
        ),
        smtp=SmtpSettings(
            host=_env("SMTP_HOST", ""),
            port=int(_env("SMTP_PORT", "587")),
            user=_env("SMTP_USER", ""),
            password=_env("SMTP_PASSWORD", ""),
            sender=_env("SMTP_FROM", _env("SMTP_USER", "")),
        ),
        embedding_model=_env(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        quotation_api_url=_env("QUOTATION_API_URL", "https://apidevis.onrender.com/api/auto/packs"),
        auth_demo_mode=_env_bool("AUTH_DEMO_MODE", False),
        session_ttl_seconds=int(_env("SESSION_TTL_SECONDS", "7200")),
    )
