"""Central configuration, read from environment variables (see .env.example)."""
import os

from dotenv import load_dotenv

load_dotenv()


def env(name, default=None):
    """Return an environment variable, or fail clearly if a required one is missing."""
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing environment variable: {name} (copy .env.example to .env and fill it in)")
    return value


CONFIG = {
    "qdrant": {
        "host": env("QDRANT_HOST", "localhost"),
        "port": int(env("QDRANT_PORT", "6333")),
        "collection_name": "insurance_embeddings",
    },
    "neo4j": {
        "uri": env("NEO4J_URI", "bolt://localhost:7687"),
        "auth": (env("NEO4J_USER", "neo4j"), env("NEO4J_PASSWORD")),
    },
    "postgres": {
        "host": env("POSTGRES_HOST", "localhost"),
        "port": int(env("POSTGRES_PORT", "5432")),
        "database": env("POSTGRES_DB", "data"),
        "user": env("POSTGRES_USER", "postgres"),
        "password": env("POSTGRES_PASSWORD"),
    },
    "openrouter": {
        "api_key": env("OPENROUTER_API_KEY", ""),  # optional: without it the API starts with the LLM disabled
        "model": "deepseek/deepseek-chat-v3.1:free",
        "base_url": "https://openrouter.ai/api/v1",
        "http_referer": "https://localhost:8002",
    },
    "quotation_api": {
        "base_url": env("QUOTATION_API_URL", "https://apidevis.onrender.com"),
        "endpoint": "/api/auto/packs",
    },
}
