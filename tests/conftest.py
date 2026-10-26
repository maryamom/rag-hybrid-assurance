import os

import asyncpg
import pytest

from src.config import get_settings

TEST_DB = os.getenv("TEST_POSTGRES_DB", "bhbot_test")  # never the application's database


@pytest.fixture
async def pg_env(monkeypatch):
    """Point the settings at a throw-away test database (created if needed)."""
    if "TEST_POSTGRES_PORT" not in os.environ:
        pytest.skip("no test PostgreSQL")
    host = os.getenv("TEST_POSTGRES_HOST", "127.0.0.1")
    port = os.environ["TEST_POSTGRES_PORT"]
    password = os.getenv("POSTGRES_PASSWORD", "testpw")
    admin = await asyncpg.connect(host=host, port=port, user="postgres", password=password, database="postgres")
    try:
        if not await admin.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", TEST_DB):
            await admin.execute(f'CREATE DATABASE "{TEST_DB}"')
    finally:
        await admin.close()
    for key, value in {"POSTGRES_HOST": host, "POSTGRES_PORT": port, "POSTGRES_PASSWORD": password,
                       "POSTGRES_DB": TEST_DB, "NEO4J_PASSWORD": os.getenv("TEST_NEO4J_PASSWORD", "unused")}.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
