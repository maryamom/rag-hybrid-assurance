"""Helpers shared by the ingestion scripts."""
from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

import asyncpg

from src.config import get_settings

if TYPE_CHECKING:
    from .schema import Source


def slugify(text: str) -> str:
    """'Données assurance S1.2' -> 'donnees_assurance_s1_2' (safe as a SQL identifier)."""
    ascii_text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")
    if not slug:
        slug = "col"
    return slug if not slug[0].isdigit() else f"t_{slug}"


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


async def connect() -> asyncpg.Connection:
    return await asyncpg.connect(get_settings().postgres.dsn)


async def table_columns(conn: asyncpg.Connection) -> dict[str, list[str]]:
    """{table: [columns]} for every table and view of the public schema."""
    rows = await conn.fetch(
        """
        SELECT table_name, column_name FROM information_schema.columns
        WHERE table_schema = 'public' ORDER BY table_name, ordinal_position
        """
    )
    tables: dict[str, list[str]] = {}
    for row in rows:
        tables.setdefault(row["table_name"], []).append(row["column_name"])
    return tables


class Resolved:
    """A logical source matched to a real table: `columns` maps logical -> real names."""

    def __init__(self, table: str, columns: dict[str, str], missing: list[str]) -> None:
        self.table, self.columns, self.missing = table, columns, missing

    @property
    def usable(self) -> bool:
        return not self.missing

    def col(self, logical: str) -> str | None:
        real = self.columns.get(logical)
        return quote_ident(real) if real else None


def resolve(source: "Source", tables: dict[str, list[str]]) -> Resolved | None:
    """Find the table and columns of a logical source, or None if no table matches."""
    for candidate in source.tables:
        matches = sorted(t for t in tables if t == candidate or t.endswith(candidate))
        if matches:
            table = matches[0]
            available = set(tables[table])
            columns = {
                logical: next(name for name in names if name in available)
                for logical, names in source.columns.items()
                if any(name in available for name in names)
            }
            missing = [c for c in source.required if c not in columns]
            return Resolved(table, columns, missing)
    return None
