"""Step 5 – expose the client records under the names the API reads: clients, devis, contrats.

    python -m ingestion.build_client_views

These are SQL views over the loaded hackathon tables, with standard column names
(id, nom, prenom, telephone, email / client_id, date_devis / client_id, date_effet).
Identifiers are cast to text so that clients and their records always join.
"""
from __future__ import annotations

import asyncio
from typing import Any

from .common import Resolved, connect, quote_ident, resolve, table_columns
from .schema import CLIENT_CONTRACTS, CLIENT_QUOTES, CLIENTS


def clients_view_sql(found: Resolved) -> str:
    return (
        "CREATE VIEW clients AS SELECT "
        f"{found.col('id')}::text AS id, {found.col('nom')}::text AS nom, "
        f"{found.col('prenom')}::text AS prenom, {found.col('telephone')}::text AS telephone, "
        f"{found.col('email')}::text AS email "
        f"FROM {quote_ident(found.table)}"
    )


def records_view_sql(view: str, date_alias: str, found: Resolved, all_columns: list[str]) -> str:
    """client_id + a standard date column first, then every other column of the table."""
    used = {found.columns.get("client_id"), found.columns.get("date")}
    reserved = {"client_id", date_alias}
    others = [quote_ident(c) for c in all_columns if c not in used and c not in reserved]
    date_expr = found.col("date") or "NULL"
    select = [f"{found.col('client_id')}::text AS client_id", f"{date_expr} AS {date_alias}", *others]
    return f"CREATE VIEW {view} AS SELECT {', '.join(select)} FROM {quote_ident(found.table)}"


async def build_views(conn: Any) -> list[str]:
    tables = await table_columns(conn)
    created = []
    for view in ("contrats", "devis", "clients"):
        await conn.execute(f"DROP VIEW IF EXISTS {view}")

    clients = resolve(CLIENTS, tables)
    if clients is None or not clients.usable:
        print("clients: skipped (run python -m ingestion.check_data)")
        return created
    await conn.execute(clients_view_sql(clients))
    created.append("clients")

    for view, date_alias, source in (("devis", "date_devis", CLIENT_QUOTES), ("contrats", "date_effet", CLIENT_CONTRACTS)):
        found = resolve(source, tables)
        if found is None or not found.usable:
            print(f"{view}: skipped (run python -m ingestion.check_data)")
            continue
        await conn.execute(records_view_sql(view, date_alias, found, tables[found.table]))
        created.append(view)
    return created


async def main() -> None:
    conn = await connect()
    try:
        created = await build_views(conn)
    finally:
        await conn.close()
    print(f"Views created: {', '.join(created) or 'none'}")


if __name__ == "__main__":
    asyncio.run(main())
