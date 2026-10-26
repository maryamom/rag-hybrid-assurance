"""Step 2 – check that the loaded tables contain what the pipeline expects.

    python -m ingestion.check_data

For each piece of information (clients, contracts, guarantees, FAQ, ...) it shows the
table and columns found, or what is missing. If a name differs, add it to the
candidate lists in ingestion/schema.py and run the check again.
"""
from __future__ import annotations

import asyncio

from .common import connect, resolve, table_columns
from .schema import ALL_SOURCES


def report(tables: dict[str, list[str]]) -> tuple[list[str], bool]:
    lines, all_ok = [], True
    for name, source in ALL_SOURCES.items():
        found = resolve(source, tables)
        if found is None:
            all_ok = False
            lines.append(f"MISSING  {name:<24} no table ending with {' / '.join(source.tables)}")
            lines.append(f"         ({source.description})")
            continue
        mapped = ", ".join(f"{logical}={real}" for logical, real in found.columns.items())
        if found.usable:
            lines.append(f"OK       {name:<24} {found.table}: {mapped}")
        else:
            all_ok = False
            lines.append(f"PARTIAL  {name:<24} {found.table}: missing column(s) {', '.join(found.missing)}")
            for logical in found.missing:
                lines.append(f"         {logical}: expected one of {', '.join(source.columns[logical])}")
            lines.append(f"         columns in the table: {', '.join(tables[found.table])}")
    return lines, all_ok


async def main() -> None:
    conn = await connect()
    try:
        tables = await table_columns(conn)
    finally:
        await conn.close()
    lines, all_ok = report(tables)
    print("\n".join(lines))
    print("\nEverything the pipeline needs was found." if all_ok else
          "\nSome information is missing: the steps that need it will be skipped.")


if __name__ == "__main__":
    asyncio.run(main())
