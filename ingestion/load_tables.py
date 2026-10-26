"""Step 1 – load every Excel/CSV file of the hackathon into PostgreSQL, one table per sheet.

    python -m ingestion.load_tables data/raw/tables

Table name = file name + sheet name, in snake_case (e.g. `description_garanties_sheet1`).
Column names are converted the same way. Tables are dropped and recreated, so the
script can be run again safely.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import pandas as pd

from .common import connect, quote_ident, slugify


def _sql_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "bigint"
    if pd.api.types.is_float_dtype(series):
        return "double precision"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamp"
    return "text"


def _python_value(value: Any, sql_type: str) -> Any:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if sql_type == "bigint":
        return int(value)
    if sql_type == "double precision":
        return float(value)
    if sql_type == "boolean":
        return bool(value)
    if sql_type == "timestamp":
        return pd.Timestamp(value).to_pydatetime()
    return str(value).strip()


def read_tables(path: Path) -> dict[str, pd.DataFrame]:
    """Return {table_name: dataframe} for one Excel or CSV file."""
    if path.suffix.lower() == ".csv":
        frames = {"": pd.read_csv(path)}
    else:
        frames = pd.read_excel(path, sheet_name=None)
    tables = {}
    for sheet, frame in frames.items():
        frame = frame.dropna(how="all").dropna(axis=1, how="all")
        if frame.empty:
            continue
        columns, seen = [], set()
        for column in frame.columns:
            name = slugify(column)
            while name in seen:
                name += "_bis"
            seen.add(name)
            columns.append(name)
        frame.columns = columns
        table = slugify(f"{path.stem}_{sheet}" if sheet else path.stem)
        tables[table] = frame
    return tables


async def load_frame(conn: Any, table: str, frame: pd.DataFrame) -> int:
    types = {column: _sql_type(frame[column]) for column in frame.columns}
    columns_sql = ", ".join(f"{quote_ident(c)} {t}" for c, t in types.items())
    await conn.execute(f"DROP TABLE IF EXISTS {quote_ident(table)} CASCADE")
    await conn.execute(f"CREATE TABLE {quote_ident(table)} ({columns_sql})")
    records = [
        tuple(_python_value(row[c], types[c]) for c in frame.columns) for _, row in frame.iterrows()
    ]
    await conn.copy_records_to_table(table, records=records, columns=list(frame.columns))
    return len(records)


async def main(folder: Path) -> None:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".xlsx", ".xls", ".csv"})
    if not files:
        raise SystemExit(f"No .xlsx/.csv file in {folder}")
    conn = await connect()
    try:
        for path in files:
            for table, frame in read_tables(path).items():
                count = await load_frame(conn, table, frame)
                print(f"{table:<60} {count:>7} rows  ({path.name})")
    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder with the hackathon's .xlsx/.csv files")
    asyncio.run(main(parser.parse_args().folder))
