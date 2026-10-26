"""Step 3 – turn the OCR output into one clean text per document (table `cleaned_docs`).

    python -m ingestion.clean_documents                      # from the OCR tables
    python -m ingestion.clean_documents --export data/processed/cleaned_docs.jsonl
    python -m ingestion.clean_documents --import data/processed/cleaned_docs.jsonl

The OCR step (data_extraction/) fills `documents` and `pages_content` (if a PDF was
processed several times, its latest run is used). This step joins
the pages, removes headers and footers repeated on most pages, normalizes spacing,
and records the insurance branch, taken from the PDF's folder name
(data/raw/pdfs/<branche>/file.pdf).

`--export` writes the result to a JSONL file that can be committed, so others can
rebuild the indexes without paying for OCR; `--import` loads it back.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from .common import connect

_FENCE = re.compile(r"^```\w*\s*$", re.MULTILINE)
_SPACES = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def branch_of(file_path: str | None) -> str:
    """data/raw/pdfs/auto/cg.pdf -> 'auto'; a PDF directly in pdfs/ -> 'general'."""
    if not file_path:
        return "general"
    parent = Path(file_path).parent.name
    return parent if parent and parent not in {"pdfs", "raw", "data", "."} else "general"


def clean_page(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = _FENCE.sub("", text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def remove_repeated_lines(pages: list[str], min_share: float = 0.6) -> list[str]:
    """Drop short lines (headers, footers, page numbers) found on most pages."""
    if len(pages) < 3:
        return pages
    counts = Counter(
        line for page in pages for line in set(page.splitlines()) if 0 < len(line) <= 80
    )
    repeated = {line for line, n in counts.items() if n / len(pages) >= min_share}
    page_number = re.compile(r"^(page\s*)?\d{1,3}(\s*/\s*\d{1,3})?$", re.IGNORECASE)
    return [
        "\n".join(line for line in page.splitlines() if line not in repeated and not page_number.match(line))
        for page in pages
    ]


def clean_document(pages: list[str]) -> str:
    cleaned = remove_repeated_lines([clean_page(p) for p in pages])
    return _BLANK_LINES.sub("\n\n", "\n\n".join(p for p in cleaned if p.strip())).strip()


CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS cleaned_docs (
    filename text PRIMARY KEY,
    branche text NOT NULL,
    pages integer NOT NULL,
    text_cleaned text NOT NULL
)
"""


async def from_ocr_tables(conn: Any) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT d.filename, d.file_path, p.page_number, p.page_content
        FROM documents d
        JOIN (SELECT filename, max(id) AS id FROM documents GROUP BY filename) latest ON latest.id = d.id
        JOIN pages_content p ON p.document_id = d.id
        ORDER BY d.filename, p.page_number
        """
    )
    pages: dict[str, list[str]] = {}
    paths: dict[str, str] = {}
    for row in rows:
        pages.setdefault(row["filename"], []).append(row["page_content"] or "")
        paths[row["filename"]] = row["file_path"]
    return [
        {"filename": name, "branche": branch_of(paths[name]), "pages": len(texts), "text_cleaned": clean_document(texts)}
        for name, texts in pages.items()
    ]


async def save(conn: Any, documents: list[dict[str, Any]]) -> None:
    await conn.execute(CREATE_TABLE)
    await conn.execute("TRUNCATE cleaned_docs")
    await conn.executemany(
        "INSERT INTO cleaned_docs (filename, branche, pages, text_cleaned) VALUES ($1, $2, $3, $4)",
        [(d["filename"], d["branche"], d["pages"], d["text_cleaned"]) for d in documents if d["text_cleaned"]],
    )


async def main(export: Path | None, import_file: Path | None) -> None:
    conn = await connect()
    try:
        if import_file:
            documents = [json.loads(line) for line in import_file.read_text(encoding="utf-8").splitlines() if line]
        else:
            documents = await from_ocr_tables(conn)
        await save(conn, documents)
        print(f"cleaned_docs: {len(documents)} documents, {sum(len(d['text_cleaned']) for d in documents):,} characters")
        if export:
            export.parent.mkdir(parents=True, exist_ok=True)
            with export.open("w", encoding="utf-8") as handle:
                for document in documents:
                    handle.write(json.dumps(document, ensure_ascii=False) + "\n")
            print(f"exported to {export}")
    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, help="also write the cleaned documents to this JSONL file")
    parser.add_argument("--import", dest="import_file", type=Path, help="load cleaned documents from a JSONL file instead of the OCR tables")
    args = parser.parse_args()
    asyncio.run(main(args.export, args.import_file))
