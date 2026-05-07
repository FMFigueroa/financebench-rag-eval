"""Parse FinanceBench source PDFs to JSONL using pdfplumber.

For each PDF in ``data/raw/pdfs/``, emit one JSONL file in ``data/processed/parsed/``
where each line represents a single page with the canonical schema:

    {
        "doc_name": "3M_2018_10K",
        "page_num": 56,
        "text": "Item 8. Financial Statements...",
        "tables": [[["Net sales", "", "$", "32,765", ...], ...]]
    }

Why this design (sub-block 4 decisions, see Notion descriptive memory):

- **Library**: pdfplumber wins for 10-K filings (heuristic table extraction works
  on bordered tables, exposes raw coordinates, zero setup friction).
- **Table settings**: ``vertical_strategy="text"`` + ``horizontal_strategy="text"``
  is the project default. Default settings fragment the Income Statement into
  ~12 sub-tables because of horizontal separator lines; text-based strategy
  yields a single clean matrix.
- **Granularity**: 1 line = 1 page. This aligns with FinanceBench's
  ``evidence_page_num`` (singular) — no need to reunify multi-page tables.
- **Execution**: serial. The parser runs 1-2 times in the whole project (output
  cached as JSONL); parallelism would add complexity without a real win.
- **Idempotency**: skips PDFs whose JSONL already exists and is non-empty.

Usage:
    uv run python scripts/parse_pdfs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pdfplumber
from tqdm import tqdm

# --- Configuration ---

PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "pdfs"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "parsed"

# Project-default table_settings — text-based strategy avoids fragmentation
# caused by horizontal separator lines in financial statements.
TABLE_SETTINGS = {"vertical_strategy": "text", "horizontal_strategy": "text"}


def parse_page_to_record(page: pdfplumber.page.Page, doc_name: str) -> dict:
    """Convert a single pdfplumber Page to the canonical JSONL record.

    The minimal schema is intentionally small. Extra fields (bbox, char_count,
    etc.) get added only if downstream metrics justify them.
    """
    return {
        "doc_name": doc_name,
        "page_num": page.page_number,
        "text": page.extract_text() or "",
        "tables": page.extract_tables(TABLE_SETTINGS),
    }


def parse_one(pdf_path: Path) -> tuple[str, str, int, int]:
    """Parse a single PDF into a JSONL file.

    Returns (doc_name, status, n_pages, jsonl_bytes).
    Status ∈ {'parsed', 'skipped' (already present), 'failed'}.
    """
    doc_name = pdf_path.stem
    out_path = OUTPUT_DIR / f"{doc_name}.jsonl"

    # Idempotency: skip if JSONL already exists and is non-empty.
    if out_path.exists() and out_path.stat().st_size > 0:
        return (doc_name, "skipped", 0, out_path.stat().st_size)

    try:
        with pdfplumber.open(pdf_path) as pdf, open(out_path, "w", encoding="utf-8") as f:
            n_pages = 0
            for page in pdf.pages:
                record = parse_page_to_record(page, doc_name)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_pages += 1
        return (doc_name, "parsed", n_pages, out_path.stat().st_size)
    except Exception as e:
        # Cleanup any partial JSONL on failure.
        if out_path.exists():
            out_path.unlink()
        print(f"\n  [FAIL] {doc_name}: {e}", file=sys.stderr)
        return (doc_name, "failed", 0, 0)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {PDF_DIR}. Run scripts/download_pdfs.py first.", file=sys.stderr)
        return 1

    print(f"Found {len(pdf_paths)} PDFs to parse in {PDF_DIR}.\n")

    results = {"parsed": 0, "skipped": 0, "failed": 0}
    total_pages = 0
    total_bytes = 0
    failed_names: list[str] = []

    # Serial loop with progress bar (decision #5: parser runs 1-2 times total).
    for pdf_path in tqdm(pdf_paths, desc="Parsing", unit="pdf"):
        name, status, n_pages, size = parse_one(pdf_path)
        results[status] += 1
        total_pages += n_pages
        total_bytes += size
        if status == "failed":
            failed_names.append(name)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("PARSE SUMMARY")
    print("=" * 60)
    print(f"Parsed:      {results['parsed']}")
    print(f"Skipped:     {results['skipped']} (JSONL already present)")
    print(f"Failed:      {results['failed']}")
    print(f"Total pages: {total_pages:,}")
    print(f"Total size:  {total_bytes / 1_000_000:.1f} MB")
    print(f"Destination: {OUTPUT_DIR}")

    if failed_names:
        print("\nFailed PDFs:")
        for name in failed_names:
            print(f"  - {name}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
