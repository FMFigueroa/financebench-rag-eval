"""Build the indexable corpus from parsed JSONL files.

Reads:  data/processed/parsed/<doc_name>.jsonl   (1 line per page)
Writes: data/processed/chunks/<doc_name>.jsonl   (1 line per chunk)

Per-chunk schema (sub-block 5 decision #3):
    {
        "chunk_id":   "<doc_name>_<idx_within_doc:04d>",
        "doc_name":   "3M_2018_10K",
        "page_num":   56,                # 1-indexed; matches dataset evidence_page_num
        "chunk_type": "text" | "table",
        "text":       "...",
        "n_tokens":   <int>              # measured with tiktoken cl100k_base
    }

Sub-block 5 decisions materialized here:

1. Tokenizer: tiktoken cl100k_base (OpenAI) as single reference.
2. Pre-filter false-positive tables: keep only tables with ≥3 rows × ≥2 cols.
3. Chunk schema: 6 fields above (chunk_id, doc_name, page_num, chunk_type, text, n_tokens).
4. Header-repetition: header_row_idx=0; whole-table-as-1-chunk if ≤512 tokens,
   else header-repetition with header preserved in each sub-chunk.
5. Overlap: 50 tokens for text chunks (token-based via tiktoken), NO overlap
   between table chunks (header-repetition serves the same purpose).

Idempotent: skips PDFs whose chunk JSONL already exists and is non-empty.

Usage:
    uv run python scripts/build_corpus.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import tiktoken
from tqdm import tqdm

# --- Configuration ---

PARSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "parsed"
CHUNKS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "chunks"

CHUNK_SIZE = 512        # max tokens per chunk
OVERLAP = 50            # tokens of overlap between consecutive TEXT chunks
MIN_TABLE_ROWS = 3      # filter false-positive tables: header + ≥2 data rows
MIN_TABLE_COLS = 2      # filter false-positive tables: anything narrower is a list
NUMERIC_COL_RATIO = 0.3 # filter false-positive tables: ≥1 column where ≥30% cells are numeric

ENCODING_NAME = "cl100k_base"  # OpenAI's tokenizer; project-default reference

# Numeric-cell pattern: digits, comma/dot/dollar/percent/parens/dash. Captures
# things like "32,765", "$1,577", "(123.4)", "5.6%", "-1.2". Anything mixed
# with letters fails — that filters out fragments like "Notes" or "STAT".
_NUMERIC_RE = re.compile(r"^[\d,.()$%\s+-]+$")

# Lazy global encoder (loaded once, reused for the whole batch).
_encoder = tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Token count with the project-reference encoder (cl100k_base)."""
    return len(_encoder.encode(text))


def _is_numeric_cell(cell) -> bool:
    if cell is None:
        return False
    s = str(cell).strip()
    if not s:
        return False
    return bool(_NUMERIC_RE.match(s))


def _has_numeric_column(table: list[list[str]]) -> bool:
    """At least one column where ≥NUMERIC_COL_RATIO cells are numeric.

    This is the discriminator that rules out cover pages, headers, and
    text-only block layouts that pdfplumber detects as 'tables'. Real
    financial tables (Income Statement, Balance Sheet, Cash Flows, Notes
    detail) all have at least one column of numbers; cover-page artifacts
    do not.
    """
    if not table:
        return False
    n_rows = len(table)
    n_cols = max(len(r) for r in table)
    for col_idx in range(n_cols):
        col_cells = [row[col_idx] if col_idx < len(row) else None for row in table]
        n_numeric = sum(1 for c in col_cells if _is_numeric_cell(c))
        if n_numeric / n_rows >= NUMERIC_COL_RATIO:
            return True
    return False


def is_real_table(table: list[list[str]]) -> bool:
    """Filter heuristic for false-positive tables (decision #2, hardened).

    Three checks must all pass:
      1. Shape: ≥3 rows × ≥2 cols (excludes 1-row 'tables' and N×1 lists).
      2. Numeric-column presence: at least one column with ≥30% numeric
         cells (excludes cover-page artifacts and text-block false positives).

    The numeric-column rule was added after the initial filter let through
    cover pages parsed as 70×13 'tables' from page-1 form text. Density
    alone (% non-empty cells) does not discriminate — real and false-positive
    tables both sit around 35%. Numeric-column presence does.
    """
    if not table or len(table) < MIN_TABLE_ROWS:
        return False
    if len(table[0]) < MIN_TABLE_COLS:
        return False
    if not _has_numeric_column(table):
        return False
    return True


def table_to_markdown(table: list[list[str]]) -> str:
    """Render a 2D table as a pipe-separated string.

    Keeps the row ↔ column association readable for the embedder. Empty cells
    become empty strings (preserves spacing) — None gets coerced to ''.
    """
    return "\n".join(
        " | ".join(str(cell) if cell is not None else "" for cell in row)
        for row in table
    )


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping token-based chunks (decision #5, text branch).

    Uses tiktoken cl100k_base as the reference tokenizer. Window of CHUNK_SIZE
    tokens, sliding stride of (CHUNK_SIZE - OVERLAP). Last chunk may be shorter.
    """
    tokens = _encoder.encode(text)
    if len(tokens) <= CHUNK_SIZE:
        return [text]

    stride = CHUNK_SIZE - OVERLAP
    chunks: list[str] = []
    i = 0
    while i < len(tokens):
        window = tokens[i : i + CHUNK_SIZE]
        chunks.append(_encoder.decode(window))
        if i + CHUNK_SIZE >= len(tokens):
            break
        i += stride
    return chunks


def chunk_table(table: list[list[str]]) -> list[str]:
    """Apply the table-chunking strategy (decision #4 + #5, table branch).

    If the rendered table fits in ≤CHUNK_SIZE tokens → emit as 1 chunk.
    Otherwise: header-repetition (header_row_idx=0) — split rows greedily so
    each chunk stays within CHUNK_SIZE tokens, with the header row prepended
    to every sub-chunk. No overlap between table sub-chunks (header serves
    the anchoring role).
    """
    full_text = table_to_markdown(table)
    if count_tokens(full_text) <= CHUNK_SIZE:
        return [full_text]

    # Header-repetition path
    header = table[0]
    data_rows = table[1:]
    header_text = table_to_markdown([header])
    header_tokens = count_tokens(header_text)
    # The header consumes some budget; the rest is for data rows.
    budget_for_data = CHUNK_SIZE - header_tokens

    chunks: list[str] = []
    current_rows: list[list[str]] = []
    current_tokens = 0

    for row in data_rows:
        row_text = table_to_markdown([row])
        row_tokens = count_tokens(row_text) + 1  # +1 for the joining newline
        # If adding this row would overflow, emit the current chunk and reset.
        if current_rows and current_tokens + row_tokens > budget_for_data:
            chunks.append(table_to_markdown([header] + current_rows))
            current_rows = [row]
            current_tokens = row_tokens
        else:
            current_rows.append(row)
            current_tokens += row_tokens

    if current_rows:
        chunks.append(table_to_markdown([header] + current_rows))

    return chunks


def chunks_for_page(page_record: dict, doc_name: str) -> list[dict]:
    """Produce all chunks (text + table) for a single parsed page.

    chunk_id is left blank here — assigned at the doc level by build_one() so
    that IDs are stable per document and resumable across runs.
    """
    out: list[dict] = []
    page_num = page_record["page_num"]

    # --- Text branch ---
    text = (page_record.get("text") or "").strip()
    if text:
        for piece in chunk_text(text):
            out.append(
                {
                    "doc_name": doc_name,
                    "page_num": page_num,
                    "chunk_type": "text",
                    "text": piece,
                    "n_tokens": count_tokens(piece),
                }
            )

    # --- Table branch ---
    for table in page_record.get("tables") or []:
        if not is_real_table(table):
            continue
        for piece in chunk_table(table):
            out.append(
                {
                    "doc_name": doc_name,
                    "page_num": page_num,
                    "chunk_type": "table",
                    "text": piece,
                    "n_tokens": count_tokens(piece),
                }
            )

    return out


def build_one(parsed_path: Path) -> tuple[str, str, int, int]:
    """Build chunks for a single parsed JSONL. Returns (doc_name, status, n_chunks, bytes)."""
    doc_name = parsed_path.stem
    out_path = CHUNKS_DIR / f"{doc_name}.jsonl"

    # Idempotency: skip if already present and non-empty.
    if out_path.exists() and out_path.stat().st_size > 0:
        return (doc_name, "skipped", 0, out_path.stat().st_size)

    try:
        n_chunks = 0
        with open(parsed_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                page = json.loads(line)
                for chunk in chunks_for_page(page, doc_name):
                    chunk["chunk_id"] = f"{doc_name}_{n_chunks:04d}"
                    fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    n_chunks += 1
        return (doc_name, "built", n_chunks, out_path.stat().st_size)
    except Exception as e:
        if out_path.exists():
            out_path.unlink()
        print(f"\n  [FAIL] {doc_name}: {e}", file=sys.stderr)
        return (doc_name, "failed", 0, 0)


def main() -> int:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    parsed_paths = sorted(PARSED_DIR.glob("*.jsonl"))
    if not parsed_paths:
        print(f"No parsed JSONL found in {PARSED_DIR}. Run scripts/parse_pdfs.py first.", file=sys.stderr)
        return 1

    print(f"Found {len(parsed_paths)} parsed JSONL files in {PARSED_DIR}.\n")

    results = {"built": 0, "skipped": 0, "failed": 0}
    total_chunks = 0
    total_bytes = 0
    failed_names: list[str] = []

    for parsed_path in tqdm(parsed_paths, desc="Chunking", unit="doc"):
        name, status, n_chunks, size = build_one(parsed_path)
        results[status] += 1
        total_chunks += n_chunks
        total_bytes += size
        if status == "failed":
            failed_names.append(name)

    print("\n" + "=" * 60)
    print("CORPUS BUILD SUMMARY")
    print("=" * 60)
    print(f"Built:          {results['built']}")
    print(f"Skipped:        {results['skipped']} (chunks JSONL already present)")
    print(f"Failed:         {results['failed']}")
    print(f"Total chunks:   {total_chunks:,}")
    print(f"Total size:     {total_bytes / 1_000_000:.1f} MB")
    print(f"Destination:    {CHUNKS_DIR}")

    if failed_names:
        print("\nFailed docs:")
        for name in failed_names:
            print(f"  - {name}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
