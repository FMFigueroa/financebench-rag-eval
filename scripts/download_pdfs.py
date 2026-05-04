"""Download FinanceBench source PDFs from the patronus-ai/financebench GitHub repo.

Selective download: only fetches the unique `doc_name` values present in the
open-source HuggingFace dataset (~84 files, ~170 MB), not the full repo (368
files, ~705 MB).

Idempotent: skips files already present locally.

Usage:
    uv run python scripts/download_pdfs.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---

# Source: raw URLs of the patronus-ai/financebench repo on GitHub.
# We hit raw.githubusercontent.com directly (no clone needed).
BASE_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"

# Destination: data/raw/pdfs/ inside this repo (gitignored via data/raw/*).
DEST_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "pdfs"

# Concurrency: 8 parallel downloads is a good balance between speed and being
# polite to GitHub's CDN. Higher values may trigger rate limits.
MAX_WORKERS = 8

# Timeout per request (seconds). 10-K PDFs can be a few MB, so 60s is generous.
TIMEOUT = 60


def get_unique_doc_names() -> list[str]:
    """Load the FinanceBench dataset and return the unique doc_name values."""
    ds = load_dataset("PatronusAI/financebench")
    train = ds["train"]
    # set() collapses duplicates; sorted() makes the order deterministic.
    return sorted(set(train["doc_name"]))


def download_one(doc_name: str) -> tuple[str, str, int]:
    """Download a single PDF. Returns (doc_name, status, bytes_downloaded).

    Status is one of: 'downloaded', 'skipped' (already exists), 'failed'.
    """
    dest = DEST_DIR / f"{doc_name}.pdf"

    # Idempotency: skip if already present and non-empty.
    if dest.exists() and dest.stat().st_size > 0:
        return (doc_name, "skipped", dest.stat().st_size)

    url = f"{BASE_URL}/{doc_name}.pdf"

    try:
        # stream=True avoids loading the entire PDF into memory at once;
        # we write it chunk by chunk to disk.
        with requests.get(url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return (doc_name, "downloaded", dest.stat().st_size)
    except Exception as e:
        # Cleanup any partial file on failure.
        if dest.exists():
            dest.unlink()
        print(f"\n  [FAIL] {doc_name}: {e}", file=sys.stderr)
        return (doc_name, "failed", 0)


def main() -> int:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FinanceBench dataset to extract unique doc_names...")
    doc_names = get_unique_doc_names()
    print(f"Found {len(doc_names)} unique PDFs to download.\n")

    # Download in parallel with a progress bar.
    results = {"downloaded": 0, "skipped": 0, "failed": 0}
    total_bytes = 0
    failed_names: list[str] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, name): name for name in doc_names}
        with tqdm(total=len(futures), desc="PDFs", unit="file") as pbar:
            for future in as_completed(futures):
                name, status, size = future.result()
                results[status] += 1
                total_bytes += size
                if status == "failed":
                    failed_names.append(name)
                pbar.update(1)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Downloaded:  {results['downloaded']}")
    print(f"Skipped:     {results['skipped']} (already present)")
    print(f"Failed:      {results['failed']}")
    print(f"Total size:  {total_bytes / 1_000_000:.1f} MB")
    print(f"Destination: {DEST_DIR}")

    if failed_names:
        print("\nFailed downloads:")
        for name in failed_names:
            print(f"  - {name}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
