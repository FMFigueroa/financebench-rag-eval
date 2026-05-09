"""Batch runner for embedding the FinanceBench corpus through any embedder.

Implements:
  - Pre-batch tokenization with tiktoken (no surprises at request time)
  - Dynamic batches respecting BOTH Azure OpenAI limits (max tokens AND max inputs)
  - Hybrid persistence: .npy per batch (vectors) + manifest.parquet (queryable metadata)
  - Resume on partial failure (skips batches already on disk)
  - Cost tracking + ETA in real time
  - Exponential backoff on transient errors

Why hybrid persistence (.npy + .parquet) instead of one or the other:
  - .npy = NumPy/PyTorch native, zero-overhead loads for math-heavy ops
    (Eval Pipeline, Qdrant indexing, similarity search)
  - .parquet = columnar metadata, queryable with Polars/DuckDB for the
    failure-mode analysis in Stage 4 ("which chunks of Apple 2023 failed?")
  - Each format does what it's best at. No conversion overhead.

Memoria descriptiva: https://www.notion.so/35a6ad30ca11811d96ebf4a9d7dde20b
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import tiktoken

from src.embeddings.foundry_client import (
    DEFAULT_CHUNKS_DIR,
    REPO_ROOT,
    FoundryEmbedder,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "embeddings"

# Limits for Azure OpenAI text-embedding-3-small (per portal, May 2026)
AZURE_TPM_MAX = 350_000
AZURE_RPM_MAX = 2_100
AZURE_MAX_INPUTS_PER_REQUEST = 2_048
AZURE_MAX_TOKENS_PER_INPUT = 8_192

# Pricing (USD per 1M tokens) — text-embedding-3-small
PRICE_PER_MILLION_TOKENS = 0.02

# tiktoken encoder used by OpenAI text-embedding-3-* models
ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Exact token count using the same encoder OpenAI uses internally."""
    return len(ENCODER.encode(text))


def load_chunks(chunks_dir: Path) -> list[dict]:
    """Load all chunks from every .jsonl file in chunks_dir, in stable order."""
    chunks = []
    for jsonl_path in sorted(chunks_dir.glob("*.jsonl")):
        with jsonl_path.open() as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
    return chunks


def make_batches(
    chunks: list[dict],
    max_tokens: int = 800_000,
    max_inputs: int = AZURE_MAX_INPUTS_PER_REQUEST,
) -> list[list[dict]]:
    """Group chunks into dynamic batches that respect BOTH limits.

    max_tokens default (800K) leaves a margin below the request size that would
    saturate the TPM limit (~1.05M tokens at full array size), giving the
    server room to breathe between batches.
    """
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = chunk.get("n_tokens") or count_tokens(chunk["text"])
        chunk["n_tokens"] = chunk_tokens  # cache for downstream

        # Close current batch if adding this chunk would exceed either limit
        if current_batch and (
            current_tokens + chunk_tokens > max_tokens
            or len(current_batch) >= max_inputs
        ):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(chunk)
        current_tokens += chunk_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def embed_corpus(
    embedder: FoundryEmbedder | None = None,
    embedder_name: str = "foundry_text_3_small",
    chunks_dir: Path = DEFAULT_CHUNKS_DIR,
    output_dir: Path | None = None,
    chunks_limit: int | None = None,
    pause_between_batches: float = 60.0,
    max_retries: int = 6,
) -> dict:
    """End-to-end pipeline: load → tokenize → batch → embed → persist → resume.

    Returns a summary dict with totals (chunks, tokens, cost, time, batches).
    """
    output_dir = output_dir or (DEFAULT_OUTPUT_DIR / embedder_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Loading chunks from {chunks_dir}...")
    chunks = load_chunks(chunks_dir)
    if chunks_limit:
        chunks = chunks[:chunks_limit]
        print(f"   ⚠️  Limited to first {chunks_limit} chunks (test mode)")
    print(f"   ✅ Loaded {len(chunks):,} chunks")

    print("🔢 Counting tokens...")
    total_tokens = sum(c.get("n_tokens") or count_tokens(c["text"]) for c in chunks)
    print(f"   ✅ {total_tokens:,} total tokens (~{total_tokens/1e6:.2f}M)")

    print("📊 Splitting into batches...")
    batches = make_batches(chunks)
    print(f"   ✅ {len(batches)} batches")

    estimated_cost = total_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS
    print(f"💰 Estimated cost: ${estimated_cost:.4f}")
    print(f"⏱️  Pause between batches: {pause_between_batches:.0f}s")

    # Resume detection
    existing_batches = sorted(output_dir.glob("batch_*.npy"))
    completed = len(existing_batches)
    if completed > 0:
        print(f"🔍 Resume: {completed}/{len(batches)} batches already on disk, will skip")

    # Reload manifest if exists (preserves metadata of completed batches)
    manifest_path = output_dir / "manifest.parquet"
    manifest_rows: list[dict] = []
    if manifest_path.exists():
        manifest_rows = pl.read_parquet(manifest_path).to_dicts()
        print(f"   ✅ Loaded existing manifest ({len(manifest_rows):,} rows)")

    embedder = embedder or FoundryEmbedder()
    start_time = time.time()
    tokens_processed = sum(
        c["n_tokens"] for batch in batches[:completed] for c in batch
    )

    print()
    for batch_id, batch in enumerate(batches):
        if batch_id < completed:
            continue

        batch_tokens = sum(c["n_tokens"] for c in batch)
        batch_start = time.time()

        print(
            f"▶️  Batch {batch_id+1:02d}/{len(batches)} — "
            f"{len(batch):,} chunks, {batch_tokens:,} tokens... ",
            end="",
            flush=True,
        )

        # Embed with exponential backoff on transient errors
        retries = 0
        while True:
            try:
                texts = [c["text"] for c in batch]
                vectors = embedder.embed(texts)  # np.ndarray shape (n, dim)
                break
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    raise
                wait = 30 * (2 ** (retries - 1))
                print(
                    f"\n   ⚠️  {type(e).__name__}: retrying in {wait}s "
                    f"(attempt {retries+1}/{max_retries+1})..."
                )
                time.sleep(wait)

        # Save vectors
        batch_path = output_dir / f"batch_{batch_id:02d}.npy"
        np.save(batch_path, vectors)

        # Append to manifest
        for pos, chunk in enumerate(batch):
            manifest_rows.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_name": chunk["doc_name"],
                    "page_num": chunk["page_num"],
                    "n_tokens": chunk["n_tokens"],
                    "batch_id": batch_id,
                    "position_in_batch": pos,
                }
            )

        # Persist manifest after each batch (atomic resume)
        pl.DataFrame(manifest_rows).write_parquet(manifest_path)

        # Stats
        batch_elapsed = time.time() - batch_start
        tokens_processed += batch_tokens
        elapsed = time.time() - start_time
        cost_so_far = tokens_processed / 1_000_000 * PRICE_PER_MILLION_TOKENS
        progress_pct = (batch_id + 1) / len(batches) * 100
        batches_done_this_run = batch_id + 1 - completed
        eta_seconds = (
            elapsed / batches_done_this_run * (len(batches) - batch_id - 1)
            if batches_done_this_run > 0
            else 0
        )

        print(
            f"✅ {_format_duration(batch_elapsed)} — "
            f"{progress_pct:.0f}% — "
            f"${cost_so_far:.4f} — "
            f"ETA {_format_duration(eta_seconds)}"
        )

        # Pause between batches (rate-limit safety)
        if batch_id + 1 < len(batches):
            time.sleep(pause_between_batches)

    total_elapsed = time.time() - start_time
    final_cost = tokens_processed / 1_000_000 * PRICE_PER_MILLION_TOKENS

    print()
    print("🎯 Done!")
    print(f"   Total time:    {_format_duration(total_elapsed)}")
    print(f"   Total cost:    ${final_cost:.4f}")
    print(f"   Total tokens:  {tokens_processed:,}")
    print(f"   Output:        {output_dir}")
    print(f"   Vectors:       batch_00.npy ... batch_{len(batches)-1:02d}.npy")
    print(f"   Manifest:      manifest.parquet ({len(manifest_rows):,} rows)")

    return {
        "total_chunks": len(chunks),
        "total_tokens": tokens_processed,
        "total_cost_usd": final_cost,
        "total_time_seconds": total_elapsed,
        "batches": len(batches),
        "output_dir": str(output_dir),
    }
