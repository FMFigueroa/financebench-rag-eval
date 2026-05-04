# Reproducibility

This document specifies the exact procedure to reproduce every result reported
in this repository. The goal is bit-level reproducibility for offline metrics
and statistical reproducibility (within bootstrap confidence intervals) for
results that depend on remote APIs.

## Environment

- **Python**: 3.12 (pinned via `.python-version`)
- **Package manager**: [uv](https://docs.astral.sh/uv/)
- **Lockfile**: `uv.lock` (committed) — defines exact versions of every transitive dependency

```bash
uv sync          # installs from uv.lock
uv run pytest    # sanity check
```

## Seeds

All non-deterministic operations use a fixed seed. Default: `SEED = 42`.

| Operation                  | Seeded via                                |
| -------------------------- | ----------------------------------------- |
| Bootstrap resampling       | `numpy.random.default_rng(SEED)`          |
| Train/eval splits          | `random.Random(SEED)`                     |
| Fine-tuning data shuffling | `torch.manual_seed(SEED)` + cuDNN deterministic |
| Vector store ingestion order | Sorted by document ID, not insertion order |

## Data sources

| Dataset       | Source                                                   | Version          |
| ------------- | -------------------------------------------------------- | ---------------- |
| FinanceBench  | https://github.com/patronus-ai/financebench              | TBD (commit SHA) |
| FinMTEB       | https://huggingface.co/datasets/FinMTEB/FinMTEB          | TBD (revision)   |

Dataset download scripts live in `scripts/` and pin exact revisions.

### Why source data is not committed

This repository **does not commit source data**. The 84 FinanceBench PDFs
(~165 MB) and any future corpora live under `data/raw/` and `data/processed/`,
both gitignored. Three reasons drive this convention:

1. **Repository hygiene**: Git is not designed for large binary blobs. Storing
   PDFs in the history would inflate clone size and make `git log` operations
   slow as the project grows.
2. **Licensing**: FinanceBench is distributed under `CC-BY-NC-4.0` by Patronus
   AI. Re-distributing the PDFs in a separate public repo enters a legally
   ambiguous zone. Pointing users at the upstream source is cleaner.
3. **Single source of truth**: the upstream Patronus repo is the canonical
   location. Duplicating data here means future updates require manual sync.

### How users obtain the data

The download scripts in `scripts/` are the **recipe**. They are idempotent
(skip files already present) and parallelized (8 workers). After a fresh clone:

```bash
uv sync
uv run python scripts/download_pdfs.py     # FinanceBench PDFs (~84 files, ~165 MB, ~10 s)
# Future: uv run python scripts/download_finmteb.py
```

The HuggingFace `datasets` cache (`~/.cache/huggingface/datasets/`) handles the
QA pairs and metadata automatically — no extra script needed for those.

### Corpus integrity

Each download script logs:

- Number of files expected vs. downloaded
- Total bytes
- Any failures (with the upstream URL for manual retry)

If integrity matters for a specific experiment (e.g., a reviewer checks our
results), pin the upstream commit SHA in the script's `BASE_URL` instead of
relying on `main`. We do this once a `v1.0.0` release is cut.

## Running experiments

> **TODO**: fill in once the eval pipeline is implemented.

Each experiment is a single CLI invocation. Results land in
`results/<strategy>/<embedder>/<timestamp>.json` with the full config
embedded for traceability.

```bash
# Example (placeholder):
uv run python -m src.eval.run \
    --embedder voyage-finance-2 \
    --chunking contextual \
    --benchmark financebench \
    --seed 42
```

## Remote API determinism

Closed-source embedders (OpenAI, Voyage, Cohere) are not bit-level
deterministic across calls. We mitigate this by:

1. Caching all embedding outputs to `data/processed/embeddings_cache/`
   keyed by `sha256(text + model + version)`
2. Pinning the model version string in every config (e.g.
   `text-embedding-3-large@2024-01-25`, not just `text-embedding-3-large`)
3. Reporting bootstrap CIs over the metric, not point estimates

## Hardware

Experiments were run on:

- **Local**: Apple Silicon (MPS) for inference-only embedders
- **GPU**: TBD (rented A100/H100 for fine-tuning runs)

Hardware does not affect closed-API results. For local embedders we report
the device used in each result file.

## What is NOT guaranteed reproducible

- **Wall-clock latency** (depends on network, API load, hardware)
- **Exact embeddings from closed APIs** if the provider silently bumps the
  model. We detect this via the cache key check on every run and fail loudly.
