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
