# financebench-rag-eval

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Status: WIP](https://img.shields.io/badge/status-WIP-orange.svg)](#status)

> Rigorous evaluation of contextual retrieval techniques on FinanceBench: comparing 5 embedders × 4 chunking strategies with bootstrapped confidence intervals on FinMTEB and FinanceBench.

## Status

🚧 **Work in progress.** This repository implements a paper-quality evaluation suite for financial document retrieval, comparing state-of-the-art embedding models and chunking strategies on the FinanceBench benchmark (Patronus AI, 2023) and FinMTEB (2025).

## Hypothesis

Domain-specific fine-tuning combined with modern chunking strategies (Anthropic's Contextual Retrieval, Late Chunking) can outperform general-purpose commercial embedders on financial document QA, even when those embedders use Matryoshka representations or 3072-dim outputs.

## Scope

This project focuses on **Layer 1 of the AI Engineer stack**: ML/embeddings/retrieval. RAG generation, agents, and cloud deployment are out of scope for this repository — they are addressed in follow-up projects.

## 📘 Financial context and metrics

New to finance or to retrieval evaluation? Before diving into the technical content, this repo includes a **plain-English glossary** that explains every acronym and concept used in the project: SEC, 10-K filings, GICS sectors, what PatronusAI built, what an `evidence` passage is, and how Recall@k / MRR / NDCG / MAP actually work — with concrete examples grounded in our dataset.

👉 **Read the full guide**: [`docs/CONTEXT.md`](./docs/CONTEXT.md)

> 💡 Recommended for: anyone reviewing the repo without a finance background, or anyone who wants to understand **what we're evaluating, against what, and why it matters** before reading the code.

## Methodology (planned)

| Component               | Plan                                                                                                     |
| ----------------------- | -------------------------------------------------------------------------------------------------------- |
| **Datasets**            | FinanceBench (150 QA pairs, public) + FinMTEB (academic finance benchmark)                               |
| **Embedders**           | OpenAI text-embedding-3-large · Voyage finance-2 · BGE-M3 · Jina v5 / Qwen3 · BGE-M3 fine-tuned (custom) |
| **Chunking strategies** | Naive fixed-size · Semantic · Anthropic Contextual Retrieval · Late Chunking                             |
| **Reranking**           | Cohere Rerank v3.5 · BGE Reranker v2                                                                     |
| **Metrics**             | Recall@k, MRR, NDCG@10, MAP — all with bootstrap confidence intervals                                    |
| **Evaluation**          | 3 retrieval modes (dense, hybrid, hybrid+rerank) on both benchmarks                                      |

## Repository structure

```
├── data/             # FinanceBench corpus (gitignored)
├── notebooks/        # Exploratory and tutorial notebooks
├── src/
│   ├── embeddings/   # Embedder wrappers
│   ├── chunking/     # Chunking strategies
│   ├── eval/         # Evaluation pipeline + bootstrap
│   └── utils/        # Shared helpers
├── results/          # Per-experiment metrics
├── docs/             # Methodology & decision logs
├── scripts/          # CLI entry points
└── tests/            # Unit tests
```

## Setup

```bash
# 1. Install Python dependencies (Python 3.12 + uv required)
uv sync

# 2. Download the FinanceBench source PDFs (~84 files, ~165 MB)
uv run python scripts/download_pdfs.py
```

> 📦 **Why is the data not in the repo?** The 165 MB of source PDFs from
> FinanceBench (CC-BY-NC-4.0) are excluded from version control for three
> reasons: (1) Git doesn't scale well with large binary files, (2) the dataset
> is third-party content, and (3) avoiding duplication of data already hosted
> upstream by [`patronus-ai/financebench`](https://github.com/patronus-ai/financebench).
> The `scripts/download_pdfs.py` script is the **canonical recipe**: it reads
> the unique `doc_name` values from the HuggingFace dataset and fetches each
> PDF directly from the upstream raw GitHub URLs into `data/raw/pdfs/`. Idempotent
> (skips files already present), parallel (8 workers), takes ~10 seconds on a
> normal connection.

## Results

> 🚧 **TBD.** Results will be published here once the evaluation pipeline lands. Each experiment will report Recall@k, MRR, NDCG@10, and MAP with bootstrap 95% confidence intervals across both benchmarks.

Per-experiment artifacts (configs, raw metrics, plots) will live under [`results/`](./results) organized by strategy.

## Reproducibility

Every result in this repository is reproducible. See [`docs/reproducibility.md`](./docs/reproducibility.md) for exact commands, seeds, and dataset revisions.

## License

MIT — see [LICENSE](./LICENSE).
