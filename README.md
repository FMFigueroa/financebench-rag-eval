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
# Requires Python 3.12 and uv
uv sync
```

## Results

> 🚧 **TBD.** Results will be published here once the evaluation pipeline lands. Each experiment will report Recall@k, MRR, NDCG@10, and MAP with bootstrap 95% confidence intervals across both benchmarks.

Per-experiment artifacts (configs, raw metrics, plots) will live under [`results/`](./results) organized by strategy.

## Reproducibility

Every result in this repository is reproducible. See [`docs/reproducibility.md`](./docs/reproducibility.md) for exact commands, seeds, and dataset revisions.

## License

MIT — see [LICENSE](./LICENSE).
