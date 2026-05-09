"""CLI entry point: embed the full FinanceBench corpus through an embedder.

Usage:
    uv run python scripts/embed_corpus.py
    uv run python scripts/embed_corpus.py --limit 50    # test mode
    uv run python scripts/embed_corpus.py --pause 30    # custom pause

Output goes to: data/processed/embeddings/<embedder_name>/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the repo root is on sys.path so `src.*` imports resolve.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.embeddings.batch_runner import embed_corpus
from src.embeddings.foundry_client import FoundryEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed FinanceBench corpus")
    parser.add_argument(
        "--embedder",
        choices=["foundry-text-3-small"],
        default="foundry-text-3-small",
        help="Which embedder to use (default: foundry-text-3-small)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N chunks (for testing). Default: process all.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=60.0,
        help="Seconds to pause between batches (default: 60)",
    )
    args = parser.parse_args()

    if args.embedder == "foundry-text-3-small":
        embedder = FoundryEmbedder()
        embedder_name = "foundry_text_3_small"
    else:
        raise ValueError(f"Unknown embedder: {args.embedder}")

    embed_corpus(
        embedder=embedder,
        embedder_name=embedder_name,
        chunks_limit=args.limit,
        pause_between_batches=args.pause,
    )


if __name__ == "__main__":
    main()
