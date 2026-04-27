"""Smoke tests verifying the runtime environment is configured correctly.

These tests do not exercise project logic — they fail fast if the development
environment drifts from what `pyproject.toml` declares.
"""

import sys


def test_python_version() -> None:
    assert sys.version_info >= (3, 12), f"Need Python 3.12+, got {sys.version_info}"


def test_production_imports_resolve() -> None:
    import anthropic  # noqa: F401
    import cohere  # noqa: F401
    import datasets  # noqa: F401
    import dotenv  # noqa: F401
    import openai  # noqa: F401
    import pdfplumber  # noqa: F401
    import qdrant_client  # noqa: F401
    import sentence_transformers  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import voyageai  # noqa: F401
