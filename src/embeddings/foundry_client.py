"""Microsoft Foundry embedding client (provider transversal cloud).

Wrapper sobre `azure.ai.inference.EmbeddingsClient` que carga credenciales
desde `.env` y expone una interfaz consistente con los otros embedders del
proyecto (`OpenAIEmbedder`, `BGEEmbedder`, etc.) para drop-in en el
`Eval Pipeline`.

Uso típico:

    from src.embeddings.foundry_client import FoundryEmbedder

    embedder = FoundryEmbedder()
    vectors = embedder.embed(["chunk de prueba", "otro chunk"])
    # vectors: np.ndarray shape (2, 1536) para text-embedding-3-small

Memoria descriptiva: https://www.notion.so/35a6ad30ca11811d96ebf4a9d7dde20b
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

# Repo root = 2 niveles arriba de este file (src/embeddings/foundry_client.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_DIR = REPO_ROOT / "data" / "processed" / "chunks"


@dataclass(frozen=True)
class FoundryConfig:
    endpoint: str
    api_key: str
    deployment: str

    @classmethod
    def from_env(cls) -> "FoundryConfig":
        endpoint = os.getenv("AZURE_FOUNDRY_ENDPOINT", "")
        api_key = os.getenv("AZURE_FOUNDRY_API_KEY", "")
        deployment = os.getenv("AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT", "")

        missing = [
            k
            for k, v in {
                "AZURE_FOUNDRY_ENDPOINT": endpoint,
                "AZURE_FOUNDRY_API_KEY": api_key,
                "AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT": deployment,
            }.items()
            if not v or "PEGA" in v
        ]
        if missing:
            raise RuntimeError(
                f"Faltan variables Foundry en .env: {missing}. "
                f"Ver docs/foundry_setup.md."
            )
        return cls(endpoint=endpoint, api_key=api_key, deployment=deployment)


class FoundryEmbedder:
    """Embedder que consume modelos de embeddings desde Microsoft Foundry."""

    def __init__(self, config: FoundryConfig | None = None) -> None:
        self.config = config or FoundryConfig.from_env()
        self._client = EmbeddingsClient(
            endpoint=self.config.endpoint,
            credential=AzureKeyCredential(self.config.api_key),
        )

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """Vectoriza una lista de textos. Devuelve ndarray shape (n, dim)."""
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        response = self._client.embed(input=texts, model=self.config.deployment)
        vectors = [np.asarray(item.embedding, dtype=np.float32) for item in response.data]
        return np.stack(vectors, axis=0)

    def embedding_dim(self) -> int:
        """Devuelve la dimensión del vector que produce el modelo desplegado.

        Para `text-embedding-3-small` es 1536. Para `-large` es 3072.
        Si necesitás el valor antes de la primera llamada, hardcodeálo desde
        el config; aquí lo derivamos del primer embed real para evitar
        suposiciones.
        """
        return self.embed(["probe"]).shape[1]


def load_first_chunk(jsonl_path: Path) -> dict:
    """Carga el primer chunk de un .jsonl del corpus FinanceBench.

    Estructura esperada del chunk:
        {doc_name, page_num, chunk_type, text, n_tokens, chunk_id}
    """
    with jsonl_path.open("r") as f:
        line = f.readline()
    return json.loads(line)


def smoke_test() -> dict:
    """Smoke test de conexión + 1 embedding sobre texto sintético.

    Útil para validar credenciales sin tocar el corpus. Lanza si algo falla.
    """
    embedder = FoundryEmbedder()
    sample = "Apple's revenue in fiscal year 2023 was $383 billion."
    vec = embedder.embed([sample])
    return {
        "endpoint": embedder.config.endpoint,
        "deployment": embedder.config.deployment,
        "input_text": sample,
        "vector_shape": tuple(vec.shape),
        "vector_dim": int(vec.shape[1]),
        "first_5_values": vec[0, :5].tolist(),
        "norm_l2": float(np.linalg.norm(vec[0])),
    }


def smoke_test_with_real_chunk(
    jsonl_path: Path | None = None,
) -> dict:
    """Smoke test sobre 1 chunk real del corpus FinanceBench.

    Por default agarra el primer chunk de `MICROSOFT_2023_10K.jsonl` (estable
    como referencia). Pasale otro path si querés probar con un doc distinto.
    """
    path = jsonl_path or (DEFAULT_CHUNKS_DIR / "MICROSOFT_2023_10K.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. ¿Está cerrado el sub-bloque de FinanceBench Loader?"
        )

    chunk = load_first_chunk(path)
    embedder = FoundryEmbedder()
    vec = embedder.embed([chunk["text"]])

    return {
        "endpoint": embedder.config.endpoint,
        "deployment": embedder.config.deployment,
        "chunk_id": chunk["chunk_id"],
        "doc_name": chunk["doc_name"],
        "page_num": chunk["page_num"],
        "n_tokens": chunk["n_tokens"],
        "text_preview": chunk["text"][:120] + "...",
        "vector_shape": tuple(vec.shape),
        "vector_dim": int(vec.shape[1]),
        "first_5_values": vec[0, :5].tolist(),
        "norm_l2": float(np.linalg.norm(vec[0])),
    }


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    if mode == "real":
        result = smoke_test_with_real_chunk()
    else:
        result = smoke_test()
    print(json.dumps(result, indent=2))
