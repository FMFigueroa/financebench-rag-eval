"""Microsoft Foundry embedding client (provider transversal cloud).

Wrapper sobre el SDK `openai` apuntando al endpoint Azure OpenAI v1
(`https://<resource>.openai.azure.com/openai/v1/`). Carga endpoint +
deployment desde `.env` y autentica via Microsoft Entra ID usando OAuth.
usa el token de la sesión `az login` — sin API keys que rotar ni almacenar.

Por qué Microsoft Entra ID en vez de API key:
  - Cero secretos persistentes en el código ni en el `.env`.
  - Auth atada a tu identidad de Microsoft Entra ID (revocación inmediata si rotas).
  - Estándar production: mismo patrón funciona en CI/CD con Managed Identity.
  - Pre-requisito: ejecutar `az login` antes de usar este módulo.

Por qué SDK `openai` y no `azure-ai-inference`:
  - Microsoft mismo recomienda `openai` para embeddings (el endpoint del
    Project del Foundry SDK no rutea embedding requests).
  - `azure-ai-inference` está deprecated (retire 26 ago 2026).
  - Compatibilidad 1:1 con OpenAI directo (cambiar 1 var migra de Azure
    OpenAI a OpenAI sin tocar código).

Uso típico:

    from src.embeddings.foundry_client import FoundryEmbedder

    embedder = FoundryEmbedder()
    vectors = embedder.embed(["chunk de prueba", "otro chunk"])
    # vectors: np.ndarray shape (2, 1536) para text-embedding-3-small

Memoria descriptiva: https://www.notion.so/35a6ad30ca11811d96ebf4a9d7dde20b
Setup detallado:     docs/foundry_setup.md
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Repo root = 2 niveles arriba de este file (src/embeddings/foundry_client.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_DIR = REPO_ROOT / "data" / "processed" / "chunks"

# Scope de Microsoft Entra ID para Cognitive Services / Azure OpenAI. Identifier
# que indica qué API estás autorizando — no es una URL llamable.
AZURE_OPENAI_SCOPE = "https://cognitiveservices.azure.com/.default"


@dataclass(frozen=True)
class FoundryConfig:
    endpoint: str
    deployment: str

    @classmethod
    def from_env(cls) -> FoundryConfig:
        endpoint = os.getenv("AZURE_FOUNDRY_ENDPOINT", "")
        deployment = os.getenv("AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT", "")

        missing = [
            k
            for k, v in {
                "AZURE_FOUNDRY_ENDPOINT": endpoint,
                "AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT": deployment,
            }.items()
            if not v or "PEGA" in v
        ]
        if missing:
            raise RuntimeError(
                f"Faltan variables Foundry en .env: {missing}. Ver docs/foundry_setup.md."
            )
        return cls(endpoint=endpoint, deployment=deployment)


class FoundryEmbedder:
    """Embedder que consume modelos de embeddings desde Microsoft Foundry
    via el endpoint Azure OpenAI v1 + SDK `openai` + auth Microsoft Entra ID."""

    def __init__(self, config: FoundryConfig | None = None) -> None:
        self.config = config or FoundryConfig.from_env()

        # Token provider: función que devuelve un Bearer token fresco para Microsoft
        # Entra ID cada vez que se invoca. DefaultAzureCredential autodetecta la fuente
        # (az CLI, env vars, Managed Identity, etc.) en orden estándar.
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            AZURE_OPENAI_SCOPE,
        )

        self._client = OpenAI(
            base_url=self.config.endpoint,
            api_key=token_provider(),
        )

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """Vectoriza una lista de textos. Devuelve ndarray shape (n, dim).

        ⚠️ Sin batch logic interna — una sola llamada por invocación. Para
        corpus grandes (≥2K inputs por request es el max de Azure OpenAI),
        orquestar el batch desde fuera con cost tracking (sub-bloque 7).
        """
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        response = self._client.embeddings.create(
            input=texts,
            model=self.config.deployment,
        )
        vectors = [np.asarray(item.embedding, dtype=np.float32) for item in response.data]
        return np.stack(vectors, axis=0)

    def embedding_dim(self) -> int:
        """Devuelve la dimensión del vector que produce el modelo desplegado.

        Para `text-embedding-3-small` es 1536. Para `-large` es 3072.
        Si necesitas el valor antes de la primera llamada, hardcodéalo desde
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
    sample = "smoke test from Foundry"
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
    como referencia). Pasale otro path si quieres probar con un doc distinto.
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
    result = smoke_test_with_real_chunk() if mode == "real" else smoke_test()
    print(json.dumps(result, indent=2))
