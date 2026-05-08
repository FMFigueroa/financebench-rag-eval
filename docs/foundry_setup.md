# Microsoft Foundry — Setup Guide

> **Provider transversal cloud** del proyecto. Sirve para acceder a `text-embedding-3-small` (Stage 1) y a los modelos de embedding/LLM/rerank de Stages 2-3 sin manejar N API keys distintas.
>
> Este doc cubre **cómo conseguir credenciales y validarlas**. La integración en código vive en [`src/embeddings/foundry_client.py`](../src/embeddings/foundry_client.py) y se documenta paso a paso en [`notebooks/03_foundry_setup.ipynb`](../notebooks/03_foundry_setup.ipynb).

---

## ¿Qué necesito antes de empezar?

1. Cuenta de Azure activa (free tier sirve para smoke tests)
2. Acceso a [Microsoft Foundry](https://ai.azure.com) (login con la misma cuenta de Azure)
3. Un **Project** de Foundry creado (no un Hub legacy — Foundry usa el concepto "Project")
4. Un **deployment** de `text-embedding-3-small` dentro de ese Project (UI → Models + endpoints → Deploy)

---

## Conseguir las 3 variables que van al `.env`

### 1. `AZURE_FOUNDRY_ENDPOINT`

UI: **Project overview → Project details → Project endpoint**

Dos formatos válidos:

- **Inference endpoint** (recomendado, multi-modelo): `https://<resource>.services.ai.azure.com/models`
- **Azure OpenAI v1 endpoint** (solo modelos OpenAI): `https://<resource>.openai.azure.com/openai/v1/`

Para este proyecto usamos el primero porque accedemos a múltiples modelos del catalog (no solo OpenAI).

### 2. `AZURE_FOUNDRY_API_KEY`

UI: **Project overview → Project details → Keys → "Show key"**

Es un string opaco largo. **NO la pegues nunca en el chat ni en commits.** Va directo al `.env` local.

### 3. `AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT`

UI: **Models + endpoints → tu deployment → Name**

⚠️ **Gotcha**: el deployment name NO siempre coincide con el model name de OpenAI. Si lo creaste con el nombre por defecto, suele ser `text-embedding-3-small`. Si le pusiste otro (`my-embedder-prod`, etc.), usa ese.

---

## Llenar el `.env`

```bash
cp .env.example .env
# Edita el .env y reemplaza los placeholders:
#   AZURE_FOUNDRY_ENDPOINT=PEGA_TU_ENDPOINT_AQUI
#   AZURE_FOUNDRY_API_KEY=PEGA_TU_KEY_AQUI
#   AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

Verificar que cargan:

```bash
uv run python -c "from dotenv import load_dotenv; load_dotenv(); import os; \
  ok = all(os.getenv(k) and 'PEGA' not in os.getenv(k) for k in ['AZURE_FOUNDRY_ENDPOINT','AZURE_FOUNDRY_API_KEY','AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT']); \
  print('✅ OK' if ok else '❌ falta cargar variables o quedaron placeholders')"
```

---

## Smoke test (sin código aún, solo curl)

Antes de tocar Python, validar que la combinación endpoint + key + deployment responde HTTP 200 desde la línea de comandos:

```bash
source .env

curl -X POST "${AZURE_FOUNDRY_ENDPOINT%/}/embeddings?api-version=2024-10-21" \
  -H "Content-Type: application/json" \
  -H "api-key: ${AZURE_FOUNDRY_API_KEY}" \
  -d "{\"input\":[\"hello world\"],\"model\":\"${AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT}\"}" \
  | head -c 200
```

Resultado esperado: un JSON que empieza con `{"data":[{"object":"embedding","index":0,"embedding":[...]`. Si recibes 401 → key mal. Si recibes 404 → endpoint o deployment mal escrito. Si recibes 429 → quota agotada.

El siguiente paso es enchufar esto desde Python en el notebook `03_foundry_setup.ipynb`.

---

## Costos

`text-embedding-3-small` cuesta **$0.02 por 1M tokens** vía Azure (mismo precio que OpenAI directo). Vectorizar los 31K chunks del FinanceBench Loader (~12.58M tokens según el log del Stage 1) cuesta:

```
12.58M tokens × $0.02/M ≈ $0.25 por embedder pass
```

**Nota**: usamos `text-embedding-3-small` (1536 dim) en lugar de `large` (3072 dim) porque es **6× más barato** y para el dominio financiero (10-K filings) el lift de calidad de `large` no compensa. Si querés re-embeddear todo con `large`, el deployment es independiente y podés hacerlo en paralelo (cuesta ~$1.64 por pass).

Una sola corrida. Si re-corres por bug, son otros $1.64. Vale la pena cachear los vectores en disco después del primer pass (planeado en `Sub-bloque 7 — Cost tracking básico`).

---

## Referencias oficiales

- [Azure AI Foundry overview](https://learn.microsoft.com/azure/ai-foundry/)
- [azure-ai-inference Python SDK readme](https://learn.microsoft.com/python/api/overview/azure/ai-inference-readme)
- [Foundry model catalog](https://ai.azure.com/explore/models)
- [Embeddings via Azure OpenAI — pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)
