# Microsoft Foundry — Setup Guide

> **Provider transversal cloud** del proyecto. Sirve para acceder a `text-embedding-3-small` (Stage 1) y a los modelos de embedding/LLM/rerank de Stages 2-3 sin manejar N API keys distintas.
>
> Este doc cubre **qué SDK + qué endpoint usar y por qué**, cómo conseguir credenciales del portal Foundry, y cómo validar el setup. La integración en código vive en [`src/embeddings/foundry_client.py`](../src/embeddings/foundry_client.py) y se documenta paso a paso en [`notebooks/03_foundry_setup.ipynb`](../notebooks/03_foundry_setup.ipynb).

---

## Prerequisitos

1. **Cuenta de Azure activa** (free tier sirve para smoke tests)
2. **Acceso a Microsoft Foundry** ([https://ai.azure.com](https://ai.azure.com)) con la misma cuenta de Azure
3. **Un Project de Foundry creado** (no un Hub legacy — Foundry usa el concepto "Project")
4. **Un deployment de `text-embedding-3-small`** dentro de ese Project (UI → *Models + endpoints* → *Deploy*). En Foundry **todo modelo se deploya primero** y eso te genera un `Target URI` + `Key` consumibles vía SDK.

---

## Decisión técnica — qué SDK y qué endpoint usar

Un Foundry resource expone **3 servicios distintos** desde el mismo dominio del resource. Cada uno tiene su SDK y su path. Elegir mal el combo causa errores opacos (404/path mismatch). Esta es la matriz oficial:

| SDK Python | Endpoint | Sirve para |
|------------|----------|------------|
| `azure-ai-projects` | `https://<resource>.services.ai.azure.com/api/projects/<project-name>` | Agents, evaluations, fine-tuning, project management |
| `azure-ai-inference` | `https://<resource>.services.ai.azure.com/models` | Foundry Models multi-modelo (Cohere, Llama, BGE…) — ⚠️ **deprecated, retire 26 ago 2026** |
| **`openai`** | **`https://<resource>.openai.azure.com/openai/v1/`** | **Embeddings + Chat + Image — full OpenAI API surface** ✅ |

### En este proyecto usamos `openai` SDK + endpoint `/openai/v1/`

Razones (todas confirmadas en docs oficiales de Microsoft):

1. **Microsoft mismo lo recomienda** para embeddings — cita textual:
   > *"Use the OpenAI SDK endpoint for generating embeddings. The project endpoint used by the Foundry SDK doesn't currently route embedding requests."*
2. **`azure-ai-inference` está deprecated** — se retira el **26 de agosto 2026**. Microsoft pide migrar al SDK `openai`.
3. **`azure-ai-projects` no rutea embeddings** — es solo para agents/evaluations/fine-tuning.
4. **Portabilidad** — si en algún momento queremos migrar de Azure OpenAI a OpenAI directo, cambiamos 1 línea (`base_url`). Mismo código, otro proveedor.

### Detalles importantes del endpoint

- **Subdominio `.openai.azure.com`** (NO `.services.ai.azure.com`). El endpoint Azure OpenAI vive en su propio subdominio del resource Foundry. Ambos coexisten en el mismo recurso.
- **Path `/openai/v1/`** con **trailing slash**. El SDK `openai` espera la URL terminada en `/` para concatenar `/embeddings` correctamente.
- **Sin `api-version` en query string**. La versión "v1" ya está en el path. La API es estable y el SDK la maneja internamente.

---

## Conseguir las 3 variables que van al `.env`

### 1. `AZURE_FOUNDRY_ENDPOINT`

UI: Vas al deployment del modelo (UI → *Models + endpoints* → click en `text-embedding-3-small`). El portal te muestra **Target URI** del resource:

```
https://<resource>.openai.azure.com
```

⚠️ **Pero en el `.env` necesitas el path completo** que espera el SDK `openai`:

```
https://<resource>.openai.azure.com/openai/v1/
```

(con `/openai/v1/` agregado al final, **incluyendo trailing slash**).

### 2. `AZURE_FOUNDRY_API_KEY`

UI: Mismo panel del deployment, campo **Key** (con icono de ojo para revelar).

Es un string opaco largo. **NO la pegues nunca en el chat ni en commits.** Va directo al `.env` local.

### 3. `AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT`

UI: *Models + endpoints* → tu deployment → **Name**.

⚠️ **Gotcha**: el deployment name NO siempre coincide con el model name de OpenAI. Si lo creaste con el nombre por defecto, suele ser `text-embedding-3-small`. Si le pusiste otro (`my-embedder-prod`, etc.), usa ese.

---

## Llenar el `.env`

```bash
cp .env.example .env
# Edita el .env y reemplaza los placeholders con tus valores reales:
#   AZURE_FOUNDRY_ENDPOINT=https://<resource>.openai.azure.com/openai/v1/
#   AZURE_FOUNDRY_API_KEY=<tu key>
#   AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

Verificar que cargan (sin leakear el contenido de la key):

```bash
uv run python -c "
from dotenv import load_dotenv
load_dotenv()
import os
keys = ['AZURE_FOUNDRY_ENDPOINT', 'AZURE_FOUNDRY_API_KEY', 'AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT']
for k in keys:
    v = os.getenv(k, '')
    ok = bool(v) and 'PEGA' not in v
    masked = v[:12] + '...' + v[-4:] if ok and len(v) > 16 else v
    print(f\"{'✅' if ok else '❌'} {k}: {masked if ok else '(no configurado)'}\")
"
```

---

## Auth: solo API key (NO Microsoft Entra ID)

Cita oficial de la doc de Azure OpenAI embeddings:

> *"The Azure OpenAI embeddings API does not currently support Microsoft Entra ID with the v1 API. Use API key authentication."*

Sin `DefaultAzureCredential`, sin `az login`, sin token providers. Solo la key del `.env` pasada como `api_key=<key>` en el constructor del cliente OpenAI.

---

## Smoke test desde Python (con el SDK)

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
    base_url=os.getenv("AZURE_FOUNDRY_ENDPOINT"),
)

response = client.embeddings.create(
    input="smoke test from Foundry",
    model=os.getenv("AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT"),
)

vec = response.data[0].embedding
print(f"vector_dim   : {len(vec)}")           # esperado: 1536
print(f"first_5      : {vec[:5]}")
print(f"total_tokens : {response.usage.total_tokens}")
```

**Output esperado**:

```
vector_dim   : 1536
first_5      : [0.00239, -0.0090, -0.0110, -0.0387, -0.0294]
total_tokens : 4
```

El notebook [`notebooks/03_foundry_setup.ipynb`](../notebooks/03_foundry_setup.ipynb) ejecuta este flujo paso a paso con explicaciones pedagógicas.

---

## Limits importantes (afectan el batch de los 31K chunks)

| Limit | Valor | Impacto en este proyecto |
|-------|-------|--------------------------|
| Max tokens por input | **8,192** | OK — nuestros chunks son 512 tokens (16× margen) |
| Max array size por request | **2,048 inputs** | Tenemos 31,216 chunks → necesitamos **≥16 batches** |
| Quota default | **350,000 TPM** por región | 31K chunks × 512 tokens ≈ 16M tokens → mínimo ~46 min teórico |

Conclusión: **no podemos hacer una sola request** con los 31K chunks. Hay que batchear (planeado en `Sub-bloque 7 — Cost tracking básico` del notebook).

---

## Costos

`text-embedding-3-small` cuesta **\$0.02 por 1M tokens** vía Azure (mismo precio que OpenAI directo). Vectorizar los 31K chunks del FinanceBench Loader (~12.58M tokens según el log del Stage 1) cuesta:

```
12.58M tokens × $0.02/M ≈ $0.25 por embedder pass
```

**Por qué `-small` y no `-large`**: usamos `text-embedding-3-small` (1536 dim) en lugar de `text-embedding-3-large` (3072 dim) porque es **6× más barato** y para el dominio financiero estructurado (10-K filings) el lift de calidad de `large` no compensa el costo. Si en algún punto queremos comparar contra `large`, el deployment es independiente y se puede correr en paralelo (cuesta ~\$1.64 por pass).

Vale la pena **cachear los vectores en disco** después del primer pass para no re-pagar si re-corremos por bug. Eso queda planeado en el `Sub-bloque 7 — Cost tracking básico`.

---

## Troubleshooting

| Error | Causa probable | Cómo fixear |
|-------|----------------|-------------|
| `401` o `403` | API key inválida o revocada | Regenerar key en el portal y actualizar `.env` |
| `404 Resource not found` | Endpoint sin `/openai/v1/` o base URL mal escrito | Verificar que `AZURE_FOUNDRY_ENDPOINT` termina en `/openai/v1/` (con slash) |
| `400 Bad Request` | `model` no coincide con el deployment name del portal | Verificar exact match en *Models + endpoints* → Name |
| `429 Too Many Requests` | Quota TPM agotada | Esperar 1 min o pedir aumento de quota en el portal |
| `JSONDecodeError` en cliente | Respuesta no-JSON (probablemente endpoint mal) | Re-revisar path completo del endpoint |

---

## Referencias oficiales

- [Microsoft Foundry — overview](https://learn.microsoft.com/azure/ai-foundry/)
- [Microsoft Foundry — SDKs and Endpoints](https://learn.microsoft.com/azure/ai-foundry/concepts/sdks-and-endpoints) (mapa oficial de los 3 SDKs)
- [Endpoints for Microsoft Foundry Models](https://learn.microsoft.com/azure/ai-foundry/foundry-models/concepts/endpoints) (incluye nota de deprecación de `azure-ai-inference`)
- [Generate embeddings with Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/how-to/embeddings) (doc canónica de embeddings + best practices + limits)
- [OpenAI Python SDK (GitHub)](https://github.com/openai/openai-python)
- [Azure OpenAI pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)
