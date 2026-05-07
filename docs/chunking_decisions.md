# Chunking decisions — log

> Living document of the chunking design choices for `financebench-rag-eval`.
> Each entry follows the **ADR (Architecture Decision Record)** pattern: context, options considered, decision, rationale. New decisions get appended in chronological order; existing ones get updated only with a `**Update <date>:**` note (never silently rewritten).
>
> The decisions logged here drive the implementation in [`scripts/build_corpus.py`](../scripts/build_corpus.py) and the demos in [`notebooks/02_financebench_exploration.ipynb`](../notebooks/02_financebench_exploration.ipynb) §4. The conceptual framework lives in [`docs/CONTEXT.md`](./CONTEXT.md) §13 (parsing) and §14 (chunking).

---

## Decision 1 — Tokenizer reference: `tiktoken cl100k_base`

**Status**: Accepted (2026-05-07)
**Sub-block**: 5

### Context

The chunk size budget (512 tokens) only makes sense relative to a tokenizer. The 5 embedders we'll evaluate use different tokenizers (OpenAI cl100k_base, BGE-M3 XLM-R, Voyage proprietary, Jina v5 internal, Qwen3 internal). Each measures the same string with up to ~10% variance.

### Options considered

| Option | Cost | Comparison clarity |
|---|---|---|
| **A) Single reference tokenizer (`tiktoken cl100k_base`)** | One chunking pass | Clean — all embedders see identical chunks |
| B) Per-embedder tokenizer | 5× chunking compute + 5× disk | Muddied — each embedder sees different chunks |
| C) Char-based approximation (chars/4) | Cheap | Rough — drifts 15-20% on financial text |

### Decision

**Option A** — `tiktoken cl100k_base` as the project's single token-counting reference.

### Rationale

1. All 5 target embedders accept ≥8192 token contexts (OpenAI 8191, BGE-M3 8192, Voyage 16k, Jina v5 8192, Qwen3 32k). A 512-token chunk measured in cl100k will never exceed any embedder's actual window even if their tokenizer counts ~10% more.
2. Standard practice in RAG benchmarks (FinanceBench paper, LangChain default, LlamaIndex default).
3. Single corpus = clean comparison between embedders.
4. If Stage 2-3 detects retrieval failures attributable to tokenizer drift on a specific embedder, a focused experiment can re-chunk that embedder only.

---

## Decision 2 — False-positive table filter: numeric-column heuristic

**Status**: Accepted (2026-05-07)
**Sub-block**: 5

### Context

`pdfplumber` with `text-based` strategy detects ANY columnar text layout as a "table" — including cover pages, page headers, callouts, and tab-separated lists. Without filtering, the 3M cover page parses as a 70×13 "table" with the same cell density (~35%) as the real Income Statement (33×10). Density alone does not discriminate.

### Options considered

| Option | Discriminator | Result on cover-page-vs-Income-Statement |
|---|---|---|
| A) Shape only (≥3 rows × ≥2 cols) | Dimensions | Both pass — cover 70×13 ✗, IS 33×10 ✓ (FAILS to discriminate) |
| B) Shape + density (≥30% non-empty cells) | Cell density | Both ~35% — FAILS to discriminate |
| **C) Shape + numeric-column heuristic** | At least 1 column with ≥30% numeric cells | Cover ✗, IS ✓ — DISCRIMINATES |
| D) ML classifier (table-vs-text) | Model output | Overkill for Stage 1; deferred |

### Decision

**Option C** — `(rows ≥ 3) AND (cols ≥ 2) AND (∃ column with ≥30% numeric cells)`. Numeric pattern: `^[\d,.()$%\s+-]+$`.

### Rationale

1. Real financial tables (Income Statement, Balance Sheet, Cash Flows, detail Notes) all have at least one numeric column by construction. Cover pages, headers, and text-block artifacts do not.
2. Empirical impact measured during sanity test: cut 3M's chunk count from 1,029 → 502 (~50% noise reduction). Full corpus dropped from ~63K (estimated) → 31K (real).
3. Threshold of 30% picked empirically: lower (10-20%) lets through some cover pages with numeric fragments; higher (50%+) excludes legitimate Notes mixing prose and numbers.
4. Pattern allows for typical financial cell formats: `32,765` / `$1,577` / `(123.4)` / `5.6%` / `-1.2`.

---

## Decision 3 — Chunk schema: 6 minimal fields

**Status**: Accepted (2026-05-07)
**Sub-block**: 5

### Context

Each chunk needs metadata for two downstream stages: (a) embedding (only `text` is consumed), (b) retrieval evaluation (the `page_num` of the chunk is compared against `evidence_page_num` from the FinanceBench dataset to compute Recall@k, MRR, NDCG, MAP).

### Decision

```json
{
  "chunk_id":   "<doc_name>_<idx_within_doc:04d>",
  "doc_name":   "3M_2018_10K",
  "page_num":   56,
  "chunk_type": "text" | "table",
  "text":       "...",
  "n_tokens":   <int>
}
```

### Rationale

- `page_num` is the **non-negotiable critical field** — without it, retrieval metrics cannot be computed.
- `chunk_type` enables post-hoc analysis (e.g., *"OpenAI wins on text chunks, BGE-M3 wins on table chunks"*) without re-tokenizing.
- `n_tokens` precomputed avoids re-running the tokenizer for stats / cost projections / chunk-budget validation.
- `chunk_id` is doc-scoped and zero-padded so sort order matches insertion order.

### Fields explicitly NOT included (and why)

| Field | Why excluded for Stage 1 |
|---|---|
| `bbox` (page coordinates) | Useful for visual debug, but adds ~100 bytes per chunk × 31K chunks = 3 MB overhead. Defer until visual debug is needed. |
| `parent_chunk_id` | Useful to link header-repetition siblings, but baseline doesn't use sibling info. Defer until Stage 2-3 if metrics demand. |
| `table_row_range` | Useful for table-aware retrieval, but baseline embeds chunks independently. Defer. |

---

## Decision 4 — Header-repetition with token threshold

**Status**: Accepted (2026-05-07)
**Sub-block**: 5

### Context

Tables that exceed 512 tokens (~5-10% of the corpus, mostly detail Notes on stock-based compensation, segment reporting, leases) need splitting. Naive row-splitting orphans the header and breaks the row ↔ column association. Header-repetition prepends the header to every sub-chunk.

### Options considered

| Option | Behavior |
|---|---|
| A) Always apply header-repetition | Even small tables get fragmented unnecessarily |
| **B) Threshold-based**: tables ≤512 tokens emit as 1 chunk; >512 trigger header-rep | Preserves small-table integrity, applies the strategy only where needed |
| C) Whole-table-as-1-chunk (no splitting) | Breaks 512 budget for ~5-10% of tables; bimodal chunk-length distribution |
| D) Row-as-sentence linearization | Best for narrative queries, worst for lookup queries; defer to Stage 3 |

### Decision

**Option B** — threshold-based header-repetition. `header_row_idx=0` (first row assumed to be the header). Multi-row headers and sub-headers not detected.

### Rationale

1. Most financial tables in 10-K (Income Statement, Balance Sheet, Cash Flows) have a single-row header in row 0. Heuristic detection of header rows by font-style or other signals adds complexity without measurable upside for the baseline.
2. Threshold avoids fragmenting small self-contained tables.
3. Row 0 is assumed the header even for atypical tables (footnote tables, list-like tables) — the embedder handles the noise.
4. If Stage 2-3 detects systematic failure on tables with multi-row headers, header detection can be added then.

---

## Decision 5 — Asymmetric overlap: text yes, table no

**Status**: Accepted (2026-05-07)
**Sub-block**: 5

### Context

Overlap between consecutive chunks is the standard mitigation for concepts being cut at chunk boundaries. The question is whether to apply it uniformly or differently for text vs. table chunks.

### Decision

- **Text chunks**: 50-token overlap measured with the reference tokenizer (cl100k_base). Window of 512 tokens, stride of 462.
- **Table sub-chunks** (when header-repetition applies): **no overlap** between sibling sub-chunks of the same table.

### Rationale

The header-repetition mechanism in Decision 4 already serves the role overlap plays for text: it preserves the row ↔ column association across chunk boundaries. Adding 50 extra tokens of "the previous rows" on top of the repeated header would be redundant and would increase token cost without improving retrieval.

The asymmetry is intentional, not an oversight, and is documented in `scripts/build_corpus.py` and `notebooks/02_financebench_exploration.ipynb` §4.4.

---

## Decision 6 — Numeric-column ratio threshold: 30%

**Status**: Accepted (2026-05-07)
**Sub-block**: 5 (emerged during sanity test)

### Context

Decision 2 picks the numeric-column heuristic but the ratio is a tunable parameter. Pick it too low and false positives slip through (cover pages with stray digits); too high and Notes pages with mixed prose+numbers get rejected.

### Options considered (empirically tested on 3M)

| Threshold | False positives surviving | Real tables rejected |
|---|---|---|
| 0.10 | Several (cover pages with footnote markers) | 0 |
| 0.20 | Some (TOC pages with page numbers in 1 column) | 0 |
| **0.30** | None visible | 0 |
| 0.50 | None | A few (Notes pages mixing prose + small tables) |
| 0.70 | None | Many |

### Decision

`NUMERIC_COL_RATIO = 0.30`.

### Rationale

The 30% threshold is the elbow point: highest threshold that excludes all observed false positives without rejecting any real table. The decision is empirical — if Stage 2-3 retrieval analysis reveals systematic loss on tables that fall just below this threshold, it can be relaxed (with documented evidence of which tables are recovered).

---

## Update log

- **2026-05-07** — Initial 5 decisions (1-5) accepted. Decision 6 emerged during sanity test against the 3M Income Statement.
