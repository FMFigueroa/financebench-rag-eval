# 📘 Financial context and glossary

> **Audience**: anyone entering this repo without a finance background who needs to understand **what we're evaluating, against what, and why it matters**.
>
> This document explains every financial term, acronym, and metric used in the project, in plain English, with concrete examples grounded in our dataset (FinanceBench).

---

## Table of contents

1. [The game: public companies and their reports](#1-the-game-public-companies-and-their-reports)
2. [The SEC — the referee](#2-the-sec--the-referee)
3. [Filings — mandatory company reports](#3-filings--mandatory-company-reports)
4. [Earnings — careful, NOT a formal filing](#4-earnings--careful-not-a-formal-filing)
5. [GICS — how the universe of companies is classified](#5-gics--how-the-universe-of-companies-is-classified)
6. [PatronusAI and the FinanceBench dataset](#6-patronusai-and-the-financebench-dataset)
7. [What we validate in Stage 1 (Baselines)](#7-what-we-validate-in-stage-1-baselines)
8. [Passages and chunks: the units of RAG retrieval](#8-passages-and-chunks-the-units-of-rag-retrieval)
9. [Descriptive statistics primer (min, mean, median, percentiles)](#9-descriptive-statistics-primer-min-mean-median-percentiles)
10. [How we measure success: retrieval metrics](#10-how-we-measure-success-retrieval-metrics)

---

## 1. The game: public companies and their reports

When a company **goes public** (sells shares to the general public via NYSE, Nasdaq, etc.), strangers give it money in exchange for a slice of ownership. Those investors need **reliable information** to decide whether to buy, hold, or sell.

But they can't simply trust marketing speak — historically that ended in massive frauds (Enron, WorldCom). That's why a **federal referee** exists: the SEC.

---

## 2. The SEC — the referee

**SEC** = **Securities and Exchange Commission** (United States Securities and Exchange Commission).

It's the federal agency that **forces public companies** to publish standardized reports on a regular schedule. If you don't file them, you get delisted from the market or sued. It's the "Wall Street police".

These mandatory reports are called **filings**.

---

## 3. Filings — mandatory company reports

**Filing** = literally "a document submitted to the SEC". There are several types based on frequency and purpose:

| Filing | Frequency | What it contains |
|---|---|---|
| **10-K** | Annual | Complete fiscal-year report: audited financial statements, risks, corporate governance, MD&A. **The most important one.** |
| **10-Q** | Quarterly (every 3 months) | Stripped-down version of the 10-K, unaudited. 3 per year (Q4 is folded into the 10-K). |
| **8-K** | Event-driven | "Emergency" report when something material happens: CEO change, acquisition, major lawsuit, etc. |

**The 10-K is the company's financial bible**: 100-300 pages covering everything a professional investor needs to know.

### Anatomy of a 10-K (high level)

A 10-K is divided into parts and items by SEC regulation:

- **Part I — Business and Risk Factors**
  - Item 1: Business (operations, products, markets, competition)
  - Item 1A: Risk Factors (specific risks that could harm the business)
  - Item 2: Properties (physical assets, plants, offices)
  - Item 3: Legal Proceedings (significant ongoing lawsuits)
- **Part II — MD&A and Financial Statements** (the most analytically dense section)
  - Item 7: MD&A — Management's Discussion and Analysis (management's narrative on financial performance)
  - Item 8: Audited Financial Statements (Balance Sheet, Income Statement, Cash Flow Statement)
- **Part III — Governance** (directors, executive compensation)
- **Part IV — Exhibits and Signatures**

Most FinanceBench questions target **Item 7 (MD&A)** and **Item 8 (financial statements)**, where the hard numbers live.

---

## 4. Earnings — careful, NOT a formal filing

**Earnings releases** and **earnings calls** are communications a company makes **voluntarily** to its investors when reporting quarterly results. They typically include:

- A press release with highlights ("Q3 revenue grew 15% YoY...")
- A conference call with analysts (live Q&A)
- Presentation slides

They share financial information, but **they do NOT have the legal rigor of a 10-K**: not audited, don't follow SEC-mandated format, and reflect the management's "spin" more than regulatory scrutiny. That's why we tag them separately in the dataset.

> 💡 **Why this matters for our evaluation**: when you evaluate a RAG system on a mix of 10-K (formal) + earnings (informal) documents, you're testing the system on **heterogeneous source quality**. We document this trade-off in our analysis.

---

## 5. GICS — how the universe of companies is classified

**GICS** = **Global Industry Classification Standard**. A system developed by MSCI + S&P Dow Jones in 1999 to classify companies globally.

It divides the market into **11 main sectors**:

1. Energy
2. Materials
3. Industrials
4. Consumer Discretionary
5. Consumer Staples
6. Health Care
7. Financials
8. Information Technology
9. Communication Services
10. Utilities
11. Real Estate

GICS is hierarchical: 11 sectors → 25 industry groups → 74 industries → 163 sub-industries.

**Why we care for this project**: in our exploratory analysis, the 150 QA pairs cover **9 of the 11 sectors** — solid diversity. If all questions were Tech-only, the evaluation would be biased and our results wouldn't generalize. GICS lets us measure that "fairness of coverage" objectively.

---

## 6. PatronusAI and the FinanceBench dataset

**Patronus AI** is a San Francisco startup founded in 2023, specialized in **LLM evaluation for regulated domains** (finance, legal, healthcare). Their thesis: commercial LLMs (GPT, Claude, etc.) fail silently on technical questions in professional domains, and nobody was rigorously measuring it.

In **November 2023** they released **FinanceBench**: the first public LLM evaluation benchmark on real SEC filings. The original paper had 10,231 questions, but they only released **150 to the public** (the version we use) — the rest stays for enterprise clients.

### What they built exactly

- **150 real questions** that a junior financial analyst would ask
- Each question linked to a specific filing (10-K, 10-Q, 8-K, or earnings release)
- Each question with a **human-verified answer**
- Each answer accompanied by the **exact passage from the PDF** that justifies it → this is the famous `evidence` field

### How we use it

```
For each of the 150 questions:
  1. Take the question (e.g., "What was 3M's CapEx in FY2018?")
  2. Ask our RAG system to retrieve the most relevant chunks
  3. Compare the retrieved chunks against the original `evidence`
  4. If a chunk contains the evidence → ✅ hit
  5. If not → ❌ miss
  6. Average over 150 questions → metrics (Recall@k, MRR, NDCG, MAP)
```

The `evidence` field is what makes FinanceBench a **rigorous benchmark**: without ground truth, evaluation would be guesswork.

---

## 7. What we validate in Stage 1 (Baselines)

The goal of **Stage 1** is to answer two concrete questions:

1. **How well does a standard commercial embedder (OpenAI `text-embedding-3-large`) retrieve the evidence?** → reference baseline.
2. **Can a quality open-source embedder (BGE-M3) compete?** → alternative baseline, self-hosted (no API cost).

This sets the performance "floor". In **Stage 2** we add finance-specific embedders (Voyage finance) and SOTA 2025 techniques (Contextual Retrieval, Late Chunking) to measure how much they improve over the baselines.

---

## 8. Passages and chunks: the units of RAG retrieval

A **passage** is a fragment of text extracted from a larger document, sized to be meaningful on its own — typically a few related paragraphs.

### Hierarchy of text granularity

| Unit | What it is |
|---|---|
| **Document** | An entire file (e.g., a 200-page 10-K) |
| **Page** | A single page of the document |
| **Passage** | One or more related paragraphs (e.g., the section explaining CapEx) |
| **Token** | Word or subword (the model's atomic unit, ~3-4 chars on average) |

### Passages in FinanceBench

The `evidence_text` field that ships with each FinanceBench question **is a passage** — a fragment of the PDF that contains the answer. For example, the FY2018 CapEx question for 3M points to a passage from the Statement of Cash Flows that includes the line "Purchases of PP&E" with the value $1,577M (~2,800 characters of raw table text).

These passages are **human-curated**: Patronus annotators read each 10-K and selected which fragment justifies each answer. That makes them the **ground truth** for evaluating retrieval.

### Why passages are the key unit in RAG

If your RAG system operated on **entire documents** (200 pages → millions of tokens), the embedder couldn't process them (typical max ~512 tokens per call) and search would be imprecise (too much irrelevant context drowning the signal). If it operated on **individual words**, it would lose context.

**The passage is the sweet spot**: small enough that an embedding represents it well, large enough to contain a complete answer.

### Passage vs chunk

These two terms are used interchangeably in practice, with a subtle distinction:

| Term | Connotation |
|---|---|
| **Passage** | Academic/neutral term — agnostic about how the text was produced |
| **Chunk** | Operational RAG term — a passage produced by a specific **chunking strategy** |

In FinanceBench the `evidence_text` fields are **human-curated passages**. In our RAG system we'll generate **chunks automatically** with different strategies, and compare them against those human passages — that's how the metrics measure whether our chunking captures the same information humans selected.

### Chunking strategies compared in this project

The project benchmarks **4 chunking strategies** (formal decisions land in `docs/chunking_decisions.md` once Sub-block 6 closes):

1. **Naive fixed-size** — 512 tokens per chunk with 50-token overlap. The simplest baseline.
2. **Semantic chunking** — boundaries at sentence transitions detected by sentence-transformers. Tries to keep semantically coherent units together.
3. **Contextual Retrieval** ([Anthropic, 2024](https://www.anthropic.com/news/contextual-retrieval)) — each chunk is prepended with a short summary of its surrounding context, generated by Claude Haiku with prompt caching. Designed to fix the "isolated chunk" problem (a chunk that says "the company's revenue grew 15%" without saying *which* company).
4. **Late Chunking** ([Jina AI, 2024](https://arxiv.org/abs/2409.04701)) — embeds the entire document first, then chunks the embedding (not the text). Preserves cross-chunk context implicitly.

A "winning chunking strategy" is one whose generated chunks consistently match the human-curated `evidence_text` passages — measured by the metrics in the next section.

---

## 9. Descriptive statistics primer (min, mean, median, percentiles)

When we report dataset analyses (lengths, evidence counts, coverage) and benchmark results (Recall, MRR, NDCG, MAP), we don't report a single number — we report the **shape of the distribution**. This section covers the basics needed to read those reports correctly.

### The 7 terms in one table

| Term | What it is | When you use it |
|---|---|---|
| **min** | The **smallest** value in the dataset | Detect floor / low outliers |
| **max** | The **largest** value in the dataset | Detect ceiling / high outliers |
| **mean** | **Arithmetic average** (sum / count) | Center of the distribution, **but sensitive to outliers** |
| **median** (= **p50**) | **Middle value**: 50% of data below, 50% above | **Robust** center, ignores outliers |
| **p25** | **25th percentile**: 25% of data below this value | Lower bound of the "central body" |
| **p75** | **75th percentile**: 75% of data below this value | Upper bound of the "central body" |
| **p95** | **95th percentile**: 95% of data below this value | Long tail / extreme but non-outlier cases |

### Analogy: a line of 100 people sorted by height

- **min** = shortest person (position 1)
- **p25** = height of the person at position 25
- **median** (p50) = height of the person at position 50 (the one in the middle)
- **p75** = height of the person at position 75
- **p95** = height of the person at position 95 (almost at the end)
- **max** = tallest person (position 100)
- **mean** = average of ALL heights summed / 100

### Why we report all 7 together: the SHAPE of the distribution

**Mean vs median** is the key indicator of distribution shape:

| If... | It means... |
|---|---|
| `mean ≈ median` | **Symmetric** distribution (classic bell curve, few surprises) |
| `mean >> median` | **Long tail to the right** (some high outliers pull the average up) |
| `mean << median` | Long tail to the left (rare, but possible) |

### Worked example: evidence lengths from our analysis

```
evidence:  median = 1,450    mean = 1,712    p95 = 4,194    max = 12,123
```

Technical reading:

- **mean (1,712) > median (1,450)** → long tail to the right (some extremely long evidences pull the average up).
- **p95 (4,194) << max (12,123)** → the max is an **extreme outlier**, almost 3× the p95. Only ~5% of the dataset reaches 4,194 chars; one isolated case shoots up to 12,123.
- **median (1,450)** → **half** of the queries have evidence ≤ 1,450 chars. That's the "typical case" — more representative than the mean.

**This is why we use median + percentiles instead of just mean**: the mean alone doesn't show you the shape. If you were told "the average evidence measures 1,712 chars", you'd think most of it is close to 1,712. With the full distribution, you see that **half is under 1,450** and that **there's a 12,123-char monster distorting the average**.

### Golden rule

When the distribution has a long tail (typical for real-world data: salaries, file sizes, text lengths, retrieval scores), **always report median + percentiles**, not just mean. Honest journalism uses "median salary", not "average salary", for exactly this reason.

This is also why we report **bootstrap confidence intervals** for our retrieval metrics in later stages — instead of a single number for "Recall@5 = 0.80", we report something like "Recall@5 = 0.80 [0.74, 0.85] (95% CI)". The interval captures the uncertainty in our estimate, computed by resampling the evaluation set 1,000 times.

---

## 10. How we measure success: retrieval metrics

When our RAG system processes a question from the dataset, it returns an **ordered list of chunks** (the top-k most similar according to the embedder). But we need concrete numbers to say "this model is better than that one". That's where metrics come in.

**Analogy**: imagine you ask a librarian for the **5 most relevant books** to your question. Different metrics measure different aspects of how well they did:

- Is the **correct book** in the 5 they brought? → **Recall@k**
- In **what position** did they put it (first, third, last)? → **MRR**
- How well did they **order the whole list**? → **NDCG**
- If there are **multiple relevant books**, did they put them all near the top? → **MAP**

We use them together for a complete picture.

---

### 8.1 Recall@k — is the correct one in the top-k?

**Question it answers**: among the top-k the system returned, is the chunk with the correct `evidence` included (yes/no)?

**Formula** (intuitive):

```
Recall@k = (# of queries where evidence appeared in top-k) / (total queries)
```

**Concrete example with FinanceBench**:

```
150 questions evaluated with k=5
→ The system hit (placed the correct evidence in its top-5) in 120 cases
→ Recall@5 = 120 / 150 = 0.80 = 80%
```

**Why we report multiple k** (k=1, 3, 5, 10):
- **Recall@1** measures extreme precision: did the system put it FIRST?
- **Recall@10** measures "at least it's nearby": useful for pipelines with a downstream reranker.

**Limitation**: doesn't penalize by position. If evidence is in position 1 vs position 5, Recall@5 doesn't care (both = 1). For that we use the next ones.

---

### 8.2 MRR (Mean Reciprocal Rank) — what position is it in?

**Question it answers**: how high in the ranking did the first relevant chunk appear? Strongly rewards having the correct one in position 1.

**Formula**:

```
For each query: RR = 1 / (position of the first relevant chunk)
MRR = average of RR across all queries
```

**Example**:

```
Query 1: evidence at position 1 → RR = 1/1 = 1.00
Query 2: evidence at position 3 → RR = 1/3 = 0.33
Query 3: evidence at position 2 → RR = 1/2 = 0.50
Query 4: evidence NOT in top-k → RR = 0

MRR = (1.00 + 0.33 + 0.50 + 0) / 4 = 0.46
```

**When it matters**: in systems where the user only sees top-1 or top-3 (e.g., a chatbot showing "the answer"). Each position you drop, relevance falls exponentially.

---

### 8.3 NDCG@k (Normalized Discounted Cumulative Gain) — how well did it order the list?

**Question it answers**: did the system order the chunks optimally? Penalizes by position using a logarithm (softer than MRR).

**Why "Normalized"**: the metric is divided by the "theoretical perfect ranking", so the result ranges from 0 to 1 (comparable across datasets of different sizes).

**Intuition** (without dropping the full formula):

```
DCG = sum of (relevance_chunk_i / log2(position_i + 1))
NDCG = DCG_actual / DCG_ideal
```

**Why it matters**: NDCG is the standard metric in **information retrieval** literature (Google search, recommender systems). Your Recall@k could be 80%, but if all the correct chunks land in positions 8-10, NDCG will be low → signal that the embedder "almost gets it right" but isn't precise.

---

### 8.4 MAP (Mean Average Precision) — does it handle multiple relevants?

**Question it answers**: when there are **several relevant chunks** for the same query (typical in FinanceBench: `evidence` is a list of 1-3 passages), did the system put them all near the top?

**Difference vs MRR**: MRR only looks at the first relevant. MAP looks at **all relevants** and averages.

**When it matters**: when a query has multiple valid answers. In FinanceBench, ~30% of queries have 2-3 distinct evidences — if the system only retrieves one and misses the others, MAP penalizes it, MRR doesn't.

---

### 8.5 Why we use all 4 together and not just one

| Metric | Measures |
|---|---|
| **Recall@k** | Basic coverage: did it find it? |
| **MRR** | Position of the first hit |
| **NDCG@10** | Global ranking quality |
| **MAP** | Coverage of multiple relevants |

Each one has a blind spot. **Reporting all 4 is academic standard** and protects us from cherry-picking ("look, my model has 95% Recall@10!" — yes, but MRR is 0.20 because it puts the correct ones in position 8).

**Plus**: we report all of them with **bootstrap confidence intervals** (1000 resamples), so we don't present a single number but a range — honest scientific practice.

---

## Quick glossary

| Term | One-line definition |
|---|---|
| **SEC** | US federal regulator that forces public companies to publish standardized reports |
| **Filing** | A document a public company must submit to the SEC |
| **10-K** | Annual mandatory filing — the financial bible of the company (audited) |
| **10-Q** | Quarterly mandatory filing (unaudited, less detailed) |
| **8-K** | Event-driven mandatory filing (when something material happens) |
| **Earnings release** | Voluntary communication to investors with quarterly results (NOT a formal filing) |
| **MD&A** | Management's Discussion and Analysis — the narrative section of the 10-K |
| **GICS** | Global Industry Classification Standard — 11-sector taxonomy of companies |
| **FinanceBench** | Public dataset of 150 QA pairs over real SEC filings (PatronusAI, 2023) |
| **Evidence** | The exact passage from the PDF that justifies a question's answer (ground truth) |
| **QA pair** | A (question, answer) tuple — the basic unit of an evaluation dataset |
| **Passage** | A meaningful fragment of a document (typically a few paragraphs) |
| **Chunk** | A passage produced by a specific chunking strategy — the operational unit of a RAG system |
| **Chunking strategy** | The method used to split documents into chunks (naive, semantic, contextual, late) |
| **Token** | A word or subword — the model's atomic unit (~3-4 characters on average) |
| **Recall@k** | % of queries where the correct evidence appeared in top-k retrieved chunks |
| **MRR** | Mean Reciprocal Rank — average of 1/position-of-first-hit across queries |
| **NDCG@k** | Normalized Discounted Cumulative Gain — standard IR ranking quality metric |
| **MAP** | Mean Average Precision — handles multiple relevants per query |
| **Bootstrap CI** | Confidence interval estimated by resampling (no normality assumption) |
| **min / max** | Smallest / largest value in a dataset |
| **mean** | Arithmetic average — sensitive to outliers |
| **median (p50)** | Middle value — robust to outliers |
| **percentile (p25, p75, p95)** | Value below which N% of the data falls |

---

## References

- [SEC EDGAR — public filings database](https://www.sec.gov/edgar.shtml)
- [Patronus AI — FinanceBench announcement](https://www.patronus.ai/announcements/announcing-financebench-the-first-stock-market-benchmark-for-llms)
- [FinanceBench paper (arXiv 2311.11944)](https://arxiv.org/abs/2311.11944)
- [GICS — official MSCI page](https://www.msci.com/our-solutions/indexes/gics)
- [10-K official SEC guide for investors](https://www.sec.gov/files/reada10k.pdf)
