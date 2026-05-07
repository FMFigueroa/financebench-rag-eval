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
10. [The RAG pipeline — retrieval, reranking, generation](#10-the-rag-pipeline--retrieval-reranking-generation)
11. [How we measure success: retrieval metrics](#11-how-we-measure-success-retrieval-metrics)
12. [Putting it all together — a worked example end-to-end](#12-putting-it-all-together--a-worked-example-end-to-end)
13. [Parsing PDFs — from positioned glyphs to indexable structure](#13-parsing-pdfs--from-positioned-glyphs-to-indexable-structure)

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

> ⚠️ **Important about evidence structure**: the `evidence` field is a **list of 1 to 3 passages** (not a single text block). Each item has its own `evidence_text` and `evidence_page_num`. For ~23% of the dataset, a single question is justified by multiple passages — sometimes in **different sections of the same PDF** (e.g., a number from the Income Statement + a narrative from the Risk Factors section). This compositional structure is critical for understanding length statistics (§9) and chunking implications (§8).

### How Patronus categorized the questions

Each question carries **two independent labels** in the dataset:

- **`question_type`** — origin / style of the question
  - `metrics-generated`
  - `domain-relevant`
  - `novel-generated`
- **`question_reasoning`** — type of reasoning required to answer it
  - `Information extraction`
  - `Numerical reasoning`
  - `Logical reasoning (multi-step)`
  - `None` (`novel-generated` questions are unclassified)

**Independent** means that **one label does not determine the other** — neither is a difficulty hierarchy of the other. A `metrics-generated` question can be easy or hard; a `novel-generated` one can be trivial or complex. The difference is in their **origin**, not their inherent complexity.

**Why these specific categories?** Because the PDF of a 10-K has **3 distinct types of content**, and each `question_type` reflects one. The classification is not arbitrary — it follows the structure of the actual document.

#### `question_type` — origin / style of the question

| Category | Origin | Example | Why Patronus created it |
|---|---|---|---|
| **`metrics-generated`** | Generated by **fixed templates** over standard financial metrics (CapEx, revenue, margin, etc.) | *"What is the FY2018 capital expenditure amount in USD millions for 3M?"* — rigid pattern | Systematic coverage + benchmark reproducibility |
| **`domain-relevant`** | Written by humans with **clear focus** on 10-K aspects (risks, governance, strategy) | *"What are the main risks 3M identifies in its 2018 annual report?"* | Cover more natural questions, still structured |
| **`novel-generated`** | Human questions **without template or fixed pattern** — the most "wild" | *"How is Pfizer's R&D pipeline positioning the company in oncology?"* — free format | **Stress test**: replicates how a real user would ask questions in natural language |

##### Detailed view — what in the PDF generates each category

**`metrics-generated`** → originates in the **standardized financial statements** of the 10-K. A 10-K always contains 3 audited financial tables, each one regulated by SEC GAAP rules and structurally identical across all companies:

- **Income Statement** — revenue, costs, profitability (`Net sales`, `Cost of goods sold`, `Operating income`, etc.)
- **Balance Sheet** — assets, liabilities, equity (`Total assets`, `Long-term debt`, `Stockholders' equity`, etc.)
- **Cash Flow Statement** — cash inflows/outflows (`Capital expenditures`, `Operating cash flow`, etc.)

Patronus took a list of standard metrics from these tables and applied rigid templates of the form *"What is the {metric} for {company} in fiscal year {year}?"* — one template generates 50+ questions varying parameters.

> Real example: 3M 2018 10-K, page 59 (Statement of Cash Flows): line `Purchases of property, plant and equipment ... $(1,577)` → templated question yields the literal answer **$1,577M**.

**`domain-relevant`** → originates in the **mandatory narrative sections** of the 10-K. Beyond the audited financial tables, the 10-K has free-prose sections regulated by SEC item structure:

- **Item 1** — Business (operations, products, markets, competition)
- **Item 1A** — Risk Factors (specific risks the company identifies)
- **Item 7** — MD&A (Management's Discussion and Analysis)

*(See §3 — Anatomy of a 10-K — for the complete item structure.)*

These sections are **predictable in their existence** (every 10-K has them) but **variable in content** (each company describes its own risks). Humans wrote questions with clear focus on these sections but free format.

> Real example: 3M 2018 10-K, Item 1A → several paragraphs describing PFAS contamination, litigation, supply chain disruption → question *"What are the main risks 3M identifies?"* requires reading and synthesizing prose, no single value to extract.

**`novel-generated`** → originates in **information that crosses sections** of the 10-K. Instead of staying within one table or one item, novel questions combine signals from multiple parts of the document:

- A **metric** from the financial statements (Income Statement / Balance Sheet / Cash Flow Statement)
- A **narrative** from the qualitative items (Item 1 Business / Item 7 MD&A)
- A **risk** from Item 1A

Humans wrote these questions completely open, simulating how a real analyst explores a 10-K without a predefined agenda. Their unpredictability makes them the **stress test** of the benchmark.

> Real example: Pfizer 2022 10-K → question *"How is Pfizer's R&D pipeline positioning the company in oncology?"* requires crossing 3 sections (R&D spend in Income Statement + pipeline narrative in Item 1 Business + competitive risks in Item 1A).

**Why this matters for evaluation**: a RAG system can do well on `metrics-generated` (predictable templates) but **fail miserably on `novel-generated`** — that would reveal the system overfits to prompt style instead of understanding semantics. The novel category forces true generalization.

#### `question_reasoning` — type of reasoning required

| Type | What it requires | Example |
|---|---|---|
| **Information extraction** | Direct lookup of the value in the text | *"What was 3M's revenue in 2018?"* → read the figure |
| **Numerical reasoning** | Extract numbers + compute | *"What was the YoY growth rate?"* → compute `(new - old) / old` |
| **Logical reasoning (multi-step)** | Combine inferences and comparisons | *"Did CapEx grow faster than revenue?"* → 4 numbers → 2 ratios → comparison |

##### Detailed view — where the answer lives in the PDF

**`Information extraction`** → the answer is **literal in a table cell or a direct prose mention**. The system just needs to find the correct chunk and return the text. No math, no inference.
> Real example: 3M 2018 Income Statement → line `Net sales ... $32,765` → question *"What was 3M's revenue in FY2018?"* → answer **$32,765M** is literal in the PDF.

**`Numerical reasoning`** → the answer is **NOT literal** in the PDF. The document contains the inputs (2+ values), but the result has to be computed by the downstream LLM after retrieval.
> Real example: 3M 2017 Net sales = $31,657M, 2018 Net sales = $32,765M → question *"What was the YoY growth rate?"* → answer is **NOT in the PDF**, must compute `(32,765 - 31,657) / 31,657 = 3.5%`. The RAG system must retrieve a chunk containing both numbers (ideally the full year-over-year comparison table).

**`Logical reasoning (multi-step)`** → the answer requires combining multiple pieces of information, possibly from **different sections of the PDF**. Extract data, compute intermediate ratios, compare results, emit a judgment.
> Real example: question *"Did 3M's CapEx grow faster than its revenue between 2017 and 2018?"* → inputs in the PDF are 4 numbers from 2 different sections: Income Statement (Revenue 2017 + 2018) and Cash Flow Statement (CapEx 2017 + 2018). Process: extract → compute 2 growth ratios → compare → emit yes/no. The RAG system **must retrieve chunks from 2 different sections** of the 10-K. This is why ~23% of the dataset has 2-3 distinct evidences (see §6 dataset analysis).

> **Caveat about `None`**: ~33% of questions (all `novel-generated`) have `question_reasoning = None`. That means **Patronus did not classify them**, NOT that they require more processing. A novel question can be simple extraction or complex numerical — Patronus left them free-form because their diversity makes them hard to bucket into fixed categories.

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

#### FinanceBench specifics — the compositional structure of `evidence`

The `evidence` field is **NOT a single text block** — it's a **list of 1 to 3 items**, where each item is a distinct passage with its own metadata:

```python
evidence: [
  {
    "evidence_text": "...",              # the passage text
    "evidence_text_full_page": "...",    # the entire PDF page where the passage lives
    "evidence_page_num": 45,             # which page of the PDF
    "doc_name": "3M_2018_10K"            # which document
  },
  # ... up to 3 items per question
]
```

**Key implications**:

- **76.7% of QAs have 1 passage** (single-evidence). The simplest case for retrieval.
- **20.7% have 2 passages** and **2.7% have 3 passages**. ~23% of the dataset is **multi-evidence**.
- When multi-evidence happens, **the passages can live in different sections of the same PDF** — for example, a number from the Income Statement (page 45) and a narrative from the Risk Factors section (page 12). The RAG system has to retrieve **both** to answer correctly.
- In our analysis (Sub-block 2.4 of the notebook), we measured: when there are 2+ passages, **91% are within ≤5 pages of each other** (same section). Only ~9% are truly cross-section.

This compositional structure is the reason why MAP exists as a metric: when there are multiple correct chunks, MRR only credits the first one, but MAP rewards retrieving them all (see §11).

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

The numbers below report **the total evidence length per question**, in characters. Important nuance: as explained in §6 and §8, each FinanceBench question has 1 to 3 passages (not a single text block). When a question has multiple passages — possibly from **different parts of the same PDF** (e.g., Income Statement page 45 + Risk Factors page 12) — we **sum the chars of all passages** to compute the "total evidence load" the RAG system would have to retrieve.

```
evidence (TOTAL length in characters, summed across all passages of each QA):
  median  = 1,450 chars
  mean    = 1,712 chars
  p95     = 4,194 chars
  max     = 12,123 chars
```

**Technical reading of each number**:

- **median (1,450 chars)** → half of the evidences are ≤ 1,450 chars. This is the **typical case**.
- **mean (1,712 chars) > median** → asymmetric distribution (see "long tail" below).
- **p95 (4,194 chars)** → 95% of the dataset falls under 4,194 chars. Only the largest 5% exceeds that threshold.
- **max (12,123 chars)** → the longest evidence in the dataset. **Almost 3× the p95** — an extreme outlier.

#### What does "long tail" mean?

When `mean (1,712) > median (1,450)`, the distribution is NOT symmetric. Most evidences are short or medium, but **a few extreme cases pull the average up**. Visualized:

```
Frequency (how many evidences fall in each range)
  │
  │  ▓▓▓▓▓▓▓
  │  ▓▓▓▓▓▓▓▓▓▓
  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓
  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░ ........... ... ..    .       .
  └──────────────────────────────────────────────────────────────────
     0     1k    2k    3k    4k    5k   ...   8k   ...   10k  ...  12k  chars
        ↑ median ↑ mean         ↑ p95                              ↑ max
        (1,450) (1,712)        (4,194)                          (12,123)
```

The "tail" is the part on the right that extends out: few cases but VERY large. Technically called a **right-skewed distribution**.

#### The operational WHY: why we measure this

This isn't an academic metric — it **drives critical technical decisions** in the RAG system. The reasoning chain:

```
1. Evidence length (chars)         ← what we just measured
   ↓ (rule of thumb: ~4 chars ≈ 1 token in English)
2. Estimated tokens                ← how many tokens the evidence would have if embedded
   ↓
3. Compare against the embedder's max_tokens (typically 512)
   ↓
4. Does it fit in 1 chunk?
     ├─ YES → you can retrieve with k=1, metrics will be optimistic
     └─ NO  → you need chunking + retrieval with k>1
   ↓
5. Chunking strategy + appropriate k value
```

Grounded in our numbers:

| Percentile | Chars | Tokens (~÷4) | Fits in 512-token chunk? |
|---|---:|---:|---|
| Median | 1,450 | ~362 | ✅ Plenty of room |
| p75 | 2,267 | ~566 | ⚠️ Just barely overflows → needs 2 chunks |
| p95 | 4,194 | ~1,048 | ❌ Needs 2-3 chunks |
| Max | 12,123 | ~3,030 | 🚨 Needs 6+ chunks |

**Technical conclusion**: for ~25-30% of the dataset, the full evidence does **NOT fit in 1 chunk**. That's why this project:

- **Reports Recall@k for several k values** (k=1, 3, 5, 10) — covers from "perfect retrieval at top-1" to "evidence somewhere in top-10"
- **Reports MAP** — rewards retrieving multiple chunks when the evidence spans across them
- **Compares 4 chunking strategies** — none wins at all percentiles

> 💡 **If we only measured the mean (1,712 chars)**, we'd lose the long tail. We'd design a system assuming the typical evidence is ~1,712 chars and the outliers would surprise us. That's why **always median + percentiles**, never mean alone.

### Golden rule

When the distribution has a long tail (typical for real-world data: salaries, file sizes, text lengths, retrieval scores), **always report median + percentiles**, not just mean. Honest journalism uses "median salary", not "average salary", for exactly this reason.

This is also why we report **bootstrap confidence intervals** for our retrieval metrics in later stages — instead of a single number for "Recall@5 = 0.80", we report something like "Recall@5 = 0.80 [0.74, 0.85] (95% CI)". The interval captures the uncertainty in our estimate, computed by resampling the evaluation set 1,000 times.

---

## 10. The RAG pipeline — retrieval, reranking, generation

A complete RAG system has **three phases**. Knowing which phase each technique belongs to is critical for understanding what this project measures and what it doesn't.

```
1. RETRIEVAL                      ← Stage 1-2 of this project
   └─ Find the correct PDF chunk that contains the answer
   └─ Metrics: Recall@k, MRR, NDCG, MAP (see §11)

2. RERANKING (optional)           ← Stage 2 of this project
   └─ Reorder the top-k retrieved so the most relevant lands on top
   └─ Models compared: Cohere Rerank v3.5, BGE Reranker v2

3. GENERATION                     ← Eslabón 2 (OUT of scope here)
   └─ The LLM reads the retrieved chunk and produces the natural answer
   └─ For numerical/logical questions: performs calculation or comparison
   └─ For extraction: just reads and reformulates
```

### Phase 1 — Retrieval

The vector store returns the **top-k chunks most similar** to the query. Similarity is measured by cosine distance between the query embedding and each chunk embedding (or by BM25 lexical matching, or by a hybrid of both).

**This is the phase where chunking strategy matters most**: the right chunk has to exist in the index AND be ranked among the top-k. The 4 chunking strategies compared in this project (see §8) are all evaluated at this stage.

### Phase 2 — Reranking (optional)

Retrieval is fast but coarse. Reranking takes the top-k from phase 1 (typically k=20-50) and **re-scores them with a more powerful model** that examines query-chunk pairs jointly (cross-encoder), placing the truly relevant ones at the top of a smaller list (e.g., top-5).

| Reranker | Type | Notes |
|---|---|---|
| **Cohere Rerank v3.5** | Commercial API | Strong out-of-the-box, no fine-tuning needed |
| **BGE Reranker v2** | Open-source, self-hosted | Comparable quality, no per-call cost |

**MMR (Maximal Marginal Relevance)** is a separate technique that promotes **diversity** in the top-k — useful when the relevant chunks are scattered across different sections of the document. In FinanceBench, ~91% of multi-evidences are within 5 pages of each other, so MMR provides little benefit and we don't use it for the baseline.

### Phase 3 — Generation (out of scope)

Once retrieval+reranking has selected the best chunk(s), the LLM (Claude, GPT, etc.) reads them and produces the final natural-language answer.

- For **information extraction** questions, the LLM just reformulates what it sees.
- For **numerical reasoning** questions, the LLM has to **perform the calculation** itself.
- For **logical reasoning multi-step**, the LLM has to **combine multiple pieces of information** and reason over them.

**This project does NOT measure phase 3**. We assume the LLM is good enough and focus on whether **the chunk it receives contains the necessary information**. If retrieval fails, generation can't recover. If retrieval succeeds, generation might still fail — but that's a separate problem, addressed in **Eslabón 2** of the roadmap (full RAG + generation evaluation end-to-end).

### What this means for the metrics

The retrieval metrics in §11 (Recall@k, MRR, NDCG, MAP) measure **only phase 1+2 quality**. A model with `Recall@5 = 0.95` means: in 95% of cases, the correct chunk is among the top-5 returned. Whether the LLM downstream uses it correctly to produce a good answer is a separate question, not captured by these numbers.

---

## 11. How we measure success: retrieval metrics

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

## 12. Putting it all together — a worked example end-to-end

> **Goal of this section**: take ONE real question from FinanceBench and trace it through every step of the pipeline — PDF → chunks → embeddings → query → retrieval → reranking → metrics → generation. After reading this you should be able to say *"now I see how all the pieces from §1-§11 actually move together"*.

§1-§11 are like a parts catalog: each component (filings, evidence, chunks, embedders, metrics) is documented in isolation. This section is the **assembly manual** — same parts, but in motion, on one concrete query.

### 12.0 The case study

We pick a single question from FinanceBench's 150 and follow it from start to finish.

```
company:              3M
doc_name:             3M_2018_10K
question:             "What was 3M's revenue YoY growth from 2017 to 2018?"
question_type:        metrics-generated      ← origin label (§6)
question_reasoning:   Numerical reasoning    ← reasoning label (§6)
ground-truth answer:  3.5%
evidence (1 item):    "Net sales ............ $32,765 (2018) ... $31,657 (2017)"
                      Item 8 — Income Statement
```

Why this question? It hits the sweet spot for a teaching example:

- **`Numerical reasoning`** → forces the distinction between *retrieval* (find the chunk with the numbers) and *generation* (do the arithmetic). This is the line we draw at the boundary of Stage 1.
- **`metrics-generated`** → the answer lives in a structured table, so chunking strategy is testable.
- **Single evidence** → simpler to follow Recall, MRR, NDCG, MAP without multi-evidence noise. We address multi-evidence at the end (§12.8).

> ⚠️ The question text and `Net sales` figures come from 3M's real 2018 10-K Income Statement. The embedding values, cosine similarities, and chunk IDs shown in §12.2 onward are **illustrative of the typical behavior** — they reflect what a competent embedder produces on a question like this, but are not measured. The point is the *shape* of the flow, not the exact decimals.

---

### 12.1 What the system actually receives

Critical distinction before we start: the system at query time only sees **the question string**. Everything else in the JSON record (tags, evidence, page numbers, ground-truth answer) is **dataset metadata** that the evaluator uses to *score* the system afterwards.

```
INPUT to the RAG system           METADATA (used by the evaluator only)
─────────────────────────         ──────────────────────────────────────
question (string)                 financebench_id
                                  doc_name
                                  question_type
                                  question_reasoning
                                  evidence (list of passages)
                                  evidence_page_num
                                  answer (ground truth)
                                  justification
```

> 💡 Why this distinction matters: confusing "what the system sees" with "what we use to grade it" is the most common mistake when reading retrieval papers. The evidence is **never an input** — it's the answer key.

---

### 12.2 Indexing — turning the PDF corpus into a searchable vector space

Indexing happens **once, offline**, before any query is asked. It has 3 steps.

#### 12.2.1 PDF parsing

The 3M 2018 10-K is a ~150-page PDF with prose, tables, footnotes, and exhibits. We run it through `pdfplumber` (chosen over `pypdf` because it preserves table structure better — see CLAUDE.md):

```python
import pdfplumber

with pdfplumber.open("3M_2018_10K.pdf") as pdf:
    pages = [p.extract_text() for p in pdf.pages]
full_text = "\n".join(pages)
```

Output: ~300,000 tokens of cleaned text.

> ⚠️ Gotcha: `pdfplumber` does NOT magically reconstruct tables as Markdown. The Income Statement comes out as raw text with column alignment loosely preserved. For naive chunking that's fine; for richer strategies (semantic, contextual) you'd post-process tables separately. We log this trade-off in `docs/decisions/`.

#### 12.2.2 Chunking

We split the 300K-token text into overlapping chunks. For this walkthrough we use **naive fixed-size chunking** (the baseline strategy from §8) — 512 tokens per chunk, 50 tokens overlap.

Result: **~640 chunks** for this single document. A few representative ones:

```
chunk_id   content (truncated)                                        section
─────────  ──────────────────────────────────────────────────────     ──────────────
chunk_007  "...as a diversified technology company, 3M operates..."   Item 1 — Business
chunk_142  "...risks related to the discontinuation of LIBOR..."      Item 1A — Risks
chunk_287  "...non-GAAP measures discussed in this MD&A..."           Item 7 — MD&A
chunk_312  "Net sales              $32,765   $31,657   $30,109        Item 8 — Income
            Cost of sales           16,682    16,001    15,041        Statement
            Gross profit            16,083    15,656    15,068..."
chunk_487  "...Capital expenditures           (1,577)   (1,373)..."   Cash Flow Statement
```

The chunk we *want* the system to find for our question is **`chunk_312`**: it contains both `$32,765` (2018) and `$31,657` (2017) in the same window — exactly the evidence Patronus tagged.

> 💡 Why chunking strategy matters: naive 512-token chunking *happened* to keep both years in the same window because the Income Statement is compact. If our chunker had split mid-table, the answer would be **fragmented across two chunks** and Recall@1 would be 0 even with a perfect embedder. Quantifying this kind of failure is exactly what the "4 chunking strategies" study (§8) is for.

#### 12.2.3 Embedding

Each of the ~640 chunks is passed through an embedder. For this example, OpenAI's `text-embedding-3-large`, which produces a **3072-dimensional vector** per chunk:

```
chunk_312 → [0.0142, -0.0287, 0.0913, ..., 0.0058]   ← 3072 floats
```

These vectors are stored in **Qdrant** (our vector store, see CLAUDE.md), each one paired with its original text and metadata (`chunk_id`, `page`, `doc_name`, etc.).

After indexing the entire corpus, we have a **searchable index of ~640 vectors** for 3M's 10-K. (Across all 150 questions in FinanceBench, the full corpus has ~360 documents and ~230K chunks indexed.)

> 🧠 Mental model: think of the vector space as a **3072-dimensional warehouse** where every chunk has a coordinate. Chunks about "revenue" cluster in one neighborhood, chunks about "litigation risk" in another, chunks about "executive compensation" in a third. The embedder's job is to put semantically similar texts physically close in that space.

---

### 12.3 Query time — how the system answers ONE question

#### 12.3.1 Embedding the query

When the user (or the eval harness) submits the question, **the same embedder** is applied to the query string:

```
"What was 3M's revenue YoY growth from 2017 to 2018?"
                         ↓ text-embedding-3-large
[0.0198, -0.0341, 0.0876, ..., 0.0073]   ← also 3072-d
```

Now the query is a point in the **same warehouse** as the chunks.

#### 12.3.2 Similarity scoring

For every one of the ~640 chunks, the system computes the **cosine similarity** between the query vector and the chunk vector:

```
cos_sim(q, c) = (q · c) / (||q|| × ||c||)   →  value in [-1, 1]
```

Higher = more semantically similar. (Both vectors are L2-normalized at index time, so this reduces to a plain dot product — fast on GPU.) For our query, the resulting scores might look like:

```
chunk_id    cosine_sim   section                  notes
─────────   ──────────   ───────────────────────  ─────────────────────────────
chunk_312     0.847      Income Statement         ← contains BOTH years (ground truth!)
chunk_287     0.812      MD&A                     discusses revenue trends, no exact #s
chunk_410     0.794      Segment results          segment-level revenue 2018
chunk_315     0.781      Comprehensive income     related but not exact
chunk_007     0.776      Business overview        mentions 2018 sales generically
chunk_487     0.512      Cash Flow Statement      CapEx (numerical but unrelated)
chunk_142     0.234      Risk Factors             LIBOR risk (irrelevant)
...
```

#### 12.3.3 Top-k retrieval

We keep the top-5 chunks ranked by cosine similarity:

```
RANK   CHUNK_ID      SIM      IS_GROUND_TRUTH?
────   ──────────    ─────    ─────────────────
 1     chunk_312     0.847    ✅ YES  ← evidence in position 1
 2     chunk_287     0.812    ❌
 3     chunk_410     0.794    ❌
 4     chunk_315     0.781    ❌
 5     chunk_007     0.776    ❌
```

This is what **Phase 1 (Retrieval)** of §10 produces. The pipeline could stop here — but for a stronger system, we add **Phase 2**.

---

### 12.4 Reranking — refining the top-k with a cross-encoder

The top-5 above was ranked by **bi-encoder cosine** (cheap: query and chunks were embedded *independently*). A reranker like **Cohere Rerank v3.5** does something more expensive: it feeds **(query, chunk)** as a single joint input to a transformer that produces a relevance score, capturing fine-grained interaction between the two texts.

For our example, reranking the top-5 might produce:

```
Before rerank (cosine)              After rerank (cross-encoder)
──────────────────────────          ──────────────────────────────
 1. chunk_312  0.847  ✅             1. chunk_312  0.991  ✅  (kept #1)
 2. chunk_287  0.812                 2. chunk_410  0.823       (jumped from #3)
 3. chunk_410  0.794                 3. chunk_287  0.687       (dropped from #2)
 4. chunk_315  0.781                 4. chunk_315  0.612
 5. chunk_007  0.776                 5. chunk_007  0.354
```

The reranker recognizes that `chunk_410` (segment-level revenue breakdown) is more directly relevant to a *YoY growth* question than `chunk_287` (general MD&A prose) — a subtlety the bi-encoder cosine couldn't capture because the two texts were never seen together.

> 💡 Why we don't always rerank everything: the cross-encoder is **~50× slower per pair** than a bi-encoder lookup. So the standard pipeline is "retrieve top-50 fast, rerank to top-5 carefully" — best of both worlds.

---

### 12.5 Computing the metrics for THIS single query

Now we evaluate the result. The evaluator knows the ground-truth `chunk_312` is the only relevant one for this question. Plugging into §11's formulas:

```
Position of the relevant chunk after retrieval:    rank = 1
Number of relevant chunks for this query:          1
```

#### Recall@k

```
Recall@1   = (was relevant in top-1?)   → YES → 1
Recall@3   = (was relevant in top-3?)   → YES → 1
Recall@5   = (was relevant in top-5?)   → YES → 1
Recall@10  = (was relevant in top-10?)  → YES → 1
```

All Recall values are 1 for *this* query. (The aggregate Recall@5 across all 150 queries is what gets reported in the master table.)

#### MRR — Reciprocal Rank for this single query

```
RR = 1 / position_of_first_relevant = 1 / 1 = 1.00
```

A perfect 1.00 because the relevant chunk landed in position 1.

#### NDCG@5

With one relevant chunk in position 1:

```
DCG   = relevance_1 / log2(1 + 1) = 1 / log2(2) = 1.00
IDCG  = 1 / log2(1 + 1)                         = 1.00
NDCG  = DCG / IDCG                              = 1.00
```

Also a perfect 1.00.

#### MAP — Average Precision for this query

```
At the rank where the relevant lands (rank=1):
  precision@1 = 1 relevant retrieved / 1 retrieved = 1.0
AP = average of precisions at the ranks where relevants are found
   = 1.0 / 1 relevant = 1.00
```

Also 1.00.

> 🎯 Big idea: **for a single query, the metrics are just numbers — they only become meaningful when averaged over many queries**. Our query scores 1.00 across the board; another query where the relevant lands at rank 7 might score Recall@5 = 0, MRR = 0.143, NDCG@5 = 0. Reporting on the dataset means averaging the 150 per-query scores. This is what §11.5 means by "we use all 4 together" — each query gets 4 scores, and we aggregate over the dataset.

---

### 12.6 Generation — what the LLM does with the retrieved chunk (out of scope but instructive)

This is **Phase 3** of §10, formally Eslabón 2 of the roadmap, but worth seeing once to understand why retrieval metrics matter so much.

The top-1 chunk's text is concatenated with the question into a prompt for an LLM:

```
SYSTEM: You are a financial analyst. Answer using only the context below.

CONTEXT:
Net sales              $32,765   $31,657   $30,109
Cost of sales           16,682    16,001    15,041
Gross profit            16,083    15,656    15,068
[... rest of chunk_312 ...]

QUESTION: What was 3M's revenue YoY growth from 2017 to 2018?

ANSWER:
```

A capable LLM (Claude, GPT-4, etc.) reads the table, identifies that the columns are labeled by year (or asks for clarification if ambiguous), and computes:

```
Growth = (32,765 - 31,657) / 31,657 = 0.035 = 3.5%
```

The LLM emits: *"3M's revenue grew 3.5% from $31,657M in 2017 to $32,765M in 2018."*

> 💡 The retrieval-vs-generation handoff: notice that **if retrieval had failed** (e.g., delivered `chunk_142` about LIBOR risk instead of `chunk_312`), no LLM could recover — there are simply no numbers in that chunk to compute growth from. **Retrieval quality bounds generation quality.** That's why we measure retrieval rigorously in Stage 1 before adding generation in Eslabón 2.

---

### 12.7 From one query to 150 — what the master table actually looks like

We run §12.1 → §12.5 for **all 150 questions**, then aggregate. For one (embedder × chunking strategy) cell, the result is roughly:

```
Embedder:           text-embedding-3-large
Chunking strategy:  naive_512_overlap_50
Reranker:           none (retrieval only)
N queries:          150

Recall@1   = 0.547  [bootstrap 95% CI: 0.467, 0.627]
Recall@3   = 0.733  [bootstrap 95% CI: 0.660, 0.800]
Recall@5   = 0.807  [bootstrap 95% CI: 0.740, 0.867]
Recall@10  = 0.873  [bootstrap 95% CI: 0.813, 0.927]
MRR        = 0.658  [bootstrap 95% CI: 0.582, 0.731]
NDCG@10    = 0.706  [bootstrap 95% CI: 0.638, 0.770]
MAP        = 0.612  [bootstrap 95% CI: 0.541, 0.682]
```

(Numbers illustrative; real values land in `results/baselines/` once Stage 1 is run.)

This **single row** is the aggregation of 150 mini-versions of §12.0 → §12.5. The full master table has **5 embedders × 4 chunking strategies = 20 rows**, each computed exactly the same way. With reranking on/off it's 40 rows. That's the rigorous comparison the project promises.

> 🔑 The bootstrap CIs are critical: a difference of `0.807 vs 0.815` between two embedders is meaningless if the CIs overlap heavily. Reporting raw numbers without CIs is how teams accidentally claim improvements that are statistical noise. See §11.5.

---

### 12.8 What changes with multi-evidence queries

About **23% of FinanceBench questions** have **2 or 3 evidence passages** (e.g., a CapEx-vs-Revenue growth question requires the Income Statement *and* the Cash Flow Statement, in two different sections of the same 10-K). For those queries:

- **Retrieval** has to surface multiple correct chunks, not just one.
- **Recall@k** treats it as binary per chunk: did each relevant appear in top-k?
- **MAP** is the metric that *really* differentiates here: it averages precision at the rank of *every* relevant. A system that lands 1 of 2 relevants in top-1 but misses the second scores AP = 0.5 even though MRR = 1.0.

This is exactly why §11.5 reports all 4 metrics together: any single one is blind to a failure mode the others catch.

---

### 12.9 The complete picture — end-to-end flow at a glance

```
┌────────────────────────────────────────────────────────────────────────────┐
│ OFFLINE: INDEXING (done once)                                              │
│                                                                            │
│  3M_2018_10K.pdf  ──pdfplumber──▶  full_text (~300K tokens)                │
│                                          │                                 │
│                                    chunk into 512-token windows            │
│                                          ▼                                 │
│                                    ~640 chunks                             │
│                                          │                                 │
│                                    text-embedding-3-large                  │
│                                          ▼                                 │
│                                    ~640 vectors (3072-d)                   │
│                                          │                                 │
│                                          ▼                                 │
│                                    Qdrant index ✔                          │
└────────────────────────────────────────────────────────────────────────────┘
                                            │
─────────────────────────── one query at a time ──────────────────────────────
                                            ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ ONLINE: QUERY                                                              │
│                                                                            │
│  "What was 3M's revenue YoY growth from 2017 to 2018?"                     │
│                       │                                                    │
│                       ▼ text-embedding-3-large                             │
│             query_vector (3072-d)                                          │
│                       │                                                    │
│                       ▼ cosine vs all 640 chunk vectors                    │
│             top-5 chunks  ←  PHASE 1: Retrieval                            │
│                       │                                                    │
│                       ▼ Cohere Rerank v3.5 (cross-encoder)                 │
│             top-5 reranked  ←  PHASE 2: Reranking                          │
│                       │                                                    │
│         ┌─────────────┴──────────────┐                                     │
│         ▼                            ▼                                     │
│   EVAL (Stage 1)              GENERATION (Eslabón 2)                       │
│   compute Recall@k,           feed top-1 chunk + question                  │
│   MRR, NDCG, MAP              to Claude/GPT → "3.5%" answer                │
│   on this query                                                            │
│         │                                                                  │
│         ▼ aggregate over 150 queries                                       │
│   one row of the master table (with bootstrap 95% CIs)                     │
│                                                                            │
│   repeat for 5 embedders × 4 chunking strategies                           │
│                       │                                                    │
│                       ▼                                                    │
│           20-row master table — the deliverable of Stage 2                 │
└────────────────────────────────────────────────────────────────────────────┘
```

If this diagram clicks — *that's* the moment §1-§11 stop being a list of techniques and become **one system**. Every box above maps back to a concept defined earlier:

| Box in the diagram | Where it's defined |
|---|---|
| "PDF" | §3 — Filings |
| "chunks" | §8 — Passages and chunks |
| "embedder", "vector space" | §12.2.3 |
| "Qdrant index" | CLAUDE.md (stack) |
| "cosine + top-k" | §10 — Phase 1 |
| "rerank" | §10 — Phase 2 |
| "Recall / MRR / NDCG / MAP" | §11 |
| "150 queries" | §6 — FinanceBench |
| "bootstrap CIs" | §11.5 |
| "generation" | §10 — Phase 3 (Eslabón 2) |

That's the whole project in one picture. Welcome to RAG engineering.

---

## 13. Parsing PDFs — from positioned glyphs to indexable structure

> **Why this section exists**: §1-§12 build the conceptual framework — what a filing is, what evidence looks like, how RAG operates, how we measure it. But there's a gap between *concept* and *execution* that none of those sections close: HOW do we actually turn 84 binary PDFs into a structured corpus the embedder can ingest? This section is the theoretical anchor for `notebooks/02_financebench_exploration.ipynb` §3 and for `scripts/parse_pdfs.py`. Read this before running the notebook cells, and the demos will land with their full pedagogical weight.

### 13.1 Why a PDF is hostile to ML — the PostScript inheritance

PDF (Portable Document Format, Adobe 1993) is a direct descendant of **PostScript**, the page-description language Adobe shipped in 1985 to drive laser printers. PostScript's job was *"tell the printer where to put ink on the page"*. PDF inherited that DNA verbatim — it describes how to *render* a page, not how to *represent* its content.

The consequence for ML: when you see *"Net sales 32,765"* on screen, the PDF stores something closer to:

```text
glyph "N"  at (x0=72.0,  top=540.5)  font=BCDEEE+ArialMT  size=10
glyph "e"  at (x0=78.4,  top=540.5)  font=BCDEEE+ArialMT  size=10
glyph "t"  at (x0=84.1,  top=540.5)  font=BCDEEE+ArialMT  size=10
glyph " "  at (x0=88.6,  top=540.5)  font=BCDEEE+ArialMT  size=10
glyph "s"  at (x0=92.0,  top=540.5)  font=BCDEEE+ArialMT  size=10
...
glyph "3"  at (x0=420.0, top=540.5)  font=BCDEEE+ArialMT  size=10
glyph "2"  at (x0=425.4, top=540.5)  font=BCDEEE+ArialMT  size=10
glyph "," at (x0=430.8, top=540.5)  font=BCDEEE+ArialMT  size=10
...
```

There is **no native concept** of "paragraph", "table", "heading", "section", "row", or "cell". Just rectangles and glyphs. Any structure a human reader perceives — a heading vs. a paragraph, a table vs. a list — exists nowhere in the file; it has to be **reconstructed by the parser** from coordinate proximity and font-size patterns.

This makes parsing a PDF fundamentally different from parsing JSON or HTML, where structure is explicit. With PDF you're doing **layout reverse-engineering**, and every parser library makes different bets about how aggressive to be:

- *Conservative* (e.g., `pypdf`): just emit characters in reading order; let the user deal with structure.
- *Heuristic* (e.g., `pdfplumber`): detect tables via vertical/horizontal alignment, but accept some false positives.
- *ML-based* (e.g., `Marker`, `Docling`): train a layout-detection model on labeled documents; expensive but accurate.

> 💡 **Mental model**: parsing a PDF is closer to OCR-on-vector-graphics than to reading a text file. Treat every parser as a *guess* about layout, not a *fact* about content. Validate the guess on real pages of your corpus before trusting the output.

---

### 13.2 Head-to-head — five Python libraries evaluated

The Python ecosystem has competing PDF parsers, each making different trade-offs. We evaluated five for this project:

| Library | How it works | Tables | Speed | Setup | Verdict for 10-K |
|---|---|---|---|---|---|
| **`pypdf`** | Heuristic, pure Python | ❌ Flattens to lines | ⚡ Fast | Trivial | OK for narrative; tables masacred |
| **`pdfplumber`** | Wrapper over `pdfminer.six`; exposes coords + bboxes + heuristic table detection | ✅ Decent on bordered tables | ⚡ Fast | Trivial | **🏆 Winner** |
| **`PyMuPDF`** (fitz) | Bindings for MuPDF (Artifex C library) | ⚠️ Limited table API | 🚀 Very fast | C lib | Excellent for text; AGPL friction |
| **`tabula-py`** | Wrapper around `tabula-java` | ✅ Good for tables | 🐌 Slower | JRE required | Setup friction; tables-only |
| **`Marker`** / **`Docling`** | Layout detection via PyTorch models | 🏆 Best quality | 🐌 10-50ms/page | PyTorch + models | Overkill for Stage 1; revisit if metrics demand |

Why `pdfplumber` won, in three points anchored to **this corpus**:

1. **10-K financial statements come with visible borders and ruling lines** (Income Statement, Balance Sheet, Cash Flows). This is precisely the case where heuristic table detection works — you don't need a layout-detection ML model when the layout is already explicit in the rendering primitives.
2. **It exposes raw coordinates** (`page.chars`, `page.rects`, `page.lines`). When the heuristic fails on an edge case (merged cells, unusual notes), we can drop to low-level access and write custom logic. `pypdf` doesn't allow that — once it returns the flattened string, the structure is gone.
3. **Zero setup friction**: pure Python, MIT license, no JRE, no GPU, no model weights to download. 84 PDFs × ~140 pages averaged ~16 seconds each on a single thread — full corpus parsed in ~26 minutes total. ML-based parsers would push that to hours without GPU.

> ⚠️ **What about the AGPL on PyMuPDF?** AGPL requires that any service exposing PyMuPDF over the network release its full server source under AGPL. For a research repo this is acceptable, but if you ever wrap the parser in a SaaS product, AGPL forces decisions you may not want. `pdfplumber`'s MIT license sidesteps that risk entirely.

---

### 13.3 The default-settings gotcha — why ruling-lines fail on 10-K

This is the most common source of silent corruption when parsing financial filings, and the live demo in `notebooks/02_financebench_exploration.ipynb` §3.1 surfaces it directly. Worth understanding the underlying mechanic.

**`pdfplumber.extract_tables()` has two strategies for finding rows and columns**:

| Strategy | How it detects boundaries |
|---|---|
| `lines` (DEFAULT) | Looks for **drawn line segments** in the PDF (vector primitives) |
| `text` | Clusters glyphs into rows/columns by **x/y proximity** |

The default uses `lines` because it's the most reliable when tables have clean borders (the typical case in PDFs generated from Word/LaTeX). But in **10-K filings**, the layout of a Consolidated Statement of Income looks like this:

```
─────────────────────────────────────
Net sales       $ 32,765  $ 31,657  $ 30,109
─────────────────────────────────────
Cost of sales     16,682    16,055    15,118
─────────────────────────────────────
Operating income   7,207     6,920     6,494
─────────────────────────────────────
```

Every row has its OWN horizontal separator line. The `lines` heuristic sees those line segments and concludes that **each row is its own one-row table**. Result: a single Income Statement of 33 logical rows gets fragmented into ~12 sub-tables. Each fragment is a one-row matrix that, taken in isolation, has no header context — exactly the failure we're trying to avoid.

**The fix is one parameter change**:

```python
settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}
tables = page.extract_tables(settings)
```

With `text` strategy, the heuristic ignores the ruling lines and instead clusters glyphs by spatial proximity. The 33 logical rows become a single 33-row × 10-col matrix, header preserved at row 0. This becomes the **project-default `table_settings`**, applied uniformly across all 84 PDFs in `scripts/parse_pdfs.py`.

> 🧠 **Pedagogical lesson**: a parser's "default" reflects the assumptions of the *typical* document the maintainers had in mind. Whenever you adopt a parsing library, **explicitly verify those assumptions against your corpus** on day 1. The cost of finding this gotcha after embedding all 84 PDFs and seeing weird retrieval results would have been a full re-parse (~26 minutes), full re-embed (~$2.50), and hours of debugging. The cost of finding it on page 56 of one PDF in the notebook is zero. Spend the day-1 investment.

---

### 13.4 Three table-chunking techniques — the math behind the trade-off

Even with `pdfplumber` returning a clean 2D matrix, a chunking-layer problem remains: **what do we do when the table exceeds the embedder's context window?** Recall from §2.3 of the notebook that ~25-30% of FinanceBench evidence exceeds 512 tokens — and almost all of those large evidences are **tables**.

Three standard techniques exist. Each makes a different bet about which information to preserve:

#### a) Whole-table-as-one-chunk

Bump the chunk size *for that table* to fit it in one shot. Dense embedders that accept long contexts — BGE-M3 (8192 tokens), OpenAI `text-embedding-3-large` (8191), Voyage `voyage-finance-2` (16k) — make this feasible.

- **Preserves**: full table semantics, header ↔ data association across all rows, contextual prose surrounding the table.
- **Loses**: nothing structural.
- **Trade-off**: chunk sizes become bimodal (most prose chunks ~512 tokens, some table chunks ~3000+). This breaks the assumption of uniform chunk length that some retrieval algorithms make. Also: per-token API pricing means a 3000-token chunk costs 6× more to embed than a 512-token chunk.

#### b) Header-repetition

Split the table by rows but **repeat the header row in every chunk**.

- **Preserves**: row ↔ column association at every chunk boundary (because the header travels with each chunk).
- **Loses**: subtle inter-row context (e.g., a footnote that referred to "the prior row" is now stranded).
- **Trade-off**: simple, predictable chunk sizes, modest token duplication (~10-20% overhead from repeated headers).

The math: if the table has H rows of header + N rows of data, and we want chunks of M data rows each:

```
chunks_produced = ceil(N / M)
total_tokens    = (H + N) * tokens_per_row + (chunks_produced - 1) * H * tokens_per_row
                = (H + N) * tokens_per_row × (1 + (chunks_produced - 1) * H / (H + N))
```

For the 3M Income Statement (H=1, N=32, M=10): 4 chunks, ~10% token overhead from header duplication. Acceptable.

#### c) Row-as-sentence (linearization)

Convert each row to natural prose: *"In 2018 Net sales were $32,765M; in 2017 they were $31,657M; in 2016 they were $30,109M."*

- **Preserves**: semantic content, dense embedders read it naturally as English.
- **Loses**: numerical precision if the conversion fails on edge cases (scientific notation, parenthesized negatives, footnote markers).
- **Trade-off**: best embedding quality on **narrative-style queries** (the embedder's strong suit), worst on **lookup-style queries** (where exact numerical match matters). Also: linearization is itself a parser — bugs there propagate silently.

#### Why the baseline picks header-repetition (b)

For Stage 1 (baseline naive 512 + overlap 50), header-repetition is the cleanest contrast against the other 3 chunking strategies (semantic, contextual, late-chunking). It's also the easiest to get right — ~20 lines of code, no risk of conversion bugs. We reserve technique (a) for the **late-chunking experiment** (Stage 2) which specifically bets on long-context embedders. We reserve technique (c) for an **optional Stage 3 experiment** if numerical-reasoning queries (~29% of the dataset, see §6 reasoning types) systematically fail retrieval.

> ⚠️ **None of these techniques is universally correct.** The choice depends on (i) the embedder's context window, (ii) the query distribution, (iii) the cost-per-embedding budget. Documenting the choice and its rationale in `docs/chunking_decisions.md` (sub-block 6) is the audit trail that makes the project defensible.

---

### 13.5 Quantitative grounding — the corpus in numbers

After running `scripts/parse_pdfs.py` over the 84 PDFs, we can put hard numbers on the corpus. These are not estimates from a sample — they're measured over all 11,853 parsed pages:

| Quantity | Value | Note |
|---|---|---|
| **PDFs processed** | 84 | All open-source FinanceBench docs |
| **Pages parsed** | 11,853 | Average 141 pages/PDF |
| **JSONL on disk** | 107.5 MB | One file per PDF, gzip-compressible to ~30 MB |
| **Total text chars** | 39.9 M | Narrative + flattened table content from `extract_text()` |
| **Total table cell chars** | 37.8 M | Sum of `len(cell)` across all tables × all pages |
| **Estimated tokens** | ~19.4 M | Using the standard ~4 chars/token ratio for English |
| **Estimated chunks (baseline)** | ~63,000 | At 512 tokens with overlap 50; pre-filtering false-positive tables |

#### From token count to dollar cost

Embedding the corpus once with OpenAI `text-embedding-3-large` (price: $0.13 per 1M tokens):

```
19.4 M tokens × $0.13 / 1M = $2.52 per embedder pass
```

Across the full experiment (5 embedders × 4 chunking strategies, with caching of identical chunks across strategies):

```
Worst case (no caching):     20 × $2.52 = $50.40
Realistic (50% chunk reuse):  ~$25-30
```

Both numbers fit comfortably inside the project's $40-50 USD budget. **This is the data point that converts the experimental plan from "feasible-on-paper" to "feasible-with-evidence"** — the kind of number a reviewer at a Senior AI Engineer interview will explicitly ask about.

> 💡 **Pattern to internalize**: every architecture decision in a paper-quality project should ladder back to a measurable cost (time, money, error budget). "We chose X because Y" is weaker than "We chose X because Y, and on this corpus Y means Z dollars / Z minutes / Z accuracy points." The parsed corpus gives us Z.

---

### 13.6 Architectural payoff — the parser/chunker boundary

The deepest design lesson of this sub-block is **architectural**, not specific to PDFs or `pdfplumber`. It's about *separation of concerns* between two layers that are tempting to merge:

```
┌──────────────────────────────────┐
│  Layer 1: PARSER                 │
│  Input:  binary PDF              │
│  Output: structured JSONL        │
│          {doc_name, page_num,    │
│           text, tables}          │
│  Concern: faithful extraction    │
│           of raw structure       │
└──────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  Layer 2: CHUNKER                │
│  Input:  structured JSONL        │
│  Output: list of embeddable      │
│          chunks with metadata    │
│  Concern: how to cut and present │
│           structure for the      │
│           embedder               │
└──────────────────────────────────┘
              │
              ▼
   [Embedder layer — separate again]
```

**Why the boundary matters**:

- **Compare 4 chunking strategies on the same parsed output**. If parser and chunker were fused, we'd re-parse the 84 PDFs 4 times (~104 min total). Separated, we parse once (~26 min) and chunk 4 times in seconds.
- **Swap the parser without touching the chunker**. If `Marker` becomes the right call in Stage 2-3 (better merged-cell handling, e.g.), only Layer 1 changes — the chunking strategies, the embedders, and the metrics all remain untouched.
- **Audit each layer in isolation**. Failures in retrieval can be diagnosed by checking parser output (matrix correct?) before chunker output (chunks readable?). Without separation, every failure is a mystery.

This pattern generalizes far beyond financial RAG. Any project that ingests heterogeneous documents (legal contracts, scientific papers, medical records, code repositories) benefits from the same boundary. **The boundary is the architectural payoff that makes the project portable** to other domains.

> 🎯 **Carry-forward principle**: when you build a pipeline that has a "structured representation" stage, persist that representation to disk explicitly (JSONL, Parquet, Arrow). The downstream stages then become re-runnable, comparable, and debuggable in ways a fused pipeline never can be.

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
| **`question_type`** | FinanceBench label for question origin: `metrics-generated` / `domain-relevant` / `novel-generated` |
| **`question_reasoning`** | FinanceBench label for required reasoning: extraction / numerical / logical (or `None`) |
| **RAG pipeline** | The 3 phases of a Retrieval-Augmented Generation system: retrieval → reranking → generation |
| **Reranking** | Phase 2 of RAG: re-score top-k retrieved chunks with a cross-encoder for finer relevance ordering |
| **MMR** | Maximal Marginal Relevance — diversity-aware ranking that avoids redundant top-k results |
| **Information extraction** | Reasoning type: direct lookup of a value in the text (no calculation needed) |
| **Numerical reasoning** | Reasoning type: extract numbers and perform arithmetic |
| **Logical reasoning** | Reasoning type: multi-step inference combining multiple pieces of information |
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
