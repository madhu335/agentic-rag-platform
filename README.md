# Agentic RAG Platform (Hybrid Search + AI Workflows)

A production-grade agentic AI platform built with Spring Boot, pgvector, and Ollama, supporting hybrid retrieval (BM25 + vector + RRF), semantic chunking, structured domain ingestion, evaluation with failure diagnosis, and full observability.

---

## Overview

This platform enables:

- Agentic workflows (Planner → Executor → Tools)
- Hybrid RAG (BM25 + vector + RRF + re-ranking)
- Multi-domain ingestion: PDF, vehicle specs, CMS articles
- Semantic chunking by question category — not token count
- Many-to-many vehicle-to-article relationships
- Evaluation pipeline with precision/recall and failure diagnosis
- Observability via OpenTelemetry and LangSmith

---

## Architecture

### Core execution flow

```
User query
  → Planner (structured JSON plan)
  → Executor (step-by-step dispatch)
  → Tools (vehicle, research, email, SMS)
  → State store (session tracking)
  → Evaluation / re-plan
  → Response
```

### Retrieval pipeline

```
User query
  → Embed (Ollama nomic-embed-text)
  → BM25 keyword search
  → Vector search (pgvector cosine <=>)
  → RRF fusion
  → Re-ranking (cosine second pass)
  → Compress / answer (Ollama llama3)
```

---

## Domains

### 1. PDF documents

Semantic chunking via sliding window. Used for interview Q&A, technical guides, and general document RAG.

**Endpoint:** `POST /api/pdf/upload`

---

### 2. Vehicle specs

Rich structured ingestion — one semantic chunk per question category. Supports both simple flat records and full nested domain objects.

**Simple ingest:** `POST /vehicles/ingest`
One chunk per vehicle covering all fields.

**Rich ingest:** `POST /vehicles/ingest/rich`
Multiple semantic chunks per vehicle:

| chunk_index | Type | Answers |
|---|---|---|
| :1 | identity | what class, body style, fuel type |
| :2 | performance | engine, hp, torque, 0-60, drivetrain |
| :3 | ownership_cost | 5-year cost, insurance, depreciation, resale |
| :4 | rankings | US News, Consumer Reports, KBB rankings as prose |
| :5 | safety | NHTSA, IIHS ratings, AEB, blind spot |
| :6 | features_trims | trim levels, added features, pricing |
| :7 | reviews | expert review scores and summaries |
| :10–:14 | maintenance | one chunk per service interval milestone |
| :20+ | recall | one chunk per open recall |

Gaps at :8–:9 and :15–:19 are reserved for future chunk types — avoids re-ingestion when new types are added.

**Key design decisions:**

- Rankings converted to narrative prose: `"Ranked 2nd of 18 sports sedans"` embeds better than `rank:2, total:18`
- Maintenance split per interval: "30k service cost" retrieves exactly chunk :12, not all maintenance history
- Identity anchor in every chunk: every chunk repeats `"2025 BMW M3 Competition sports sedan"` so retrieval never loses vehicle context
- `NumericFilter`: post-retrieval re-sort for threshold queries like "over 500 horsepower" — embedding models cannot rank by numeric value

**Ask endpoint:** `POST /vehicles/{vehicleId}/ask`
**Fleet search:** `POST /vehicles/ask`
**Evaluation:** `GET /api/eval/vehicles/recall/report`

---

### 3. CMS articles (MotorTrend)

Long-form article ingestion with many-to-many vehicle references. A single article can feature multiple vehicles (comparison tests, buyers guides).

**Ingest:** `POST /articles/ingest`

#### Chunk types

| chunk_index | Type | Answers |
|---|---|---|
| :1 | identity + verdict | what is this article, overall verdict |
| :2 | ratings narrative | structured scores as prose |
| :3 | pros and cons | strengths and weaknesses |
| :4 | vehicle references | all vehicles featured (primary + competitors) |
| :10+ | article sections | one chunk per named section |
| :50+ | body text windows | overlapping windows from recursive splitter |

Gaps at :5–:9 reserved for future types (comments summary, comparison table, spec sheet).

#### Many-to-many vehicle anchor

Every chunk in every article repeats all referenced vehicle names:

```
"MotorTrend comparison featuring 2025 BMW M3 Competition,
 2025 Mercedes-AMG C63 S E Performance, 2025 Cadillac CT4-V Blackwing."
```

Querying for any referenced vehicle — primary or competitor — retrieves the article. A query for "Mercedes AMG C63" retrieves the comparison article even though Mercedes is a `competitor` role, not `primary`.

#### Recursive semantic text splitter

Long article body text is split using a 4-level separator hierarchy:

```
Level 1: paragraph boundaries (\n\n)
Level 2: sentence endings (. ! ?)
Level 3: clause boundaries (, ; :)
Level 4: word boundaries (last resort)
```

Each level is only tried when the previous produces chunks still exceeding `MAX_CHUNK_WORDS=600`. Overlap is sentence-based (2 sentences shared between adjacent chunks) — semantically cleaner than word-count overlap.

#### Section title enrichment (UAC-first)

Section titles are enriched with user query vocabulary before embedding:

```
"Performance"  → "Performance and track testing"
"Interior"     → "Interior cabin quality and comfort"
"Technology"   → "Technology infotainment and features"
"Value"        → "Value pricing and cost of ownership"
```

Queries like "performance on track" retrieve the Performance section chunk rather than a less precise body window chunk.

#### Retrieval modes

**Single-article ask:** `POST /articles/{articleId}/ask`
Scoped to one article. Uses HYBRID retrieval.

**Vehicle-scoped search:** `POST /articles/search/vehicle`
Finds all articles that mention a vehicle — works for any role (primary, competitor, mentioned).

**Cross-article semantic search:** `POST /articles/search`
Searches globally across all article chunks. Supports opinion queries, rating threshold filters, and comparison queries. Chunk preference ranking surfaces identity/verdict/ratings chunks above body windows.

#### Domain isolation via doc_type

All chunks are tagged with `doc_type` at upsert time:

| doc_type | Domain | How derived |
|---|---|---|
| article | CMS articles | docId starts with `motortrend-` |
| vehicle | Vehicle specs | docId matches `*-YYYY(-*)?` pattern |
| pdf | PDF documents | everything else |

Article search uses `WHERE doc_type = 'article'` to prevent PDF and vehicle chunks from contaminating results. Vehicle fleet search uses unfiltered cross-domain methods unchanged.

---

## Evaluation

**Run:** `GET /api/eval/vehicles/recall/report`

**Result:** 86.7% overall recall against 29 golden set entries (target: 85%)

### Golden set coverage

| Source | Count | Purpose |
|---|---|---|
| DB_GENERATED | 5 | Structured filter queries with exact DB ground truth |
| MANUAL | 10 | One query per chunk type, human-verified |
| USER_LOG | 6 | Conversational paraphrased queries |
| LLM_ASSISTED | 3 | Complex semantic cross-chunk queries |
| EDGE_CASE | 5 | Boundary conditions: nonsense queries, short tokens, missing data |

### Failure diagnosis

Each failed entry includes a `FailureAnalysis` with primary reason and actionable suggestion:

| Reason | Meaning | Fix |
|---|---|---|
| MISSING_CHUNKS | Vehicle not ingested | Call `/vehicles/ingest` |
| MISSING_CHUNK_TYPE | Simple ingest, needs rich | Call `/vehicles/ingest/rich` |
| VOCABULARY_MISMATCH | Chunk text doesn't match query terms | Rewrite chunk prose |
| OUTRANKED | Chunk exists but pushed below top-K | Increase topK |
| LOW_SCORE | Chunk scores below threshold | Add narrative context |
| EDGE_CASE_LEAK | Nonsense query returned results | Add minimum score filter |

---

## Agentic patterns

### Planner–executor

Planner generates structured JSON workflows. Executor dispatches steps sequentially with terminal-step early-return to prevent re-execution loops.

### Immutable session state

`AgentSessionState` uses `empty()` factory + `withX()` copy helpers. Adding a new field never requires updating existing call sites.

### Vehicle tools

| Tool | Purpose |
|---|---|
| FetchVehicleSpecs | Vector lookup of existing spec chunk |
| CompareVehicles | Multi-vehicle LLM comparison |
| GenerateVehicleSummary | LLM narrative from retrieved chunks |
| EnrichVehicleData | Fetch → parse → augment → re-ingest |

### Communication tools

Email (draft → review → send) and SMS (compose → confirm → send) with selective human-in-the-loop for high-impact actions.

### Divergence detection

Detects deviation from plan execution. Triggers re-planning on invalid outputs, missing data, or execution mismatch.

---

## Observability

- **OpenTelemetry:** spans on every retrieval call, embedding call, LLM call, and RRF fusion step
- **LangSmith:** execution traces, prompt inspection, retrieval debugging, workflow monitoring

---

## Tech stack

| Component | Technology |
|---|---|
| Runtime | Java 21 / Spring Boot 3 |
| Vector store | PostgreSQL + pgvector (IVFFLAT, cosine) |
| Embeddings | Ollama nomic-embed-text |
| LLM | Ollama llama3 |
| Keyword search | PostgreSQL tsvector + plainto_tsquery |
| Observability | OpenTelemetry + LangSmith |

---

## Key design decisions

**Chunking = partitioning.** Semantic chunking is a partitioning problem — split by access pattern (question type), not data structure. Too coarse loses precision, too fine loses context.

**Push filtering into retrieval, not LLM.** Every structural decision that can be made by the retriever should be. Embedding models cannot rank by numeric value — `NumericFilter` handles that. LLMs handle synthesis and language only.

**UAC-first chunk text.** Chunk text is written to match user query vocabulary, not developer field names. `"Ranked 2nd of 18 sports sedans"` embeds far better than `"rank:2, total:18"`.

**Score-regime-aware thresholds.** RRF scores top out at ~0.16. Cosine scores range 0.0–1.0. The same hardcoded threshold applied to both destroys retrieval quality. Detect the scoring regime by magnitude and apply the right threshold.

**Deterministic document IDs.** `Math.abs(vehicleId.hashCode())` not `System.currentTimeMillis()` — re-ingesting the same vehicle must produce the same documentId every time.

**Reserved chunk index gaps.** Maintenance starts at :10 not :8, recalls at :20. Leaves :8–:9 for future chunk types without requiring schema migration or re-ingestion of existing data.

---

## Roadmap

- Persistent AgentStateStore (replace in-memory with PostgreSQL)
- Async ingestion pipeline with checkpoint table for resumable exactly-once semantics
- HNSW index evaluation vs IVFFLAT at scale
- Fine-tuned embedding model for automotive domain vocabulary
- Streaming agent responses via SSE
- Live pricing and dealer inventory tools
- Multi-agent orchestration
