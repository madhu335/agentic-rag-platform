# Agentic RAG Platform (Hybrid Search + AI Workflows)

A production-grade agentic AI platform built with Spring Boot, pgvector, ParadeDB, Triton, and vLLM, supporting multi-agent orchestration, hybrid retrieval (BM25 + vector + RRF), semantic chunking, two-tier retrieval for fleet-scale search, batch ingestion, and full observability.

---

## Overview

This platform enables:

- **Multi-agent orchestration** — supervisor delegates to specialized agents (vehicle, article, research, communication) with inter-agent communication via shared tools
- **Single-agent workflows** — planner-executor pattern with 13 step types for backward compatibility
- **Hybrid RAG** — BM25 + vector + RRF fusion, with ParadeDB BM25 for single-query hybrid
- **Multi-domain ingestion** — PDF, vehicle specs, CMS articles with batch embedding
- **Semantic chunking** — by question category, not token count
- **Two-tier retrieval** — vehicle summary index (90K rows) + detail chunks for fleet-scale search
- **Three-layer answer validation** — citation validation, judge evaluation, judge reason validation
- **Partitioned vector storage** — halfvec (float16) indexes with IVFFlat on vehicle partition
- **Evaluation pipeline** — precision/recall with failure diagnosis
- **Observability** — OpenTelemetry, LangSmith, session dashboard with full context visibility

---

## Architecture

### Multi-agent flow (POST /agent/multi)

```
User query
  -> SupervisorPlanner (numbered routing rules + few-shot examples)
  -> SupervisorAgent dispatches to specialized agents:
     -> ArticleSubAgent  (article search, vehicle-enriched content, judge validation)
     -> VehicleSubAgent  (specs, summaries, comparisons)
     -> ResearchSubAgent (PDF/document RAG with divergence detection)
     -> CommunicationSubAgent (email, SMS)
  -> Session state persisted with ArticleSnapshot, VehicleSnapshot
  -> Response with citations, judge score, metadata
```

### Single-agent flow (POST /agent)

```
User query
  -> PlannerService (structured JSON plan)
  -> AgentExecutorService (step-by-step dispatch)
  -> Tools (vehicle, research, email, SMS)
  -> State store (session tracking)
  -> Evaluation / re-plan
  -> Response
```

### Retrieval pipeline

```
User query
  -> Embed (Triton text_embedding, 768d)
  -> BM25 keyword search (ParadeDB Tantivy or PostgreSQL tsvector)
  -> Vector search (pgvector halfvec cosine)
  -> RRF fusion (app-level or ParadeDB single-query)
  -> Re-ranking (Triton cross_reranker or cosine second pass)
  -> LLM synthesis (vLLM Meta-Llama-3.1-8B-Instruct)
  -> Judge evaluation (grounded/correct/complete)
  -> Response with citations
```

### Two-tier fleet retrieval

```
Fleet query ("which car has best fuel economy?")
  -> Tier 1: vehicle_summaries (90K rows, HNSW on halfvec) -> top-10 candidates  (~5ms)
  -> Tier 2: chunks_vehicle WHERE doc_id IN (top-10) -> detail chunks           (<1ms)
  -> Total: ~6ms instead of 500ms-2s with single-tier at 9M chunks
```

---

## Multi-Agent Architecture

### Supervisor planner

The supervisor decomposes user requests into agent delegations using numbered routing rules:

1. Does the user mention "article", "review", "MotorTrend", "rated"? -> article agent
2. Does the user ask about specs, performance, engine, horsepower? -> vehicle agent
3. Does the user ask about documents or general knowledge? -> research agent
4. Does the user ask to email/text/send? -> communication agent (after content agent)

Seven few-shot examples in the prompt ensure reliable routing with llama3.1.

### Article sub-agent

Four execution paths routed by task + args:

| Path | Trigger | What it does |
|---|---|---|
| Single-article ask | `articleId` in args | Scoped RAG on one article |
| Cross-article search | Default | Search all articles, LLM synthesis, judge |
| Vehicle-scoped search | `vehicleQuery` in args | Find articles about a vehicle |
| Vehicle-enriched search | Task contains "top ranked" / "with specs" | Articles + shared tool spec fetch + merge + LLM + judge |

### Inter-agent communication

The article agent calls `FetchVehicleSpecsTool` directly (same Spring bean the vehicle agent uses). No circular dependency -- both agents depend on the tool, not on each other. This is the "shared tool" pattern.

Vehicle IDs are extracted from chunk text (`vehicleId:xxx` tokens embedded by `ArticleChunkBuilder`) with three-source priority:
1. Explicit args from supervisor
2. Chunk text regex: `vehicleId:([a-zA-Z0-9-]+)`
3. ArticleId naming convention fallback

### Three-layer answer validation

| Layer | What it catches | Action on failure |
|---|---|---|
| Citation validation | Hallucinated `[ID]` tags | Auto-correct to nearest valid ID, then retry with explicit valid ID list |
| Judge evaluation | Factual errors, incomplete answers | Retry with judge feedback if reason is validated |
| Judge reason validation | False negatives from judge | Skip retry if judge's "missing X" claims are contradicted by context |

Auto-correction maps hallucinated chunk indices to valid ones (e.g. `[bmw-m3-2025-competition:6]` -> `[bmw-m3-2025-competition:2]`) eliminating retry loops.

### Session observability

Every delegation's history entry includes:
- `_result`: contextChunks, retrievedArticles, vehicleSpecs, latency_ms, specs_resolved
- `_judge`: grounded, correct, complete, score, reason
- `ArticleSnapshot`: articleIds, extractedVehicleIds, resolvedVehicleIds, operation, judgeScore

---

## Domains

### 1. PDF documents

Semantic chunking via sliding window. Used for interview Q&A, technical guides, and general document RAG.

**Endpoint:** `POST /api/pdf/upload`

---

### 2. Vehicle specs

Rich structured ingestion -- one semantic chunk per question category. Supports simple flat records, full nested domain objects, and bulk ingestion.

**Simple ingest:** `POST /vehicles/ingest`
**Rich ingest:** `POST /vehicles/ingest/rich` (batch embed + summary population)
**Bulk ingest:** `POST /vehicles/ingest/bulk` (pages of 50, batch embed + batch upsert + summaries)

#### Chunk layout

| chunk_index | Type | Answers |
|---|---|---|
| :1 | identity | class, body style, fuel type, MSRP |
| :2 | performance | engine, hp, torque, 0-60, drivetrain |
| :3 | ownership_cost | 5-year cost, insurance, depreciation, resale |
| :4 | rankings | US News, Consumer Reports, KBB as narrative prose |
| :5 | safety | NHTSA, IIHS ratings, AEB, blind spot |
| :6 | features_trims | trim levels, added features, pricing |
| :7 | reviews | expert review scores and summaries |
| :10+ | maintenance | one chunk per service interval milestone |
| :20+ | recall | one chunk per open recall |

#### Two-tier retrieval

**Fleet search:** `POST /vehicles/ask/fleet`
Uses `vehicle_summaries` table (one embedding per vehicle) for Tier 1 candidate selection, then detail chunk retrieval scoped to candidates.

**Hybrid search:** `POST /vehicles/ask/hybrid`
ParadeDB BM25 + vector similarity in a single SQL query. Tunable `vectorWeight` parameter (0.0 = pure BM25, 1.0 = pure vector).

**Admin:** `GET /vehicles/admin/summaries`
Lists all vehicle summaries with embedding status and chunk counts.

---

### 3. CMS articles (MotorTrend)

Long-form article ingestion with many-to-many vehicle references. Articles feature vehicleId tokens in chunk anchors for clean vehicle ID extraction.

**Ingest:** `POST /articles/ingest`

#### Chunk types

| chunk_index | Type | Answers |
|---|---|---|
| :1 | identity + verdict | article metadata, overall verdict |
| :2 | ratings narrative | structured scores as prose |
| :3 | pros and cons | strengths and weaknesses |
| :4 | vehicle references | all vehicles featured (primary + competitors) |
| :10+ | article sections | one chunk per named section |
| :50+ | body text windows | overlapping windows from recursive splitter |

#### Vehicle anchor with IDs

Every chunk includes machine-parseable vehicle IDs:
```
MotorTrend comparison featuring 2025 BMW M3 Competition, 2025 Mercedes-AMG C63 S E Performance.
Vehicles: vehicleId:bmw-m3-2025-competition, vehicleId:mercedes-amg-c63-2025.
```

---

## Batch Ingestion

All ingestion paths use batch embedding and batch upsert:

| Path | Before (serial) | After (batch) |
|---|---|---|
| Rich vehicle (14 chunks) | 14 embed + 14 INSERT = 28 round-trips | 1 embed + 1 batch INSERT = 2 round-trips |
| Article (15 chunks) | 15 embed + 15 INSERT = 30 round-trips | 1 embed + 1 batch INSERT = 2 round-trips |
| Bulk (50 vehicles x 14 chunks) | 700 round-trips | ~22 embed + 1 batch INSERT per page |

Triton batch embedding is used for ingestion. Sub-batches are sent per HTTP call to the `text_embedding` model, and JDBC `BatchPreparedStatementSetter` handles batch INSERT/UPSERT.

---

## Partitioned Storage

The `document_chunks` table is partitioned by `doc_type`:

| Partition | Index type | Purpose |
|---|---|---|
| `chunks_vehicle` | IVFFlat on halfvec | Fast filtered search at 90K+ vehicles |
| `chunks_article` | HNSW on halfvec | High recall for smaller article corpus |
| `chunks_pdf` | HNSW on halfvec | General document search |

All indexes use `halfvec(768)` (float16) for 2x memory reduction. A trigger auto-populates `embedding_half` from `embedding` on every INSERT/UPDATE.

IVFFlat probe count is set per connection via HikariCP:
```properties
spring.datasource.hikari.connection-init-sql=SET ivfflat.probes = 10
```

---

## Evaluation

**Run:** `GET /api/eval/vehicles/recall/report`

**Result:** 86.7% overall recall against 29 golden set entries (target: 85%)

### Failure diagnosis

Each failed entry includes a `FailureAnalysis` with primary reason and actionable suggestion:

| Reason | Meaning | Fix |
|---|---|---|
| MISSING_CHUNKS | Vehicle not ingested | Call `/vehicles/ingest` |
| MISSING_CHUNK_TYPE | Simple ingest, needs rich | Call `/vehicles/ingest/rich` |
| VOCABULARY_MISMATCH | Chunk text doesn't match query | Rewrite chunk prose |
| OUTRANKED | Chunk pushed below top-K | Increase topK |
| LOW_SCORE | Chunk scores below threshold | Add narrative context |
| EDGE_CASE_LEAK | Nonsense query returned results | Add minimum score filter |

---

## Observability

### Session dashboard
- All agent sessions persisted as immutable `AgentSessionState` snapshots
- `ArticleSnapshot` shows articleIds, extractedVehicleIds, resolvedVehicleIds, operation, judgeScore
- `VehicleSnapshot` shows vehicleId, summary, specChunkIds
- History entries include `_result` (context chunks, retrieval scores) and `_judge` (grounded, score, reason)

### Tracing
- **OpenTelemetry:** spans on retrieval, embedding, LLM, RRF fusion
- **LangSmith:** execution traces, prompt inspection, workflow monitoring

---

## Tech stack

| Component | Technology |
|---|---|
| Runtime | Java 21 / Spring Boot 3.5 |
| Vector store | PostgreSQL 16 + pgvector 0.8 (HNSW, IVFFlat, halfvec) |
| Full-text search | ParadeDB pg_search (Tantivy BM25) |
| Embeddings | Triton `text_embedding` (768d) |
| LLM | vLLM `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Keyword search | PostgreSQL tsvector + ParadeDB BM25 |
| Container | Docker (paradedb/paradedb:0.19.11-pg16) |
| Observability | OpenTelemetry + LangSmith |

---

## Key design decisions

**Chunking = partitioning.** Semantic chunking is a partitioning problem -- split by access pattern (question type), not data structure.

**Push filtering into retrieval, not LLM.** Every structural decision that can be made by the retriever should be. Embedding models cannot rank by numeric value -- `NumericFilter` handles that.

**UAC-first chunk text.** Chunk text matches user query vocabulary: `"Ranked 2nd of 18 sports sedans"` embeds better than `"rank:2, total:18"`.

**Two-tier retrieval.** At 90K vehicles, scan summaries (90K rows) first, then detail chunks within candidates. Reduces fleet search from 500ms to ~6ms.

**Auto-correct citations.** LLMs hallucinate chunk indices. Auto-correct `[vehicle:6]` to `[vehicle:2]` (same prefix, valid ID) instead of burning retries.

**Judge reason validation.** Don't blindly feed judge feedback into retry prompts. If the judge says "missing specs" but specs are in the context, the judge is wrong -- skip retry.

**Shared tool pattern.** Inter-agent communication via Spring DI -- both agents depend on the tool, not on each other. No circular dependencies.

**Partitioned indexes.** IVFFlat on vehicle partition (fast filtered search), HNSW on article/pdf (higher recall, smaller corpus). halfvec for 2x memory reduction.

---

## Experiments

### LlamaIndex comparison (`experiments/llamaindex/`)

### LlamaIndex comparison (`experiments/llamaindex/`)

Python-based LlamaIndex experiment mirroring the Java RAG pipeline stage-by-stage:
- Triton embeddings + vLLM answer generation
- Same Postgres DB (pgvector table `data_llamaindex_chunks`)
- Side-by-side retrieval comparison (vector, BM25, hybrid)
- BM25 in LlamaIndex is in-memory Python (`rank_bm25`), not pg-side FTS

### LangGraph / LangGraph multi-agent (`experiments/langgraph/`, `experiments/langgraph-multi/`)

Python-based graph and multi-agent experiments using Triton for embeddings and vLLM for planner / generation / judge, for comparison with the Java orchestration patterns.

---

## Running

### Prerequisites
- Java 21
- Docker Desktop
- Triton running with `text_embedding` and `cross_reranker`
- vLLM running with `meta-llama/Meta-Llama-3.1-8B-Instruct`

### Start infrastructure
```bash
docker compose up -d    # Postgres + Triton + vLLM
```

### Run migration
```bash
# Apply partitioning + two-tier + ParadeDB
docker exec -i ai-rag-postgres psql -U postgres -d ai_rag_assistant < V2__partition_two_tier_paradedb.sql
```

### Start application
```bash
./mvnw spring-boot:run
```

### Seed data
Run `.http` files in IntelliJ in order:
1. `src/test/java/resources/ingestion/vehicle/seed_vehicles.http`
2. `src/test/java/resources/ingestion/vehicle/rich_vehicle_ingest.http`
3. `src/test/java/resources/performance/twoTier/two-tier-retrieval-test.http`
4. `src/test/java/resources/agent/multi-agent.http`
5. any scenario-specific files under `src/test/java/resources/ask/`, `stream/`, or `ingestion/article/`

### Run tests
```bash
./mvnw test
```
