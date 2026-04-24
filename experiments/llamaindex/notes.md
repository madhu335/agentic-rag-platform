# Notes: Custom RAG vs. LlamaIndex

Lab notebook comparing a custom Java RAG pipeline (pgvector + hybrid retrieval +
rerank) against a LlamaIndex implementation on the same corpus and models.

## Development speed

Measured with `cloc` v2.08, code-only (excludes blanks and comments).

| Stage       | Custom (Java) | LlamaIndex (Python) | Ratio | Files (Java / Py) |
|-------------|--------------:|--------------------:|------:|:-----------------:|
| Ingestion   | 230 LOC       | ~70 LOC             | 3.3x  | 11 / 4            |
| Retrieval   | 657 LOC       | ~70 LOC             | 9.4x  | 9 / 4             |

| Answer      | 596 LOC       | ~30 LOC             | 19.9x | 5 / 4             |
| **Total**   | **1,483 LOC** | **195 LOC**         | **7.6x** | **25 / 4**     |

Notice the ratio gets *bigger* as you move from ingestion to retrieval to
answer:

- **Ingestion (3.3x)** — both pipelines do similar work (parse PDF, chunk,
  embed, write to pg). The work is irreducible; LlamaIndex saves wiring.
- **Retrieval (9.4x)** — Java has hand-written SQL, RRF fusion math,
  multi-mode dispatch, and observability hooks. LlamaIndex hides all of that
  behind constructors. Big code savings, big control loss.
- **Answer (19.9x)** — Java has four layers of refusal logic, dual-scale
  score thresholds, prompt cleaning, citation extraction, and a multi-LLM
  router. LlamaIndex has `query_engine.query(q)`. **And the missing
  functionality is exactly what hallucinated on the heap question.**

The layer where LlamaIndex saves the most code is the same layer where it
produced the worst answer. That's not a coincidence — the deleted code
wasn't bloat, it was correctness logic.

> LOC is approximate — measure with `cloc` or `wc -l` for exact numbers.

## Retrieval quality (same questions, same docs, same models)

Two questions, two docs, both pipelines. LlamaIndex queries the whole corpus
(17 chunks); Java requires explicit `docId` scoping (one PDF at a time). That
asymmetry is itself a finding — Java's design eliminates an entire class of
cross-doc pollution failures, but requires upstream routing to pick the
right doc.

Legend: ✅ correct grounded answer · ❌ hallucinated · 🚫 refused (returned
"I don't know") · ~ correct but weaker

| Question | Java VECTOR | Java BM25 | Java HYBRID | LI vector | LI BM25 | LI hybrid |
|----------|:-----------:|:---------:|:-----------:|:---------:|:-------:|:---------:|
| **Q1: heap sizing** (jvm-flags-guide scope) | n/a | n/a | ✅ cited [5,6] | ✅ | ✅ | ❌ hallucinated |
| **Q1: heap sizing** (spring-boot-qa scope)  | n/a | n/a | 🚫 grounded=false | (corpus-wide) | (corpus-wide) | (corpus-wide) |
| **Q2: auto-config** (spring-boot-qa scope)  | n/a | n/a | ✅ cited [1] (cosine 0.98) | ✅ | ✅ | ✅ richest |
| **Q2: auto-config** (jvm-flags-guide scope) | n/a | n/a | 🚫 grounded=false | (corpus-wide) | (corpus-wide) | (corpus-wide) |

Notes on the gaps:

- **Java VECTOR / BM25 columns are `n/a`** because `OrchestratorService` uses
  HYBRID by default via `RagRetriever.retrieve(docId, question, topK)` — the
  HTTP API doesn't expose per-mode dispatch. The Java numbers above are all
  hybrid mode. Adding a `mode` parameter to `AskController` would let us fill
  these in for direct vector/BM25 head-to-heads.
- **LlamaIndex doesn't have rows for both doc scopes** because there's no
  `docId` filter in the default retriever — it always queries the full
  corpus. The single LlamaIndex result per question covers the whole 17-chunk
  corpus.

Headline result from these four cells: **the only ❌ in the table is
LlamaIndex hybrid on the heap question.** Same model, same Postgres, same
embeddings — Java's four-layer refusal architecture caught what LlamaIndex's
default query engine couldn't. Detail in the "Live experiment" sections below.

Things to watch for in future runs:
- **BM25 lives in different places.** The Java BM25 runs in Postgres via `ts_vector` + `ts_rank_cd` — it scales with the DB and benefits from pg's FTS features (stemming, stopwords, GIN index). LlamaIndex's `BM25Retriever` pulls every node out of pgvector on startup and builds a `rank_bm25` index in Python memory. For small corpora this is fine and sometimes faster (no round-trip per query). For a large corpus it's a non-starter — you'd have to either use pg FTS manually, or write a custom retriever. **This is the single biggest architectural divergence between the two approaches.**
- **Short tokens** ("M3", "V8"). Postgres FTS drops tokens under 3 chars, so the Java BM25 falls back to vector for short-token queries. LlamaIndex's `rank_bm25` uses a different tokenizer — it may or may not handle these better. Test it.
- **Fusion algorithm.** Java does weighted linear combination in SQL. LlamaIndex's `QueryFusionRetriever` with `mode="reciprocal_rerank"` uses RRF. RRF is more robust to score-scale mismatches; linear weighting gives you a tuning knob. Neither is strictly better.
- **Chunk boundaries.** `TextChunker` splits on `Q\d+\.` which is perfect for FAQ-shaped docs and bad for everything else. `SentenceSplitter` is generic and respects sentence boundaries. Try both on a non-FAQ doc and see which retrieval wins.

## Control surface

Where the framework trades control for convenience:

- **SQL is hidden.** The Java pipeline writes `embedding <=> ?::vector` and `ts_rank_cd(...)` by hand. With `PGVectorStore`, the query shape is the framework's, not yours.
- **Rerank as a postprocessor.** The Java `ReRankService` is a first-class step in `RagRetriever`. In LlamaIndex, reranking is a `node_postprocessor` on the query engine. Fine for the default case, awkward when you want to rerank *before* fusion.
- **Chunking policy.** The Java regex-on-`Q\d+\.` splitter matches the shape of an FAQ source. `SentenceSplitter` doesn't know about that structure. You can write a custom `NodeParser`, but at that point you're writing the same code you had in Java.
- **Observability.** The Java side threads tracing attributes (LangSmith) through every call via a `TraceHelper`. LlamaIndex has callbacks/instrumentation but you have to wire them; you don't get span metadata for free.
- **Ingestion is append-only against a persistent store.** `VectorStoreIndex.from_documents(...)` with a `PGVectorStore` storage context writes new nodes without deduping or upserting by source document. Re-running ingestion on the same files doubles the corpus. Verified in practice: first ingest of 2 PDFs produced 19 chunks, second ingest of the same 2 PDFs would've produced 34. The Java `VectorIngestionService` handles this explicitly (delete-then-insert by `doc_id`) — in LlamaIndex the workaround is either `vector_store.delete(ref_doc_id=...)` before each ingest, or `DROP TABLE` / truncate. This is a real production-readiness gap: a framework that makes the happy path trivial but quietly pushes the hard correctness problem onto you.

Where the framework gains speed:

- **Zero-to-pipeline is minutes, not days.** Especially valuable for prototyping a new ingestion source or trying a new retrieval strategy.
- **Swap components with a config change.** OpenAI instead of Ollama? Change one import. Qdrant instead of in-memory? One line on `StorageContext`.
- **Built-in evals.** `llama_index.core.evaluation` has faithfulness / relevancy / correctness evaluators out of the box. The Java side would need these hand-rolled.

## When to pick which

- **LlamaIndex** when: fast prototyping, swapping components often, the team doesn't want to own retrieval internals, standard chunking/fusion is good enough, built-in evals are desired.
- **Custom (Java stack)** when: retrieval quality is the product, chunking has domain structure (FAQ, legal clauses, code), per-tenant SQL tuning is needed, Postgres is already in the stack and a second store isn't desired, observability/audit trails are non-negotiable.

## Open questions to try

- [ ] Wire `PGVectorStore` from `llama-index-vector-stores-postgres` at the same Postgres DB as the Java app. Can LlamaIndex read chunks the Java pipeline wrote?
- [ ] Add `SentenceTransformerRerank` as a node postprocessor. How close does it get to the Java `ReRankService` output?
- [ ] Swap `SentenceSplitter` for a custom `NodeParser` that does the `Q\d+\.` split. Does hybrid retrieval improve on FAQ docs?
- [ ] Turn on `llama_index.core.evaluation.FaithfulnessEvaluator` and run it on 20 Q&A pairs against both pipelines. Numbers, not vibes.

## Live experiment: "How should I configure heap size for a Spring Boot app in production?"

Corpus: 2 PDFs (spring-boot-qa.pdf, jvm-flags-guide.pdf), 17 chunks in data_llamaindex_chunks.

### Retrieval comparison (top-5 chunks, same question, same LLM)

| Rank | Vector                | BM25                       | Hybrid (RRF)               |
|------|-----------------------|----------------------------|----------------------------|
| 1    | HeapDumpOnOOM ⭐       | @Repository (irrelevant) ❌ | HeapDumpOnOOM ⭐            |
| 2    | G1HeapRegionSize ⭐    | HeapDumpOnOOM ⭐             | @Repository ❌             |
| 3    | Spring Boot intro     | Spring Boot intro          | Spring Boot intro          |
| 4    | @Repository ❌         | Hikari pool config         | Hikari pool config         |
| 5    | Hikari pool config    | @Bean vs @Component ❌      | G1HeapRegionSize ⭐         |

### Findings

1. **BM25 alone ranked an irrelevant chunk #1.** The `@Repository` chunk has no
   heap content but scored highest (2.83) because it shares surface tokens with
   the question ("configure", "production", "Spring Boot"). Classic lexical
   overlap ≠ semantic relevance failure mode.

2. **RRF hybrid didn't recover.** The `@Repository` chunk kept rank 2 in hybrid
   despite vector scoring it 4th with much lower confidence (0.59 vs 0.67 for
   the true top). RRF is rank-only — it ignores the magnitude of per-retriever
   confidence and gives each retriever an equal vote. `G1HeapRegionSize` (vector
   rank 2) got pushed to hybrid rank 5 as a result.

3. **Contrast with the Java pipeline.** `PgVectorStore.hybridSearch` uses a
   weighted linear combination of actual scores, not ranks. When one retriever
   is much more confident, it has the numerical weight to override a weak
   signal from the other. That's not strictly better — linear weighting needs
   tuning — but it's the right tool when one retriever is clearly wrong.

4. **Worse retrieval → worse answer.** Same LLM (llama3.1), same temperature,
   same question. Vector mode gave a clean, specific answer with correct
   examples (-Xms1024m -Xmx4096m). Hybrid mode drifted into hallucination,
   including the sentence "allow it to grow up to nearly a full CPU core's
   worth of memory" — CPU cores are not memory. The polluted context window
   made the LLM hedge and invent.

5. **Score scales are incomparable across modes.**
   - Vector: 0.59–0.67 (cosine similarity, higher is better, 0–1)
   - BM25: 1.68–2.83 (raw BM25, higher is better, unbounded)
   - Hybrid: 0.016–0.033 (RRF = 1/(60+rank), always tiny)

   You literally cannot read a LlamaIndex retrieval score without knowing which
   retriever produced it. The Java `SearchHit.score()` has the same problem if
   you're comparing across `RetrievalMode` values — worth documenting this
   gotcha on the Java side too.

### Second question: "What is Spring Boot auto-configuration?"

Unambiguous, softball question — only spring-boot-qa.pdf has relevant content.

| Rank | Vector                  | BM25                        | Hybrid (RRF)                |
|------|-------------------------|-----------------------------|-----------------------------|
| 1    | Q1. auto-config ⭐       | Q1. auto-config ⭐           | Q1. auto-config ⭐           |
| 2    | @Bean vs @Component     | Hikari pool                 | @Bean vs @Component         |
| 3    | LAZY/JOIN FETCH         | @Bean vs @Component         | @Repository                 |
| 4    | @Repository             | @Repository                 | Hikari pool                 |
| 5    | Runtime flags           | G1 flag                     | LAZY/JOIN FETCH             |

All three modes put the correct chunk at rank 1. Answer quality ranking:
hybrid > bm25 > vector — the *opposite* of the heap question's result.

### Reframed finding #6 (was: "hybrid is worse")

Hybrid RRF doesn't improve answer quality uniformly. It tracks the semantic
coherence of the retrieved set. Two data points:

- **Heap question (mixed corpus):** hybrid's top-5 spanned unrelated domains.
  Irrelevant chunks polluted the context window and the LLM hallucinated
  ("CPU core's worth of memory"). Vector-only was the best answer.
- **Auto-config question (coherent corpus):** hybrid's top-5 were all Spring
  Boot content. Noisy chunks were at least topically adjacent and reinforced
  the topic. The LLM produced a richer answer referencing the actual file path
  (`META-INF/spring/...`) and three example beans. Hybrid was the best answer.

**What this says about production:** reach for hybrid when the corpus is
domain-coherent and BM25 can act as a keyword-recall safety net. Reach for
vector-only when the corpus is heterogeneous and retrieval needs to
*exclude* unrelated domains. Don't assume hybrid is strictly better.

### BM25 behavior note

BM25's top score on the auto-config chunk was 3.63, the highest in the whole
experiment, because "auto-configuration" is rare and high-IDF in this corpus.
BM25 excels when question terms appear verbatim in the answer chunk and falls
apart when there's a lexical gap (heap question: question says "heap size",
chunk says "HeapDumpOnOutOfMemoryError" — no overlap, weak ranking).

## Head-to-head: Java hybrid vs LlamaIndex hybrid, same 2 questions

Setup: identical corpus (same 2 PDFs), identical models (nomic-embed-text +
llama3.1), same Postgres instance. Java uses `RagRetriever`'s default HYBRID
mode via `OrchestratorService`; LlamaIndex uses `QueryFusionRetriever` (RRF).

### Q1: "How should I configure heap size for a Spring Boot app in production?"

| Scope              | Java (hybrid)          | LlamaIndex (hybrid)    |
|--------------------|------------------------|------------------------|
| JVM flags PDF      | ✅ Correct, cited [5,6] | n/a (no scoping)       |
| Spring Boot PDF    | ✅ "I don't know"      | n/a (no scoping)       |
| Whole corpus       | n/a (no cross-doc)     | ❌ HALLUCINATED        |

Java refused to answer from the Spring Boot PDF alone (judge: grounded=false,
score=0.3). LlamaIndex RRF over the whole 17-chunk corpus generated confident-
sounding but wrong content ("CPU core's worth of memory"). Same model, same
temperature — the difference is purely the retrieval + orchestration layer.

### Q2: "What is Spring Boot auto-configuration?"

| Scope              | Java (hybrid)                      | LlamaIndex (hybrid)    |
|--------------------|------------------------------------|------------------------|
| Spring Boot PDF    | ✅ Correct, cited, bestScore 0.9791 | ✅ Richest answer       |
| JVM flags PDF      | ✅ "I don't know"                  | n/a                    |
| Whole corpus       | n/a                                | ✅ (same as above)      |

Both got the softball right on the relevant doc. Java again refused cleanly
when scoped to the irrelevant doc (judge score=0.0). Java's bestScore on Q2a
was 0.9791 — a real cosine similarity, not an RRF rank artifact — suggesting
the orchestrator short-circuited fusion when a clear winner was available.

### The core finding

LlamaIndex's `RetrieverQueryEngine` always generates an answer, even when
retrieval returned garbage. There's no built-in "I don't know" path. The Java
pipeline has:

1. **Doc-scoped retrieval** — no cross-doc chunk pollution
2. **A judge step** — independently evaluates grounded/correct/complete
3. **A refusal path** — returns "I don't know based on the ingested documents"
   when retrieval confidence or judge score is too low

Result on identical inputs: **LlamaIndex hallucinated one answer; the Java
pipeline refused to guess on two answers and got the rest right.** The
production-readiness gap isn't subtle. To match this behavior in LlamaIndex
you'd have to build:

- A custom retriever that filters by metadata (doc-scoping)
- A custom response synthesizer that evaluates the retrieved context
- A FaithfulnessEvaluator or similar postprocessor
- Glue code to threshold on both signals and emit a refusal

All of which exist as building blocks, but none of which are wired together
by default. In the Java pipeline it's the default behavior.

### How the Java refusal logic actually works

Traced through `RagAnswerService.java`. The pipeline has four layers of
defense against hallucination, each catching a different failure mode.

**Two thresholds for two score scales.** Hybrid retrieval and vector retrieval
produce scores on incompatible scales, so the gating logic uses two constants:

- `COSINE_LOW = 0.40` — for VECTOR mode results (cosine similarity, 0–1)
- `RRF_LOW = 0.04` — for HYBRID mode results (RRF caps around 0.18 with `k=10`)

`COSINE_LOW` was tuned down from 0.55 after empirical testing showed 0.55 was
too strict for structured content (FAQs, tables). `passesKeywordGuard` inspects
`bestScore` and auto-detects which scale it's on, then applies the right
threshold. Same code path handles all retrieval modes transparently.

**Refusal is gated at four layers:**

| Layer | Where | Catches |
|---|---|---|
| 1. Score gate (`passesMinimumThreshold`) | Top hit must clear `LOW` | "retrieval found nothing relevant" |
| 2. Keyword guard (`passesKeywordGuard`) | Top chunk text must overlap the question | "high cosine similarity but actually about a different topic" |
| 3. Chunk filter (`selectUsableHits`) | Individual chunks below `LOW` dropped from LLM context | "weak supporting chunks polluting the prompt" |
| 4. Prompt fallback | LLM instructed to emit refusal string when context is insufficient | "anything that slips through the first three layers" |

When a request fails layers 1 or 2, the refusal string is returned **directly,
without ever calling the LLM**. That's both faster and more reliable than the
prompt-fallback layer alone.

**Why `bestScore = 0.0909` appeared three times.** With `k = 10` in
`fuseRRF`, the maximum possible RRF score is `2/(10+1) ≈ 0.1818` (when both
vector and BM25 rank the same chunk #1). A score of exactly `0.0909 = 1/11`
means the rank-1 chunk was found by **exactly one** of the two retrievers
at rank 1, not both. So the three queries with `bestScore = 0.0909` weren't
producing degenerate scores — they were all cases where vector and BM25
disagreed on the top chunk. Q2a's `0.9791` is a real cosine similarity
because the orchestrator routed that query to VECTOR mode (the question text
matched a chunk's first line nearly exactly), bypassing fusion entirely.

**Why this beat LlamaIndex on the heap question.** LlamaIndex's
`RetrieverQueryEngine` has none of these four layers — it always synthesizes
an answer from whatever retrieval returned. On the heap-sizing question
scoped against the wrong corpus, that meant hallucinating "CPU core's worth
of memory." The Java pipeline returned the refusal cleanly because layer 1
(score gate) caught it before the LLM was even called. Same model, same
Postgres, same embeddings. The difference was the four layers of
orchestration above the retriever.
