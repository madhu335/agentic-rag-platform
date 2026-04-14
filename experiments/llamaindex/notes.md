# Notes: Custom RAG vs. LlamaIndex

Lab notebook. Fill in the blanks after running both pipelines on the same docs.

## Development speed

| Stage       | Custom (Java) LOC | LlamaIndex (Py) LOC | Notes |
|-------------|------------------:|--------------------:|-------|
| Ingestion   | ~250 (PdfExtractorService + TextChunker + EmbeddingService + PgVectorStore writes) | ~30 (`ingest.py`) | LlamaIndex hides the reader/splitter/embed/store wiring behind `VectorStoreIndex.from_documents` |
| Retrieval   | ~200 (`RagRetriever` + `PgVectorStore.hybridSearch` + `ReRankService`) | ~25 (`retrieve.py`) | Fusion is one constructor call |
| Answer      | ~150 (`RagAnswerService` + prompt building + `LlmClient`) | ~10 (`answer.py`) | `RetrieverQueryEngine.from_args` |
| **Total**   | ~600              | ~65                 | ~10x less code for the happy path |

> Fill in the real LOC after measuring. `cloc` or `wc -l` is fine.

## Retrieval quality (same questions, same docs, same models)

Run this table on 5–10 questions. Put a ✅ / ❌ / ~ in each cell by eyeballing the top-3 chunks.

| Question | Java VECTOR | LI vector | Java HYBRID | LI hybrid | Winner |
|----------|:-----------:|:---------:|:-----------:|:---------:|:------:|
| ...      |             |           |             |           |        |

Things to watch for:

- **BM25 lives in different places.** Your Java BM25 runs in Postgres via `ts_vector` + `ts_rank_cd` — it scales with the DB and benefits from pg's FTS features (stemming, stopwords, GIN index). LlamaIndex's `BM25Retriever` pulls every node out of pgvector on startup and builds a `rank_bm25` index in Python memory. For small corpora this is fine and sometimes faster (no round-trip per query). For a large corpus it's a non-starter — you'd have to either use pg FTS manually, or write a custom retriever. **This is the single biggest architectural divergence in the experiment and a great interview talking point.**
- **Short tokens** ("M3", "V8"). Your Java BM25 falls back to vector because Postgres FTS drops tokens under 3 chars. LlamaIndex's `rank_bm25` uses a different tokenizer — it may or may not handle these better. Test it.
- **Fusion algorithm.** Java does weighted linear combination in SQL. LlamaIndex's `QueryFusionRetriever` with `mode="reciprocal_rerank"` uses RRF. RRF is more robust to score-scale mismatches; linear weighting gives you a tuning knob. Neither is strictly better.
- **Chunk boundaries.** `TextChunker` splits on `Q\d+\.` which is perfect for FAQ-shaped docs and bad for everything else. `SentenceSplitter` is generic and respects sentence boundaries. Try both on a non-FAQ doc and see which retrieval wins.

## Control surface

Where you lose control going to LlamaIndex (this is the honest sales pitch against the framework):

- **SQL you can't see.** In Java you wrote `embedding <=> ?::vector` and `ts_rank_cd(...)` by hand. With the in-memory default store, you can't even look at the index. With `PGVectorStore`, you can — but the query shape is the framework's, not yours.
- **Rerank as a postprocessor.** Your Java `ReRankService` is a first-class step in `RagRetriever`. In LlamaIndex, reranking is a `node_postprocessor` on the query engine. Fine for the default case, awkward when you want to rerank *before* fusion.
- **Chunking policy.** Your regex-on-`Q\d+\.` is weird but intentional — it matches the actual shape of your FAQ source. `SentenceSplitter` doesn't know about that structure. You can write a custom `NodeParser`, but at that point you're writing the same code you had in Java.
- **Observability.** Your Java side has `TraceHelper` threading LangSmith attributes through every call. LlamaIndex has callbacks/instrumentation but you have to wire them; you don't get span metadata for free.
- **Ingestion is append-only against a persistent store.** `VectorStoreIndex.from_documents(...)` with a `PGVectorStore` storage context writes new nodes without deduping or upserting by source document. Re-running ingestion on the same files doubles the corpus. Verified in practice: first ingest of 2 PDFs produced 19 chunks, second ingest of the same 2 PDFs would've produced 34. Your Java `VectorIngestionService` handles this explicitly (upsert or truncate-then-insert by `doc_id`) — in LlamaIndex the workaround is either `vector_store.delete(ref_doc_id=...)` before each ingest, or `DROP TABLE` / truncate. This is a real production-readiness gap: a framework that makes the happy path trivial but quietly pushes the hard correctness problem onto you.

Where you gain speed:

- **Zero-to-pipeline is minutes, not days.** Especially valuable for prototyping a new ingestion source or trying a new retrieval strategy.
- **Swap components with a config change.** Want OpenAI instead of Ollama? Change one import. Want Qdrant instead of in-memory? One line on `StorageContext`.
- **Built-in evals.** `llama_index.core.evaluation` has faithfulness / relevancy / correctness evaluators out of the box. Your Java side would need these hand-rolled.

## When to pick which (interview talking-point answer)

- **LlamaIndex** when: fast prototyping, swapping components often, team doesn't want to own retrieval internals, standard chunking/fusion is good enough, you want built-in evals.
- **Custom (your Java stack)** when: retrieval quality is the product, chunking has domain structure (FAQ, legal clauses, code), you need per-tenant SQL tuning, you're already running Postgres and don't want a second store, observability/audit trails are non-negotiable.

The JD framing: "I built a custom hybrid-retrieval RAG pipeline in Java over pgvector with BM25 fusion and reranking, and I've also used LlamaIndex for faster iteration on new document types. I reach for the framework when I'm exploring and drop to the custom stack when retrieval quality or control is the constraint." That's a stronger answer than either "I use LlamaIndex" or "I built my own" alone.

## Open questions to try

- [ ] Wire `PGVectorStore` from `llama-index-vector-stores-postgres` pointing at the same Postgres DB your Java app uses. Can LlamaIndex read chunks your Java pipeline wrote?
- [ ] Add `SentenceTransformerRerank` as a node postprocessor. How close does it get to your Java `ReRankService` output?
- [ ] Swap `SentenceSplitter` for a custom `NodeParser` that does the `Q\d+\.` split. Does hybrid retrieval improve on FAQ docs?
- [ ] Turn on `llama_index.core.evaluation.FaithfulnessEvaluator` and run it on 20 Q&A pairs against both pipelines. Numbers, not vibes.

## Live experiment: heap sizing question
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
   retriever produced it. Your Java `SearchHit.score()` has the same problem if
   you're comparing across `RetrievalMode` values — worth documenting this gotcha
   on the Java side too.

### Interview talking point

"On a heap-sizing question against a 17-chunk corpus, LlamaIndex's default RRF
hybrid actually performed *worse* than its own vector-only retriever, and worse
than my Java pipeline would have, because RRF is rank-based and can't override
a single confidently-wrong retriever. That's when I'd reach for a custom fusion
— or just use vector-only when the embedding quality is high enough, which for
`nomic-embed-text` on technical docs it was."

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

**What this says about production:** reach for hybrid when your corpus is
domain-coherent and BM25 can act as a keyword-recall safety net. Reach for
vector-only when your corpus is heterogeneous and you need retrieval to
*exclude* unrelated domains. Don't assume hybrid is strictly better.

### BM25 behavior note

BM25's top score on the auto-config chunk was 3.63, the highest in the whole
experiment, because "auto-configuration" is rare and high-IDF in this corpus.
BM25 excels when question terms appear verbatim in the answer chunk and falls
apart when there's a lexical gap (heap question: question says "heap size",
chunk says "HeapDumpOnOutOfMemoryError" — no overlap, weak ranking).

## Head-to-head: Java hybrid vs LlamaIndex hybrid, same 2 questions

Setup: identical corpus (same 2 PDFs), identical models (nomic-embed-text +
llama3.1), same Postgres instance. Java uses RagRetriever's default HYBRID
mode via OrchestratorService; LlamaIndex uses QueryFusionRetriever (RRF).

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
when scoped to the irrelevant doc (judge score=0.0). Importantly, Java's
bestScore on Q2a was 0.9791 — a real cosine similarity, not an RRF rank
artifact — suggesting the orchestrator short-circuited fusion when a clear
winner was available.

### The core finding

LlamaIndex's `RetrieverQueryEngine` always generates an answer, even when
retrieval returned garbage. There's no built-in "I don't know" path. The Java
pipeline has:

1. **Doc-scoped retrieval** — no cross-doc chunk pollution
2. **A judge step** — independently evaluates grounded/correct/complete
3. **A refusal path** — returns "I don't know based on the ingested documents"
   when retrieval confidence or judge score is too low

Result on identical inputs: **LlamaIndex hallucinated one answer; Java refused
to guess on two answers and got the rest right.** The production-readiness
gap isn't subtle. To match this behavior in LlamaIndex you'd have to build:
- A custom retriever that filters by metadata (doc-scoping)
- A custom response synthesizer that evaluates the retrieved context
- A FaithfulnessEvaluator or similar postprocessor
- Glue code to threshold on both signals and emit a refusal

All of which exist as building blocks, but none of which are wired together
by default. In the Java pipeline it's the default behavior.

### Interview framing

"I ran the same two PDFs and two questions through my Java RAG pipeline and
a LlamaIndex equivalent. On a heap-sizing question, LlamaIndex's default
hybrid retriever hallucinated because reciprocal rank fusion can't override
a confidently-wrong retriever, and the query engine has no refusal path. My
Java pipeline scoped retrieval by document ID, routed through a judge step,
and cleanly returned 'I don't know' when the answer wasn't in the right doc —
then gave a correct cited answer when it was. Same LLM, same Postgres, same
embeddings. The difference was the orchestration and retrieval layer.

That said, LlamaIndex got me to a working end-to-end pipeline in about 60
lines of Python vs ~600 lines of Java. For prototyping a new document type
or trying a new retrieval strategy, that's a massive speedup. The right
answer is 'use both': LlamaIndex to explore, the custom stack to productionize
when quality and refusal behavior matter."

### Caveat: what `bestScore = 0.0909` means

Across three different queries (Q1a, Q1b, Q2b) the Java response came back
with bestScore exactly 0.0909 ≈ 1/11. That's not a cosine similarity —
it's a rank-based fusion artifact (1/(k+rank) with k=10, rank=1). Q2a
came back with 0.9791, which IS a cosine similarity. So the Java orchestrator
has two code paths: one that returns the raw vector score when there's a
clear winner, and one that falls back to rank fusion when nothing stands out.
TODO: trace the exact logic in RagAnswerService / OrchestratorService /
PgVectorStore.hybridSearch to describe this accurately. The overall finding
stands either way.