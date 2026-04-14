# LlamaIndex Experiment

A side-by-side comparison harness for your custom Java RAG pipeline vs. a LlamaIndex
implementation of the same four stages:

```
data -> ingestion -> retrieval -> answer
```

The goal is **not** to replace the Java pipeline. It's to feel the tradeoffs between
development speed and retrieval control firsthand — and to have something concrete to
point at in interviews when a JD mentions LlamaIndex.

## Why a separate Python module?

LlamaIndex's Java/Kotlin story is thin; the framework lives in Python and that's where
99% of the JDs expect you to have used it. Rather than force it into the Spring Boot
app, this is an isolated module you can run, throw away, or extend without touching
the main project. Think of it as a lab notebook.

## What this mirrors from the Java side

| Stage       | Java pipeline                                   | LlamaIndex pipeline                                 |
|-------------|-------------------------------------------------|-----------------------------------------------------|
| Ingestion   | `PdfExtractorService` + `TextChunker` (custom)  | `SimpleDirectoryReader` + `SentenceSplitter`        |
| Embedding   | Ollama `nomic-embed-text` via `EmbeddingClient` | `OllamaEmbedding` (same model)                      |
| Store       | Postgres + pgvector (`PgVectorStore`, table `document_chunks`) | Postgres + pgvector (`PGVectorStore`, table `data_llamaindex_chunks`) |
| Retrieval   | Hybrid (vector + BM25 + rerank)                 | `VectorIndexRetriever` + optional `BM25Retriever`   |
| Answer      | `RagAnswerService` + Ollama `llama3.1`          | `RetrieverQueryEngine` + `Ollama` LLM (same model)  |

Keeping the **models** identical (`nomic-embed-text`, `llama3.1`) AND pointing at
the **same Postgres instance** is the key trick — it isolates "framework overhead"
from "model quality" and "store latency" in the comparison. The two pipelines live
in separate tables in the same DB so they can't step on each other.

## Setup

```bash
cd experiments/llamaindex

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Make sure Ollama is already running (the same one your Java app uses):

```bash
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.1
```

And make sure Postgres is up with pgvector installed — the same DB your Spring
Boot app connects to. The experiment reads these env vars (defaults match
`application.properties`):

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=ai_rag_assistant
export DB_USERNAME=postgres
export DB_PASSWORD=postgres
```

LlamaIndex will auto-create its own table (`data_llamaindex_chunks`) on first
ingest. Your Java `document_chunks` table is untouched.

## Run

Ingestion and querying are now separate steps so retrieval timings are honest.

```bash
# Step 1: ingest once (reads sample-data/, embeds, writes to pg).
# Re-run whenever you change sample-data/.
python -m src.run_pipeline --ingest

# Step 2: query as many times as you want — no embedding work, just retrieval.
python -m src.run_pipeline --mode vector
python -m src.run_pipeline --mode bm25
python -m src.run_pipeline --mode hybrid

# Ask a one-off question
python -m src.run_pipeline --mode hybrid --question "What engine does the M3 use?"
```

Drop your own PDFs or `.txt` / `.md` files into `sample-data/` and re-run
`--ingest` — `SimpleDirectoryReader` picks them up automatically.

## Resetting the experiment table

Wipe LlamaIndex's chunks without touching your Java data:

```sql
DROP TABLE IF EXISTS data_llamaindex_chunks;
```

Then re-run `--ingest`.

## What to look at

After a run, open `notes.md` — that's where the observations go. The interesting
numbers are:

1. **Lines of code to get a working pipeline.** Count ingestion + retrieval + answer
   in both projects. LlamaIndex will be dramatically shorter; that's the point.
2. **Where the seams are.** In the Java version you own the SQL, the rerank call,
   the chunk boundaries. In LlamaIndex you own a config dict. Which matters for
   which problems?
3. **Retrieval quality on the same questions.** Run the same 5–10 questions against
   both and eyeball the top-k chunks. Are they the same? Different? Why?

## Files

```
experiments/llamaindex/
├── README.md           # this file
├── requirements.txt    # pinned Python deps
├── notes.md            # observations, tradeoffs, interview talking points
├── sample-data/        # drop docs here (gitignored except for README)
└── src/
    ├── __init__.py
    ├── ingest.py       # loader + splitter + embedding + index
    ├── retrieve.py     # vector and hybrid retriever factories
    ├── answer.py       # query engine wiring
    └── run_pipeline.py # CLI entry point
```
