# LlamaIndex Experiment

A side-by-side comparison harness for your custom Java RAG pipeline vs. a LlamaIndex
implementation of the same four stages:

data -> ingestion -> retrieval -> answer

---

## Why this exists

This experiment is NOT about replacing the Java pipeline.

It is about:
- understanding what LlamaIndex abstracts
- identifying what still requires custom engineering
- evaluating tradeoffs in real infrastructure (pgvector, Triton, vLLM)

This gives you concrete answers for:
"Have you worked with LlamaIndex?" → YES (with real infra, not toy examples)

---

## Current Stack (UPDATED)

| Layer | System |
|------|--------|
| Embeddings | Triton `text_embedding` (`http://localhost:8000`) |
| Reranker | Triton `cross_reranker` (`http://localhost:8000`) |
| LLM | vLLM `meta-llama/Meta-Llama-3.1-8B-Instruct` (`http://localhost:8001/v1`) |
| Judge | vLLM (Java side: `DefaultJudgeClient` wrapping `LlmRouter`) |
| DB | Postgres (`ai_rag_assistant`) |
| Table | `data_llamaindex_chunks` |

The Java side reaches these through `LlmRouter` / `JudgeRouter` /
`TritonEmbeddingAdapter` / `TritonRerankerClient`. This experiment calls
vLLM and Triton directly with `httpx` so the comparison is framework vs
framework, not wrapper vs wrapper.

---

## What this mirrors from Java

| Stage | Java Pipeline | LlamaIndex |
|------|-------------|-----------|
| Ingestion | PdfExtractor + Chunker | SimpleDirectoryReader + SentenceSplitter |
| Embedding | Triton client | custom Triton adapter |
| Store | pgvector (`document_chunks`) | pgvector (`data_llamaindex_chunks`) |
| Retrieval | VECTOR / BM25 / HYBRID | retrievers + fusion |
| Answer | RagAnswerService + vLLM | direct httpx → vLLM |

---

## Setup

```powershell
cd experiments/llamaindex

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt