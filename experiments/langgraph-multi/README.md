# LangGraph Multi-Agent Experiment

Compares Java SupervisorAgent vs LangGraph multi-agent graph.

## Stack (UPDATED)

- Planner → vLLM
- Embeddings → Triton (`text_embedding`, 768d)
- Reranker → Triton (`cross_reranker`)
- Retrieval → Postgres (pgvector)
- Answer → vLLM
- Judge → vLLM (matches Java `judge.provider=vllm` default via
  `DefaultJudgeClient`)

Mirrors Java's `LlmRouter` + `JudgeRouter` with `llm.provider=vllm` and
`judge.provider=vllm`. Swap either side independently by changing the
client class — same pattern as on the Java side.

---

## Agents

- Research agent
- Vehicle agent
- Communication agent
- Supervisor (planner + router)

---

## Flow

plan → dispatch → route → dispatch → finalize

---

## Setup

cd experiments/langgraph-multi

python -m venv .venv  
.\.venv\Scripts\Activate.ps1  
pip install -r requirements.txt  

---

## Run

python -m src.run_pipeline  

Custom:

python -m src.run_pipeline --question "Tell me about BMW M3"  

Graph:

python -m src.run_pipeline --print-graph  

---

## Important Gotcha

Planner may output:

BMW M3 Competition

But DB expects:

bmw-m3-2025-competition

👉 You must normalize IDs.

---

## Triton Contract

Mirrors the Java `TritonEmbeddingClient` and `TritonRerankerClient`:

**Embedding (`text_embedding`):**
- Endpoint: `POST /v2/models/text_embedding/infer`
- Input: `TEXT`, datatype `BYTES`, shape `[batch, 1]`
- Output: `EMBEDDING`, shape `[batch, dim]` — reshape from the flat list

**Reranker (`cross_reranker`):**
- Endpoint: `POST /v2/models/cross_reranker/infer`
- Inputs: `QUERY` + `DOCUMENT`, each `BYTES`, each shape `[N, 1]` where
  `N = documents.size()` (query is duplicated `N` times)
- Output: `SCORE`, shape `[N, 1]` (may deserialize as nested list) or
  flat `[N]` — handle both

Both clients block up to 60s per call; batch size is bounded upstream.

---

## Scenarios

1. Research only  
2. Refusal  
3. Vehicle lookup  
4. Research + communication  

---

## Key Insight

LangGraph reduces:
- coordination complexity

BUT:
- agents logic is still fully manual

---

## Purpose

Show real multi-agent orchestration:
- planning
- routing
- state passing

And compare it directly to your Java implementation.