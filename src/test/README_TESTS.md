# 🧪 README_TESTS.md

This document describes all supported test flows for the Agentic RAG Platform.

---

# 🔹 1. Document / PDF (RAG)

```json
{
  "docType": "document",
  "docId": "spring-boot-guide",
  "question": "What is dependency injection?",
  "topK": 5
}
```

### ✅ Covers:
- PDF ingestion + chunking
- pgvector retrieval
- citations
- fallback handling

---

# 🚗 2. Vehicle (Single Vehicle)

```json
{
  "docType": "vehicle",
  "docId": "bmw-m3-2025",
  "question": "What is horsepower and 0-60?",
  "topK": 5
}
```

### ✅ Covers:
- single entity retrieval
- detailed specs
- grounded answers

---

# 🚗 3. Vehicle (Fleet / Global Search)

```json
{
  "docType": "vehicle",
  "docId": "vehicles",
  "question": "Show me sporty EVs with strong performance",
  "topK": 5
}
```

### ✅ Covers:
- global vehicle search
- hybrid retrieval (summary + chunks)
- vehicle cards (UI-ready)

---

# 📰 4. Article (Single Article)

```json
{
  "docType": "article",
  "docId": "motortrend-bmw-m3-review",
  "question": "What does MotorTrend say about ride quality?",
  "topK": 5
}
```

### ✅ Covers:
- CMS article retrieval
- section-level grounding
- ratings and summaries

---

# 📰 5. Article (Global Search)

```json
{
  "docType": "article",
  "docId": "articles",
  "question": "Best sports sedans?",
  "topK": 5
}
```

### ✅ Covers:
- cross-article retrieval
- comparisons
- aggregation

---

# ⚡ 6. Streaming API (SSE)

### Endpoint:
```
POST /ask/stream
```

### Example:
```json
{
  "docType": "vehicle",
  "docId": "vehicles",
  "question": "Show me sporty EVs",
  "topK": 5
}
```

### Events:
- start
- status
- sources
- token
- done
- error

### ✅ Covers:
- real-time token streaming
- UI integration
- cards + answer streaming

---

# 🤖 7. Multi-Agent Flow

### Flow:
```
Planner → Executor → Tools
```

### ✅ Covers:
- orchestration
- research workflows
- multi-step reasoning
- session inspection

---

# 🧠 Architecture Highlights

- Unified `docType` abstraction across all domains
- Hybrid retrieval:
  - Vector (pgvector)
  - BM25
  - Re-ranking
- Streaming-first architecture (SSE)
- Structured UI responses (vehicle cards)
- Extensible platform (supports new domains easily)

---

# 🔥 Canonical Request Patterns

### Document
```json
{ "docType": "document", "docId": "doc-id" }
```

### Vehicle (Single)
```json
{ "docType": "vehicle", "docId": "vehicle-id" }
```

### Vehicle (Fleet)
```json
{ "docType": "vehicle", "docId": "vehicles" }
```

### Article (Single)
```json
{ "docType": "article", "docId": "article-id" }
```

### Article (Global)
```json
{ "docType": "article", "docId": "articles" }
```

---

# ✅ Recommendation

- Use `/ask` and `/ask/stream` as the primary API contract
- Keep multi-agent flows as orchestration layer
- Gradually migrate all test files to this structure

---

# 🔌 Provider Overrides for Tests

All test flows above go through the same `LlmRouter` / `JudgeRouter` /
`TritonEmbeddingAdapter` that the production code uses, so you can swap
the answer LLM, judge LLM, or embedding backend without touching any
test file — just override the Spring property when you start the app.

### Defaults (from `application.yml`)

| Key | Default | Clients |
|---|---|---|
| `llm.provider` | `vllm` | `vllm` · `triton-vllm` · `openai` · `claude` · `ollama` |
| `judge.provider` | `vllm` | `claude` · `openai` · `default` / `vllm` / `ollama` |
| `embedding.provider` | `triton` | `triton` · `ollama` |

`judge.provider=default`, `vllm`, or `ollama` all resolve to
`DefaultJudgeClient`, which wraps whatever `LlmRouter` currently returns
— so the judge inherits `llm.provider` unless you set `judge.provider`
to `claude` or `openai` explicitly.

### Examples

```bash
# Local vLLM answer, Claude judge (requires ANTHROPIC_API_KEY)
./mvnw spring-boot:run \
  -Dspring-boot.run.arguments="--llm.provider=vllm --judge.provider=claude"

# OpenAI answer + OpenAI judge (requires OPENAI_API_KEY)
./mvnw spring-boot:run \
  -Dspring-boot.run.arguments="--llm.provider=openai --judge.provider=openai"

# Triton-served vLLM (v2 infer) with default judge
./mvnw spring-boot:run \
  -Dspring-boot.run.arguments="--llm.provider=triton-vllm"
```

The `.http` request bodies above (`/ask`, `/ask/stream`, `/agent/multi`)
don't change — the router dispatches behind `AskController` and the
agent sub-agents, so the same JSON payload works against every provider
combination.

### What to check in each test mode

| Mode | Things to verify |
|---|---|
| `vllm` | Streaming tokens arrive on `/ask/stream`; `VllmClient` SSE parse is clean |
| `triton-vllm` | Non-streaming only; `/ask/stream` throws `UnsupportedOperationException` |
| `claude` | `JudgeResult.score` normalizes 1–10 outputs to 0.0–1.0 cleanly |
| `openai` | `openai.base-url` points where you expect (defaults to vLLM on `:8001`) |
| `embedding.provider=triton` | Ingestion batches hit `/v2/models/text_embedding/infer` in sub-batches |

---

# 💥 Summary

This system is not just a RAG demo — it is a **multi-domain, agentic, streaming AI platform** with:

- clean abstraction (`docType`)
- scalable retrieval
- UI-ready outputs
- real-time streaming

