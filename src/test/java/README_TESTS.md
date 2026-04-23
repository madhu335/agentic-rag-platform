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

# 💥 Summary

This system is not just a RAG demo — it is a **multi-domain, agentic, streaming AI platform** with:

- clean abstraction (`docType`)
- scalable retrieval
- UI-ready outputs
- real-time streaming

