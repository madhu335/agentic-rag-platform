# 🚀 Agentic RAG Platform (Hybrid Search + AI Workflows)

A production-style agentic AI platform built with Spring Boot, pgvector, and LLMs, supporting hybrid retrieval (BM25 + vector), adaptive workflows, structured data ingestion, evaluation, and observability.

## 🔹 Overview

This platform enables:

- Agentic workflows (Planner → Executor → Tools)
- Hybrid RAG (BM25 + vector + RRF + re-ranking)
- Structured + unstructured data ingestion (PDF + vehicle data)
- Multi-step execution with session state and history tracking
- Evaluation using precision/recall and LLM-based judging
- Observability using OpenTelemetry and LangSmith