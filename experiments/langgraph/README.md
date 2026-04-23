# LangGraph Experiment

Ports your Java RAG orchestration into a LangGraph state machine.

## Stack (UPDATED)

- Embeddings → Triton
- LLM → vLLM
- DB → Postgres (pgvector)

---

## What LangGraph Does

LangGraph = orchestration only

It does NOT handle:
- embeddings
- retrieval
- chunking

You still write those yourself.

---

## Flow

START → retrieve → score_gate → keyword_guard → filter → generate → judge → END

Failures route to:

→ refuse

---

## Setup

cd experiments/langgraph

python -m venv .venv  
.\.venv\Scripts\Activate.ps1  
pip install -r requirements.txt  

---

## Run

python -m src.run_pipeline  

Custom:

python -m src.run_pipeline --question "Explain Spring Boot auto-config"

Graph:

python -m src.run_pipeline --print-graph  

---

## Behavior Notes

### Score Threshold

Vector scores are small:

0.01 – 0.05 range

So thresholds must be LOW.

---

## Purpose

Compare:
- Java orchestration vs graph-based orchestration
- manual flow vs declarative graph

Key insight:
LangGraph simplifies control flow, NOT domain logic.