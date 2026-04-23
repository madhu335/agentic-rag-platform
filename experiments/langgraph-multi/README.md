# LangGraph Multi-Agent Experiment

Compares Java SupervisorAgent vs LangGraph multi-agent graph.

## Stack (UPDATED)

- Planner → vLLM
- Embeddings → Triton
- Retrieval → Postgres (pgvector)
- Answer → vLLM

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

Input:
- TEXT
- BYTES
- shape [batch,1]

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