# LangGraph Multi-Agent Experiment

Side-by-side comparison of the Java `SupervisorAgent` (multi-agent orchestration)
with a LangGraph equivalent using the supervisor pattern.

## What this compares

| Aspect | Java (`SupervisorAgent`) | LangGraph (this experiment) |
|---|---|---|
| Supervisor loop | `for (delegation : delegations)` with `switch` | Graph cycle: `dispatch → route → dispatch` |
| Routing | `switch (delegation.agent())` | Conditional edge function |
| State passing | Manual `lastContentResult` tracking | `last_content` key in graph state |
| Sub-agents | `@Component` Spring beans | Plain Python functions |
| Planner | `SupervisorPlanner` with ChatClient | Same prompt via raw httpx |
| Persistence | `AgentStateStore` → Postgres | Not implemented (add checkpointer for equivalent) |

## Key finding (predicted)

The supervisor graph in LangGraph is ~15 lines of wiring (graph.py `build_graph`).
The equivalent Java code in `SupervisorAgent.handle()` is ~100 lines. But the
sub-agent logic (retrieval, LLM calls, keyword guard, judge) is the same size in
both — you still write it yourself. LangGraph commoditizes the **coordination
structure**, not the **domain logic**.

## Setup

```powershell
cd experiments/langgraph-multi
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env with your DB password
```

Reuses `data_llamaindex_chunks` table from the LlamaIndex experiment.

## Run

```powershell
# Print the supervisor graph as Mermaid
python -m src.run_pipeline --print-graph

# Run all 4 default scenarios
python -m src.run_pipeline

# Custom question
python -m src.run_pipeline --question "What is Spring Boot auto-config?" --doc-id spring-boot-qa

# Vehicle question
python -m src.run_pipeline --question "Tell me about the BMW M3" --doc-id bmw-m3-2025-competition
```

## Default scenarios

1. **Research only** — "What is Spring Boot auto-configuration?"
   Supervisor delegates to research agent only. No email.

2. **Research refusal** — "What is Kubernetes pod autoscaling?"
   Research agent should refuse via keyword guard. No LLM call for generation.

3. **Vehicle fetch + summarize** — "Tell me about the BMW M3 Competition"
   Supervisor delegates to vehicle agent. Fetch specs → LLM summary.

4. **Research + email** — "Research Spring Boot auto-config and email it"
   Supervisor delegates: research → communication. Tests content passing.

## Files

```
experiments/langgraph-multi/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
└── src/
    ├── __init__.py
    ├── state.py       # MultiAgentState, Delegation, SubAgentResult
    ├── agents.py      # research_agent, vehicle_agent, communication_agent
    ├── graph.py       # Supervisor graph: plan → dispatch loop → finalize
    └── run_pipeline.py
```
