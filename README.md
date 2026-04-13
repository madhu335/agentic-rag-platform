# 🚀 Agentic RAG Platform (Hybrid Search + AI Workflows)

A production-grade agentic AI platform built with Spring Boot, pgvector, and LLMs, supporting hybrid retrieval (BM25 + vector), adaptive workflows, structured data ingestion, evaluation, and observability.

---

## 🔷 Overview

This platform enables:

- Agentic workflows (Planner → Executor → Tools)
- Hybrid RAG (BM25 + vector + RRF + re-ranking)
- Structured + unstructured data ingestion (PDF + vehicle data)
- Multi-step execution with session state and history tracking
- Evaluation using precision/recall and LLM-based judging
- Observability using OpenTelemetry and LangSmith

---

## 🧠 Architecture & Agentic Patterns

This platform follows a modular **Agentic AI architecture** with adaptive execution and strong separation of concerns.

---

### 🔄 Core Execution Flow

Planner → Executor → Tools → State → Evaluation → Re-plan

---

### 🧩 Implemented Patterns

#### 1. Planner–Executor Pattern
- Planner generates structured JSON workflows
- Executor interprets and executes steps sequentially
- Supports dynamic tool invocation

---

#### 2. Tool Abstraction Pattern
- Each capability is encapsulated as a Tool + Executor
- Plug-and-play extensibility

---

#### 3. ReAct (Reasoning + Acting)
- LLM plans → executes → observes → re-plans
- Enables multi-step reasoning with feedback loop

---

#### 4. State Management Pattern
- Session-based execution tracking
- Maintains:
    - Step history
    - Intermediate outputs
    - Execution status
- Supports `/agent/continue`

---

#### 5. Divergence Detection Pattern
- Detects deviation from plan execution
- Triggers re-planning on:
    - Invalid outputs
    - Missing data
    - Execution mismatch

---

#### 6. Hybrid Retrieval Pattern
- Combines BM25 + vector search
- Uses RRF + re-ranking for optimal results

---

#### 7. Chunking Strategy Pattern
- Vehicle data chunked by child objects (engine, specs, features)
- PDF data chunked semantically

---

#### 8. Evaluation Pattern
- Precision / Recall measurement
- LLM-based answer validation

---

#### 9. Observability Pattern
- OpenTelemetry tracing
- LangSmith for execution tracking

---

#### 10. Prompt-Orchestrated Planning
- Custom prompts for:
    - Planning
    - Re-planning
- Ensures structured agent behavior

---

## 🔍 Hybrid Retrieval System (BM25 + Vector + RRF + Re-ranking)

A production-grade retrieval pipeline combining lexical and semantic search.

---

### ⚙️ Retrieval Pipeline

User Query  
→ BM25 Search  
→ Vector Search  
→ RRF Fusion  
→ Re-ranking  
→ Final Context

---

### 🔑 Components

#### BM25 (Keyword Search)
- Handles structured queries
- Strong for exact matches (vehicle specs)

---

#### Vector Search (Embeddings)
- Semantic understanding
- Uses cosine similarity

---

#### Reciprocal Rank Fusion (RRF)
- Combines BM25 + vector rankings
- Balances relevance across methods

---

#### Re-ranking Layer
- Second-pass ranking
- Improves top-k precision

---

#### Chunking Strategy

**Vehicle Data:**
- Engine
- Performance
- Features
- Specs

**Documents:**
- Semantic chunking

---

#### Data Preprocessing
- Cleaned and normalized vehicle data
- Removed noise before embedding

---

### 📊 Evaluation

- Precision %
- Recall %
- LLM-based correctness scoring

---

### 💡 Outcome

- Improved answer accuracy
- Reduced hallucinations
- Better grounding across query types

---

## 🤖 Agentic Workflows & Tooling System

The system executes LLM-generated plans through modular tools.

---

### 🔄 Execution Flow

User Request  
→ Planner  
→ Executor  
→ Tools  
→ State Store  
→ Response

---

### 🧠 Planner

- Generates structured JSON workflows
- Supports multi-step reasoning

---

### ⚙️ Executor

- Executes planned steps
- Handles:
    - Routing
    - Errors
    - Re-planning
    - State updates

---

### 🧩 Tooling System

#### 🚗 Vehicle Tools
- FetchVehicleSpecsTool
- CompareVehiclesTool
- GenerateVehicleSummaryTool
- EnrichVehicleDataTool

---

#### 📩 Communication Tools
- EmailTool / DraftEmailToolExecutor
- SendEmailToolExecutor
- SmsTool / SendSmsToolExecutor
- ShortenEmailToolExecutor

---

#### 🔍 Research Tool
- ResearchTool (Hybrid retrieval powered)

---

### 🧠 State Management

Tracks:
- Step history
- Intermediate outputs
- Execution status

Supports:
- `/agent/start`
- `/agent/continue`

---

### 👤 Human-in-the-Loop (HITL)

Selective validation for high-impact actions:

- Email workflows (draft → review → send)
- SMS workflows (compose → confirm → send)

---

#### 🔄 HITL Flow

Planner → Compose → Pause  
→ User Review  
→ `/agent/continue`  
→ Send

---

#### 💡 Design Decision

- Applied only where needed
- Avoids slowing down retrieval flows

---

### 🔁 Re-planning & Adaptation

Triggered when:
- Execution fails
- Data missing
- Divergence detected

---

### 🚨 Error Handling

- Safe fallbacks
- Retry mechanisms
- State-aware recovery

---

## 📊 Evaluation & Observability

---

### 📈 Evaluation

- Precision / Recall tracking
- LLM-based answer validation
- Retrieval quality verification

---

### 🔍 Observability

- OpenTelemetry tracing
- LangSmith integration:
    - Execution traces
    - Debugging
    - Workflow monitoring

---

## 🛠️ Tech Stack

- Java / Spring Boot
- PostgreSQL + pgvector
- BM25 + Vector Search
- OpenTelemetry
- LangSmith
- LLM APIs

---

## 🚀 Key Highlights

- Production-grade hybrid retrieval system
- Fully agentic architecture with re-planning
- Real-world tool execution (vehicle + communication)
- Selective human-in-the-loop safety
- End-to-end observability and evaluation

---

## 📌 Future Enhancements

- CMS integration
- Multi-agent orchestration
- Streaming responses
- Advanced re-ranking models
- Domain expansion beyond vehicles

---