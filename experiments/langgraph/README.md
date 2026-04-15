# LangGraph Experiment

A side-by-side experiment that ports the orchestration logic from the Java
`RagAnswerService` to a LangGraph state machine. Same Postgres table as the
LlamaIndex experiment (`data_llamaindex_chunks`), same models (Ollama
`llama3.1` + `nomic-embed-text`), so all three pipelines — Java, LlamaIndex,
LangGraph — operate on identical data and can be compared head-to-head.

## What this is and isn't

**LangGraph is a workflow/orchestration framework, not a RAG framework.** It
has no concept of documents, chunks, vectors, or embeddings. It gives you a
graph runtime — you write Python functions as nodes, wire them with edges,
and LangGraph handles state flow, conditional branching, and persistence.

That means this experiment is **not** comparable to the LlamaIndex one in
the same way the LlamaIndex one was comparable to Java. LlamaIndex replaced
the retrieval-and-answer layer. LangGraph replaces the *orchestration* layer
on top of retrieval and answer code that you still write yourself.

So the comparison shape is:

```
Java RagAnswerService
└── orchestration logic (gates, refusal paths, judge, prompt building)
    └── retrieval (PgVectorStore.hybridSearch, fuseRRF)
        └── LLM client (LlmRouter, OllamaClient)

LangGraph experiment (this folder)
└── orchestration logic — implemented as a LangGraph state machine
    └── retrieval — raw psycopg2 against the same pgvector table
        └── LLM client — raw HTTP requests to Ollama
```

The retrieval and LLM client layers are deliberately **hand-written, not
LangChain**. Using LangChain would make the experiment about "LangChain +
LangGraph together" instead of "what does LangGraph specifically give me?",
which is the question worth answering.

## What it mirrors from the Java side

The graph implements the same four-layer refusal architecture documented in
the LlamaIndex experiment notes:

| Java (`RagAnswerService`) | LangGraph node |
|---|---|
| `vectorStore.hybridSearch(...)` | `retrieve` |
| `passesMinimumThreshold` (score gate) | `score_gate` (conditional edge) |
| `passesKeywordGuard` | `keyword_guard` (conditional edge) |
| `selectUsableHits` (chunk filter) | `filter_chunks` |
| Prompt template + LLM call | `generate` |
| `JudgeService` | `judge` (conditional edge) |
| The `FALLBACK` constant + early returns | `refuse` (terminal node) |

The graph shape:

```
                      START
                        │
                        ▼
                   ┌─────────┐
                   │retrieve │
                   └────┬────┘
                        │
                        ▼
                  ┌───────────┐
                  │score_gate │──fail──┐
                  └─────┬─────┘        │
                        │ pass         │
                        ▼              │
                ┌───────────────┐      │
                │ keyword_guard │─fail─┤
                └───────┬───────┘      │
                        │ pass         │
                        ▼              │
                 ┌──────────────┐      │
                 │filter_chunks │      │
                 └──────┬───────┘      │
                        │              │
                        ▼              │
                  ┌─────────┐          │
                  │generate │          │
                  └────┬────┘          │
                       │               │
                       ▼               │
                   ┌───────┐           │
                   │ judge │──ungrnd───┤
                   └───┬───┘           │
                       │ grounded      │
                       ▼               ▼
                      END         ┌────────┐
                                  │ refuse │
                                  └────┬───┘
                                       │
                                       ▼
                                      END
```

Every node reads and writes a single shared state dict (see `state.py`).

## Setup

```powershell
cd experiments/langgraph

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Make sure the same prerequisites from the LlamaIndex experiment are running:

- Ollama with `llama3.1` and `nomic-embed-text` pulled
- Postgres (Docker Compose stack) on `localhost:5432`
- `data_llamaindex_chunks` populated (run `python -m src.run_pipeline --ingest`
  from the LlamaIndex experiment first if it's empty)

Copy `.env.example` to `.env` and adjust if your DB credentials differ from
the defaults.

## Run

```powershell
# Default questions (from the LlamaIndex experiment for direct comparison)
python -m src.run_pipeline

# Custom question
python -m src.run_pipeline --question "How should I configure heap size for a Spring Boot app in production?"

# Print the graph as a Mermaid diagram and exit (no LLM calls)
python -m src.run_pipeline --print-graph
```

## What to look at after running

The interesting comparison points for `notes.md`:

1. **Lines of Python in `nodes.py`** vs lines of Java in `RagAnswerService.java`.
   The LlamaIndex experiment found the Java answer layer was 596 LOC. How
   much of that is the orchestration vs the retrieval and LLM client wrapping?

2. **What LangGraph gives you for free** that you wrote by hand in Java:
   - Declarative graph structure (no manual `if` chains for routing)
   - State accumulation across nodes (no manual passing of intermediate values)
   - Free Mermaid visualization of the graph
   - LangSmith integration if you wire it up

3. **What you still write yourself** (and would have to write in any framework):
   - The retrieval SQL
   - The score thresholds and gating predicates
   - The prompt templates
   - The judge logic
   - The HTTP calls to Ollama

4. **The same question on the heap-sizing test from the LlamaIndex experiment.**
   Does the LangGraph version, with refusal logic ported faithfully, refuse
   cleanly the way Java did — or does it hallucinate the way LlamaIndex did?
   Prediction: it refuses, because the refusal logic is structural, not
   framework-magic. The point of the experiment is to confirm that the
   refusal behavior comes from *the logic*, not from *being written in Java*.

## Future extensions (left as TODOs in code)

- **Checkpointer**: LangGraph's killer feature is resumable workflows via
  `SqliteSaver` or `PostgresSaver`. Not used here because the workflow
  finishes in seconds. Worth adding to demonstrate the persistence story.
- **Human-in-the-loop pause**: insert a `interrupt_before=["judge"]` to pause
  for manual review of the answer before the judge runs.
- **Streaming**: stream intermediate state updates to the caller instead of
  printing the final result.

## Files

```
experiments/langgraph/
├── README.md              # this file
├── notes.md               # observations and findings (start empty, fill after runs)
├── requirements.txt       # langgraph + psycopg2 + httpx
├── .env.example           # template
├── .gitignore             # secrets and venv
└── src/
    ├── __init__.py
    ├── state.py           # the typed dict that flows through the graph
    ├── nodes.py           # the 7 node functions (retrieve, gates, generate, judge, refuse)
    ├── graph.py           # builds the StateGraph and wires the edges
    └── run_pipeline.py    # CLI entry point
```
