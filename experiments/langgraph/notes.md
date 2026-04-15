# Notes: Custom Java vs LangGraph (orchestration layer)

Lab notebook for porting `RagAnswerService` orchestration to LangGraph.
Fill in after running both pipelines.

This is **not** a RAG-framework comparison like the LlamaIndex experiment.
LangGraph is a workflow engine, not a RAG framework. The retrieval and LLM
client code in `nodes.py` is hand-written Python — using LangChain wrappers
would have hidden what LangGraph specifically contributes.

## Setup

See `README.md`. Reuses `data_llamaindex_chunks` from the LlamaIndex
experiment, so no separate ingestion is needed.

## What LangGraph contributes (versus what you write yourself)

| Concern | Java (`RagAnswerService`) | LangGraph (this experiment) |
|---|---|---|
| Routing logic | `if`/`return` chains in one method | Conditional edge functions, declarative |
| State accumulation | Builder pattern manually threaded | Framework-merged dict |
| Observability | Custom `TraceHelper` | Hooks for LangSmith if wired |
| Persistence | `AgentSessionEntity` + Spring Data | `Checkpointer` (not used in this experiment) |
| Graph visualization | None | `graph.get_graph().draw_mermaid()` for free |
| Retrieval logic | You write it | You still write it |
| LLM call wrappers | You write it | You still write it |
| Prompt templates | You write it | You still write it |
| Score thresholds & gates | You write it | You still write it |
| Judge logic | You write it | You still write it |

The pattern: LangGraph commoditizes the *structure* of the workflow.
Everything that's actually domain-specific — what each node does, what
gets retrieved, how prompts are shaped, what counts as "grounded" — is
still your code.

## LOC comparison (fill after running cloc)

```powershell
cloc experiments/langgraph/src/
```

| Stage | Java | LlamaIndex (Py) | LangGraph (Py) |
|---|---:|---:|---:|
| Ingestion | 230 | ~70 | n/a (reused LI table) |
| Retrieval | 657 | ~70 | (one node in nodes.py) |
| Answer (orchestration + LLM call) | 596 | ~30 | (rest of nodes.py + graph.py) |
| **Total relevant** | TBD | TBD | TBD |

## Live experiment results (fill after runs)

Same two questions as the LlamaIndex experiment for direct comparison:

### Q1: "How should I configure heap size for a Spring Boot app in production?"

| Pipeline | Answer | Refused at | Notes |
|---|---|---|---|
| Java (HYBRID) | ✅ Cited [5,6] | n/a | (when scoped to jvm-flags-guide) |
| LlamaIndex (RRF hybrid) | ❌ Hallucinated | none | "CPU core's worth of memory" |
| LangGraph (this) | ✅ Correct, cited, `-Xms`/`-Xmx` | n/a (all gates passed) | judge: grounded=true, score=0.90 |

Prediction: LangGraph refuses cleanly via the score gate or keyword guard,
because the refusal logic is structural and was ported faithfully from Java.
If it doesn't refuse, the bug is in the port, not the framework.

### Q2: "What is Spring Boot auto-configuration?"

| Pipeline | Answer | Notes |
|---|---|---|
| Java (HYBRID) | ✅ Cited [1] (cosine 0.98) | |
| LlamaIndex (RRF hybrid) | ✅ Richest answer | META-INF path, Kafka beans |
| LangGraph (this) | ✅ Correct, cited, judge score 1.0 | |

## Findings (fill after runs)

1. **The refusal port worked — but wasn't needed.**
   Both questions passed all three gates (score gate, keyword guard, judge).
   The LangGraph version gave correct, grounded answers on both questions —
   including Q1 (heap sizing), which LlamaIndex hallucinated on.

   The reason it didn't hallucinate where LlamaIndex did is NOT the refusal
   logic — it's the retrieval mode. LangGraph used vector-only (cosine
   similarity), which kept the `@Repository` chunk at rank 4 where it
   belongs. LlamaIndex's RRF hybrid promoted it to rank 2 because BM25
   gave it a high lexical-overlap score. The polluted rank-2 chunk is what
   caused the hallucination.

   **Key insight: the hallucination on the heap question was a retrieval
   problem (RRF fusion), not an orchestration problem (missing refusal
   logic).** The refusal logic is still valuable as defense-in-depth —
   it would have caught the hallucination if retrieval had been worse —
   but in this specific case, cleaner retrieval was sufficient.

2. **What did the conditional edge functions actually buy you?**
   - Could you have written this as a single procedural Python function
     with `if`/`return` and gotten the same behavior in fewer lines?
     (Answer: probably yes for this small a graph.)
   - At what graph size does declarative routing start being worth the overhead?

3. **What did `graph.get_graph().draw_mermaid()` produce?**
   - Paste the output here. If it's a clear visualization of the workflow,
     that's a real productivity win for explaining the system to others —
     LangGraph gives you "drag-and-drop documentation" for free.

4. **What would adding a checkpointer change?**
   - Right now the workflow runs to completion in seconds. If we added
     `SqliteSaver`, what would change in `graph.py`? (Answer: one line.)
     What new capabilities would you unlock? (Answer: pause/resume, time
     travel, replay from any node.)
   
### Refusal test: "What is Kubernetes pod autoscaling?"

Question deliberately outside the corpus — neither PDF mentions Kubernetes.

- `[retrieve]` returned 5 chunks, best_score=0.6290 — **deceptively high.**
  Embeddings encode "technical infrastructure question" and find Spring
  Boot content that shares that semantic category. Score alone looks fine.
- `[score_gate]` PASSED (0.6290 >= 0.40). **Layer 1 would have missed this.**
- `[keyword_guard]` FAILED — extracted `['autoscaling', 'kubernetes', 'pod']`,
  none found in top chunk text. **Layer 2 caught it.**
- `[refuse]` returned "I don't know based on the ingested documents."
- No LLM call was made. Total time: 3.14s (embedding only).

This confirms the defense-in-depth model: the score gate alone is not
sufficient. The keyword guard catches the failure mode where embeddings
produce high-confidence false matches across semantic categories.
Without the keyword guard (i.e. LlamaIndex's default pipeline), the LLM
would have received Spring Boot chunks and synthesized a plausible-sounding
but completely fabricated Kubernetes answer.

## When to pick which

- **LangGraph** when: workflow has many steps, branching, loops, or
  pause/resume requirements; team wants free visualization and tracing;
  multiple stakeholders need to read the workflow as a diagram.
- **Custom Java orchestration** when: workflow is small enough to fit in
  one method; you want type-safe state and dependency injection without a
  framework's opinions; persistence and observability are already handled
  by your existing stack (Spring Data + your TraceHelper).

## Future extensions

- [ ] Add a `SqliteSaver` checkpointer and demonstrate pause/resume
- [ ] Insert `interrupt_before=["judge"]` for human-in-the-loop demo
- [ ] Stream intermediate state updates instead of printing the final result
- [ ] Port `AgentSessionRunner` (the multi-step agent flow) instead of
      just `RagAnswerService` — closer to the JD's "long-running workflows"
      keyword
- [ ] Add the BM25 + RRF fusion as a separate node so the retrieval is
      symmetric with Java's HYBRID mode
