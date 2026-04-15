"""
The state object that flows through every node in the graph.

LangGraph passes a single dict through the workflow. Each node receives the
current state, returns a partial update (just the keys it touches), and
LangGraph merges the update into the master state before the next node runs.

This is roughly equivalent to the Java `RagAnswerService` collecting fields
into an `AskResponse` builder as it goes — except here the accumulation
happens in the framework, not in code.

Why TypedDict and not a plain dict?
- LangGraph reads the type hints at graph-build time to validate that
  nodes return only declared keys. Catches typos like `chunks_filtered`
  vs `filtered_chunks` at startup instead of at runtime.
- IDE autocompletion in PyCharm — when a node receives `state: GraphState`,
  PyCharm will autocomplete the field names.
"""

from typing import TypedDict, Optional


class Chunk(TypedDict):
    """A single retrieved chunk from pgvector. Mirrors Java's SearchHit."""

    chunk_id: str
    text: str
    score: float  # cosine similarity from pgvector (1 - cosine distance)


class JudgeResult(TypedDict):
    """Mirrors Java's JudgeResult — the LLM-as-judge evaluation."""

    grounded: bool
    correct: bool
    complete: bool
    score: float
    reason: str


class GraphState(TypedDict, total=False):
    """
    The shared state. `total=False` means every field is optional —
    nodes only populate the keys they're responsible for, and the absence
    of a key means "not yet computed."

    Field naming follows the Java side where reasonable so a side-by-side
    code reading is straightforward.
    """

    # --- inputs (set once at graph entry) ---
    question: str
    top_k: int

    # --- retrieve node ---
    chunks: list[Chunk]            # raw retrieved hits
    best_score: Optional[float]    # chunks[0].score, or None if empty

    # --- score_gate / keyword_guard / filter_chunks ---
    usable_chunks: list[Chunk]     # survivors of selectUsableHits filter
    refusal_reason: Optional[str]  # set by gate failures, drives the refuse node

    # --- generate node ---
    answer: Optional[str]          # the LLM's grounded response
    cited_chunk_ids: list[str]     # chunk_ids the LLM referenced

    # --- judge node ---
    judge: Optional[JudgeResult]

    # --- terminal ---
    final_answer: str              # what the caller actually receives
                                   # (either `answer` or the refusal string)
