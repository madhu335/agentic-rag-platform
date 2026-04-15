"""
The graph nodes. Each function takes the current state and returns a partial
update dict. LangGraph merges the update into the state before the next node
runs.

Important design choice: this file uses raw psycopg2 and raw httpx, NOT
LangChain. The whole point of the experiment is to isolate what LangGraph
itself contributes. If we used LangChain wrappers here, the comparison
would be muddied — you wouldn't know whether a code-savings line came from
LangGraph or from LangChain.

Mapping back to Java RagAnswerService:
    retrieve()         -> PgVectorStore.hybridSearch (simplified to vector-only here;
                          full RRF fusion left as an exercise — see TODO)
    score_gate()       -> passesMinimumThreshold
    keyword_guard()    -> passesKeywordGuard
    filter_chunks()    -> selectUsableHits
    generate()         -> the prompt template + LlmRouter.complete()
    judge()            -> JudgeService.evaluate()
    refuse()           -> the FALLBACK constant + early-return paths
"""

import json
import os
import re

import httpx
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

from .state import Chunk, GraphState, JudgeResult

# --- thresholds — must match the Java side for a fair comparison ---
# RagAnswerService.java:16:  COSINE_LOW = 0.40 (was 0.55, relaxed for structured chunks)
COSINE_LOW = 0.40

FALLBACK_ANSWER = "I don't know based on the ingested documents."

# --- connection settings (read from .env via run_pipeline.py) ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "ai_rag_assistant")
DB_USER = os.getenv("DB_USERNAME", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1"


# =============================================================================
# Helpers — these would be private methods on a Java service class.
# =============================================================================

def _get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def _embed(text: str) -> list[float]:
    """Call Ollama's embedding endpoint. Equivalent to OllamaEmbeddingClient.java."""
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def _chat(prompt: str) -> str:
    """Call Ollama's chat endpoint. Equivalent to OllamaClient.complete()."""
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def _significant_terms(question: str) -> set[str]:
    """
    Extract meaningful terms from the question for the keyword guard.
    Drops common stopwords and short tokens. Same intent as the Java
    passesKeywordGuard implementation.
    """
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "should",
        "could", "may", "might", "can", "what", "which", "who", "whom",
        "when", "where", "why", "how", "of", "in", "on", "at", "to", "for",
        "with", "from", "by", "about", "as", "and", "or", "but", "if", "then",
        "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
        "they", "my", "your", "his", "her", "its", "our", "their",
    }
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", question.lower())
    return {w for w in words if w not in stopwords}


# =============================================================================
# NODES — these are what LangGraph wires together.
# Each takes a GraphState, returns a partial dict update.
# =============================================================================

def retrieve(state: GraphState) -> dict:
    """
    Vector retrieval against data_llamaindex_chunks.

    NOTE: Java's RagRetriever.HYBRID mode runs vector + BM25 + RRF fusion.
    For this experiment we use vector-only to keep the node code small —
    the focus is the orchestration layer, not the retrieval algorithm.
    Adding BM25 + fuseRRF as a separate node is a straightforward extension
    and would be the right next step if you wanted to make the comparison
    fully symmetric with Java's HYBRID mode.
    """
    question = state["question"]
    top_k = state.get("top_k", 5)

    query_vector = _embed(question)

    # The vector is stored as pgvector's `vector` type. We pass it as a
    # string literal because psycopg2 doesn't know about the pgvector type
    # natively. Same trick the LlamaIndex experiment uses internally.
    vector_literal = "[" + ",".join(str(x) for x in query_vector) + "]"

    sql = """
        SELECT
            node_id AS chunk_id,
            text,
            1 - (embedding <=> %s::vector) AS score
        FROM data_llamaindex_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with _get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (vector_literal, vector_literal, top_k))
            rows = cur.fetchall()

    chunks: list[Chunk] = [
        {
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "score": float(row["score"]),
        }
        for row in rows
    ]

    best_score = chunks[0]["score"] if chunks else None

    print(f"[retrieve] {len(chunks)} chunks, best_score={best_score}")
    return {"chunks": chunks, "best_score": best_score}


def score_gate(state: GraphState) -> dict:
    """
    Java equivalent: passesMinimumThreshold(hits, bestScore).
    Returns partial state. Routing to refuse vs continue is done by the
    conditional edge in graph.py — nodes themselves never decide control flow.
    """
    best_score = state.get("best_score")
    chunks = state.get("chunks", [])

    if not chunks or best_score is None or best_score < COSINE_LOW:
        reason = (
            f"score gate failed: best_score={best_score} < {COSINE_LOW}"
            if best_score is not None
            else "score gate failed: no chunks retrieved"
        )
        print(f"[score_gate] FAIL — {reason}")
        return {"refusal_reason": reason}

    print(f"[score_gate] PASS — best_score={best_score:.4f} >= {COSINE_LOW}")
    return {"refusal_reason": None}


def keyword_guard(state: GraphState) -> dict:
    """
    Java equivalent: passesKeywordGuard(question, bestScore, topText).

    Catches the failure mode where vector similarity is high but the chunk
    is actually about a different topic. Requires the top chunk to contain
    at least one significant term from the question.
    """
    chunks = state.get("chunks", [])
    if not chunks:
        return {"refusal_reason": "keyword guard failed: no chunks"}

    question = state["question"]
    top_text = chunks[0]["text"].lower()
    question_terms = _significant_terms(question)

    if not question_terms:
        # Question is all stopwords — let it through, nothing to check against.
        print("[keyword_guard] PASS (no significant terms in question)")
        return {"refusal_reason": None}

    matched = {term for term in question_terms if term in top_text}

    if not matched:
        reason = (
            f"keyword guard failed: top chunk shares no terms with question. "
            f"question_terms={sorted(question_terms)}"
        )
        print(f"[keyword_guard] FAIL — {reason}")
        return {"refusal_reason": reason}

    print(f"[keyword_guard] PASS — matched terms: {sorted(matched)}")
    return {"refusal_reason": None}


def filter_chunks(state: GraphState) -> dict:
    """
    Java equivalent: selectUsableHits — drops individual chunks below LOW
    so they don't dilute the LLM context window, even when the top chunk
    cleared the gate.
    """
    chunks = state.get("chunks", [])
    usable = [c for c in chunks if c["score"] >= COSINE_LOW]
    print(f"[filter_chunks] kept {len(usable)}/{len(chunks)} chunks above {COSINE_LOW}")
    return {"usable_chunks": usable}


def generate(state: GraphState) -> dict:
    """
    Java equivalent: the prompt template at RagAnswerService.java:312/347
    plus the LlmRouter.complete() call.

    The prompt deliberately tells the LLM to refuse if context is insufficient
    — that's the layer-4 prompt fallback from the four-layer refusal model.
    Layers 1-3 (score gate, keyword guard, chunk filter) have already run by
    the time we get here.
    """
    question = state["question"]
    usable = state.get("usable_chunks", [])

    if not usable:
        # Defensive — graph routing should have caught this, but never trust it.
        return {
            "answer": FALLBACK_ANSWER,
            "cited_chunk_ids": [],
        }

    context_block = "\n\n".join(
        f"[{c['chunk_id']}]\n{c['text']}" for c in usable
    )

    prompt = f"""You are a careful technical assistant. Answer the user's
question using ONLY the provided context. Cite chunks by their bracketed id
inline like [chunk_id].

Fallback:
- If the answer is not supported by the provided context, reply exactly:
{FALLBACK_ANSWER}

Context:
{context_block}

Question:
{question}
"""

    answer = _chat(prompt)
    cited = re.findall(r"\[([a-zA-Z0-9_:.\-]+)\]", answer)
    print(f"[generate] answer length={len(answer)} chars, cited={cited}")
    return {"answer": answer, "cited_chunk_ids": cited}


def judge_answer(state: GraphState) -> dict:
    """
    Java equivalent: JudgeService.evaluate(question, answer, chunks).

    LLM-as-judge: a separate Ollama call asks llama3.1 to rate the answer
    for grounded/correct/complete on the same context the generator saw.
    """
    answer = state.get("answer", "")
    question = state["question"]
    chunks = state.get("usable_chunks", [])

    if answer == FALLBACK_ANSWER or not chunks:
        # Refusals don't need to be judged — the system already said it didn't know.
        result: JudgeResult = {
            "grounded": False,
            "correct": False,
            "complete": False,
            "score": 0.0,
            "reason": "answer was the fallback string; no judgment needed",
        }
        return {"judge": result}

    context_block = "\n\n".join(c["text"] for c in chunks)

    judge_prompt = f"""You are evaluating whether a generated answer is grounded
in the provided context.

Context:
{context_block}

Question: {question}

Answer to evaluate:
{answer}

Reply with ONLY a JSON object in this exact shape:
{{"grounded": true|false, "correct": true|false, "complete": true|false, "score": 0.0-1.0, "reason": "..."}}
"""

    raw = _chat(judge_prompt)
    # Be lenient — strip code fences if the model added them.
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        parsed = json.loads(cleaned)
        result = {
            "grounded": bool(parsed.get("grounded", False)),
            "correct": bool(parsed.get("correct", False)),
            "complete": bool(parsed.get("complete", False)),
            "score": float(parsed.get("score", 0.0)),
            "reason": str(parsed.get("reason", "")),
        }
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # Java's JudgeService has a similar fallback when JSON parsing fails.
        print(f"[judge] JSON parse failed: {e}; raw={cleaned[:200]}")
        result = {
            "grounded": False,
            "correct": False,
            "complete": False,
            "score": 0.0,
            "reason": f"judge JSON parse failed: {e}",
        }

    print(f"[judge] grounded={result['grounded']} score={result['score']}")
    return {"judge": result}


def refuse(state: GraphState) -> dict:
    """
    Terminal node for any failed-gate path. Sets the final answer to the
    fallback string. The reason captured by whichever gate failed becomes
    the user-facing explanation if you want to surface it.
    """
    reason = state.get("refusal_reason", "unspecified")
    print(f"[refuse] returning fallback. reason={reason}")
    return {"final_answer": FALLBACK_ANSWER}


def finalize(state: GraphState) -> dict:
    """
    Terminal node for the happy path. Promotes `answer` to `final_answer`
    so the caller has one canonical field to read regardless of which
    branch the graph took.
    """
    return {"final_answer": state.get("answer", FALLBACK_ANSWER)}
