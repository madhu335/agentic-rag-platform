"""
Sub-agent implementations.

Each sub-agent is a plain Python function — not a LangGraph node. They're
called BY nodes in the supervisor graph. This separation is deliberate:

    LangGraph controls:  routing, state merging, loop iteration
    Python functions:    the actual domain logic (retrieval, LLM calls, etc.)

This mirrors the Java design where SupervisorAgent (the coordinator) calls
ResearchSubAgent/VehicleSubAgent/CommunicationSubAgent (the workers).

Same as the single-agent experiment: raw psycopg2 + httpx, no LangChain.
"""

import json
import os
import re

import httpx
import psycopg2
from psycopg2.extras import RealDictCursor

from .state import SubAgentResult

# --- connection settings ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "ai_rag_assistant")
DB_USER = os.getenv("DB_USERNAME", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1"

COSINE_LOW = 0.40
FALLBACK = "I don't know based on the ingested documents."


# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════

def _get_db():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
    )


def _embed(text: str) -> list[float]:
    r = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60.0,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def _chat(prompt: str) -> str:
    r = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": False},
        timeout=120.0,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


def _vector_search(doc_id: str, query_vec: list[float], top_k: int) -> list[dict]:
    vec_literal = "[" + ",".join(str(x) for x in query_vec) + "]"
    sql = """
        SELECT node_id AS chunk_id, text,
               1 - (embedding <=> %s::vector) AS score
        FROM data_llamaindex_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with _get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (vec_literal, vec_literal, top_k))
            return [dict(r) for r in cur.fetchall()]


def _significant_terms(question: str) -> set[str]:
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "should", "could", "can", "what", "which", "who", "when",
        "where", "why", "how", "of", "in", "on", "at", "to", "for",
        "with", "from", "by", "about", "as", "and", "or", "but",
        "if", "this", "that", "i", "you", "he", "she", "it", "we",
        "they", "my", "your", "his", "her", "its", "our", "their",
        "tell", "me", "show",
    }
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", question.lower())
    return {w for w in words if w not in stopwords}


# ═══════════════════════════════════════════════════════════════════════
# RESEARCH SUB-AGENT
# ═══════════════════════════════════════════════════════════════════════

def research_agent(task: str, doc_id: str | None, top_k: int) -> SubAgentResult:
    """
    Java equivalent: ResearchSubAgent.execute()
    Wraps: retrieve → keyword guard → generate → judge
    With divergence retry (max 2 attempts).
    """
    query = task if task else "answer the user's question"
    max_retries = 2

    for attempt in range(1, max_retries + 1):
        print(f"  [research] attempt {attempt}, query='{query[:60]}...'")

        # Retrieve
        query_vec = _embed(query)
        chunks = _vector_search(doc_id or "", query_vec, top_k)

        if not chunks:
            if attempt < max_retries:
                query = query + " explained with details and examples"
                continue
            return SubAgentResult(
                agent="research", summary=FALLBACK,
                citations=[], confidence=0.0, success=False,
            )

        best_score = chunks[0]["score"]

        # Keyword guard
        top_text = chunks[0]["text"].lower()
        terms = _significant_terms(query)
        matched = {t for t in terms if t in top_text} if terms else terms

        if terms and not matched and best_score < 0.70:
            print(f"  [research] keyword guard FAILED — terms={sorted(terms)}, matched=none")
            if attempt < max_retries:
                query = query + " explained with details and examples"
                continue
            return SubAgentResult(
                agent="research", summary=FALLBACK,
                citations=[], confidence=best_score, success=False,
            )

        # Generate
        usable = [c for c in chunks if c["score"] >= COSINE_LOW][:3]
        context = "\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in usable)

        prompt = f"""You are a careful technical assistant. Answer using ONLY the provided context.
Cite chunks by their bracketed id inline like [chunk_id].
If the answer is not supported, reply exactly: {FALLBACK}

Context:
{context}

Question: {query}"""

        answer = _chat(prompt)
        cited = re.findall(r"\[([a-zA-Z0-9_:.\-]+)\]", answer)

        # Judge
        judge_result = _judge(query, answer, usable)

        if judge_result.get("grounded") and judge_result.get("score", 0) >= 0.60:
            print(f"  [research] SUCCESS — score={best_score:.4f}, judge={judge_result.get('score', 0):.2f}")
            return SubAgentResult(
                agent="research", summary=answer,
                citations=cited, confidence=best_score, success=True,
            )

        # Diverged — retry
        print(f"  [research] diverged — judge.grounded={judge_result.get('grounded')}, "
              f"judge.score={judge_result.get('score', 0):.2f}")
        if attempt < max_retries:
            query = query + " explained with details and examples"

    # Max retries — return best available
    return SubAgentResult(
        agent="research", summary=answer if answer else FALLBACK,
        citations=cited if cited else [], confidence=best_score if best_score else 0.0,
        success=True,  # return what we have, even if imperfect
    )


def _judge(question: str, answer: str, chunks: list[dict]) -> dict:
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = f"""You are evaluating whether a generated answer is grounded in the provided context.

Context:
{context}

Question: {question}

Answer to evaluate:
{answer}

Reply with ONLY a JSON object: {{"grounded": true|false, "correct": true|false, "complete": true|false, "score": 0.0-1.0, "reason": "..."}}"""

    raw = _chat(prompt)
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return {"grounded": False, "score": 0.0, "reason": "judge parse failed"}


# ═══════════════════════════════════════════════════════════════════════
# VEHICLE SUB-AGENT
# ═══════════════════════════════════════════════════════════════════════

def vehicle_agent(task: str, args: dict) -> SubAgentResult:
    """
    Java equivalent: VehicleSubAgent.execute()
    Internal routing: fetch+summarize or compare (based on args).
    """
    vehicle_id = args.get("vehicleId", "")
    vehicle_ids_raw = args.get("vehicleIds", "")

    # Route: comparison if vehicleIds present, else fetch+summarize
    if vehicle_ids_raw or "compare" in (task or "").lower():
        return _vehicle_compare(task, vehicle_ids_raw)

    if not vehicle_id:
        return SubAgentResult(
            agent="vehicle", summary="vehicleId is required",
            citations=[], confidence=0.0, success=False,
        )

    return _vehicle_fetch_and_summarize(task, vehicle_id, args)


def _vehicle_fetch_and_summarize(task: str, vehicle_id: str, args: dict) -> SubAgentResult:
    question = args.get("question", task or "vehicle specifications")
    top_k = int(args.get("topK", 3))

    print(f"  [vehicle] fetch+summarize — vehicleId='{vehicle_id}'")

    query_vec = _embed(question)
    # Search scoped to this vehicle's doc_id
    vec_literal = "[" + ",".join(str(x) for x in query_vec) + "]"
    sql = """
        SELECT node_id AS chunk_id, text,
               1 - (embedding <=> %s::vector) AS score
        FROM data_llamaindex_chunks
        WHERE metadata_ ->> 'doc_id' = %s
           OR node_id LIKE %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    # Try scoped search first, fall back to global
    chunks = _vector_search(vehicle_id, query_vec, top_k)

    if not chunks:
        return SubAgentResult(
            agent="vehicle",
            summary=f"No spec data found for vehicle: {vehicle_id}",
            citations=[], confidence=0.0, success=False,
        )

    # Generate summary
    context = "\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in chunks)
    prompt = f"""You are an automotive content writer.
Write a concise, consumer-friendly summary (2–4 sentences) for this vehicle.
Use ONLY the provided specification data. Do not invent facts.
Focus on: powertrain, key performance figures, standout features, and price.
Do NOT include citations or chunk IDs in the output.

Vehicle ID: {vehicle_id}

Spec data:
{context}"""

    summary = _chat(prompt)
    print(f"  [vehicle] summary generated — {len(summary)} chars")

    return SubAgentResult(
        agent="vehicle", summary=summary,
        citations=[f"{vehicle_id}:1"], confidence=1.0, success=True,
    )


def _vehicle_compare(task: str, vehicle_ids_raw: str) -> SubAgentResult:
    if isinstance(vehicle_ids_raw, list):
        vehicle_ids = vehicle_ids_raw
    else:
        vehicle_ids = [v.strip() for v in str(vehicle_ids_raw).split(",") if v.strip()]

    if len(vehicle_ids) < 2:
        return SubAgentResult(
            agent="vehicle", summary="Need at least 2 vehicleIds to compare",
            citations=[], confidence=0.0, success=False,
        )

    print(f"  [vehicle] comparing {vehicle_ids}")

    # Fetch top chunk per vehicle
    all_chunks = []
    for vid in vehicle_ids:
        query_vec = _embed(f"{vid} {task or 'performance specs'}")
        chunks = _vector_search(vid, query_vec, 1)
        if chunks:
            all_chunks.append(f"[{chunks[0]['chunk_id']}]\n{chunks[0]['text']}")
        else:
            all_chunks.append(f"[{vid}] No data available.")

    context = "\n\n".join(all_chunks)
    prompt = f"""You are an automotive analyst.
Compare the following vehicles: {', '.join(vehicle_ids)}
Focus on: {task or 'overall comparison'}

Rules:
- Use ONLY the provided spec data — do not invent figures.
- Be concise: 3–5 sentences total.
- State clear winners/trade-offs where the data supports it.
- Do NOT include chunk IDs or citation markers in the output.

Spec data:
{context}"""

    comparison = _chat(prompt)
    print(f"  [vehicle] comparison generated — {len(comparison)} chars")

    return SubAgentResult(
        agent="vehicle", summary=comparison,
        citations=[f"{vid}:1" for vid in vehicle_ids],
        confidence=1.0, success=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# COMMUNICATION SUB-AGENT
# ═══════════════════════════════════════════════════════════════════════

def communication_agent(task: str, content: str, args: dict) -> SubAgentResult:
    """
    Java equivalent: CommunicationSubAgent.execute()
    Formats content as email or SMS. No external send in this experiment —
    just produces the formatted message.
    """
    comm_type = args.get("type", "email")
    if "sms" in (task or "").lower() or "text" in (task or "").lower():
        comm_type = "sms"

    if comm_type == "sms":
        return _compose_sms(content, args)

    return _draft_email(content, args)


def _draft_email(content: str, args: dict) -> SubAgentResult:
    recipient = args.get("recipient", "hr@company.com")
    subject = args.get("subject", "Requested Summary")

    body = f"""Hello,

Please find the requested summary below:

{content or 'No content available.'}

Regards"""

    print(f"  [communication] email drafted to={recipient}")

    return SubAgentResult(
        agent="communication",
        summary=f"Email drafted to {recipient}",
        citations=[], confidence=1.0, success=True,
    )


def _compose_sms(content: str, args: dict) -> SubAgentResult:
    phone = args.get("phoneNumber", "+10000000000")
    message = content[:160] + "..." if content and len(content) > 160 else (content or "")

    print(f"  [communication] SMS composed to={phone}, {len(message)} chars")

    return SubAgentResult(
        agent="communication",
        summary=f"SMS composed to {phone}",
        citations=[], confidence=1.0, success=True,
    )
