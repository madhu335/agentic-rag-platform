"""
CLI entry point. Run the LlamaIndex pipeline against sample-data/, backed by
the same Postgres instance as the Java app (different table).

Typical workflow:
    # One time (or whenever sample-data/ changes):
    python -m src.run_pipeline --ingest

    # Then, as many times as you want — goes straight to retrieval.
    python -m src.run_pipeline --mode vector
    python -m src.run_pipeline --mode bm25
    python -m src.run_pipeline --mode hybrid
    python -m src.run_pipeline --mode hybrid --question "What engine does the M3 use?"
"""

import argparse
import time
from pathlib import Path

from .answer import answer_with_vllm
from .ingest import build_index, load_index
from .retrieve import build_retriever

DEFAULT_QUESTIONS = [
    "What is this document about?",
    "Summarize the main points in three sentences.",
]


def do_ingest() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "sample-data"
    t0 = time.perf_counter()
    build_index(data_dir)
    elapsed = time.perf_counter() - t0
    print(f"\n[timing] ingestion: {elapsed:.2f}s")
    print("[ok] chunks persisted to Postgres. Run without --ingest to query.")


def do_query(mode: str, question: str | None, top_k: int) -> None:
    t0 = time.perf_counter()
    index = load_index()
    t_load = time.perf_counter() - t0
    print(f"[timing] index load: {t_load:.3f}s")

    retriever = build_retriever(index, mode=mode, top_k=top_k)  # type: ignore[arg-type]

    questions = [question] if question else DEFAULT_QUESTIONS

    for q in questions:
        print("\n" + "=" * 72)
        print(f"Q: {q}")
        print("-" * 72)

        t0 = time.perf_counter()
        source_nodes = retriever.retrieve(q)
        t_retrieve = time.perf_counter() - t0

        t1 = time.perf_counter()
        answer = answer_with_vllm(q, source_nodes)
        t_answer = time.perf_counter() - t1

        t_query = t_retrieve + t_answer

        print(f"A: {answer}")
        print(f"\n[timing] retrieve ({mode}): {t_retrieve:.2f}s")
        print(f"[timing] answer (vllm): {t_answer:.2f}s")
        print(f"[timing] query ({mode}): {t_query:.2f}s")
        print(f"[retrieval] {len(source_nodes)} source node(s):")
        for i, node in enumerate(source_nodes, 1):
            score = f"{node.score:.4f}" if node.score is not None else "n/a"
            snippet = node.node.get_content()[:120].replace("\n", " ")
            print(f"  {i}. score={score}  {snippet}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="LlamaIndex RAG experiment")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run ingestion pipeline (load, chunk, embed, write to pg). Do this once.",
    )
    parser.add_argument(
        "--mode",
        choices=["vector", "bm25", "hybrid"],
        default="hybrid",
        help="retrieval strategy (default: hybrid). Ignored when --ingest is set.",
    )
    parser.add_argument("--question", help="single question to ask; omit to use defaults")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.ingest:
        do_ingest()
    else:
        do_query(mode=args.mode, question=args.question, top_k=args.top_k)


if __name__ == "__main__":
    main()