"""
CLI entry point for the LangGraph experiment.

Usage:
    python -m src.run_pipeline                                # runs default questions
    python -m src.run_pipeline --question "..."               # one custom question
    python -m src.run_pipeline --print-graph                  # print Mermaid diagram
"""

import argparse
import time

from dotenv import load_dotenv

# Must load .env before importing nodes/graph (they read env vars at import time).
load_dotenv()

from .graph import build_graph  # noqa: E402  (deliberate post-load import)

# The same two questions used in the LlamaIndex experiment, so any
# comparison stays apples-to-apples.
DEFAULT_QUESTIONS = [
    "How should I configure heap size for a Spring Boot app in production?",
    "What is Spring Boot auto-configuration?",
]


def run_one(question: str, top_k: int) -> None:
    print("\n" + "=" * 72)
    print(f"Q: {question}")
    print("-" * 72)

    graph = build_graph()
    initial_state = {"question": question, "top_k": top_k}

    t0 = time.perf_counter()
    final_state = graph.invoke(initial_state)
    elapsed = time.perf_counter() - t0

    print("-" * 72)
    print(f"A: {final_state.get('final_answer', '<no answer>')}")
    print()
    print(f"[timing] total: {elapsed:.2f}s")

    judge_result = final_state.get("judge")
    if judge_result:
        print(
            f"[judge] grounded={judge_result['grounded']} "
            f"correct={judge_result['correct']} "
            f"score={judge_result['score']:.2f}"
        )

    chunks = final_state.get("chunks", [])
    if chunks:
        print(f"[retrieval] {len(chunks)} retrieved, "
              f"{len(final_state.get('usable_chunks', []))} after filter:")
        for i, c in enumerate(chunks, 1):
            snippet = c["text"][:100].replace("\n", " ")
            print(f"  {i}. score={c['score']:.4f}  {snippet}...")


def print_graph_diagram() -> None:
    """
    Prints the graph as a Mermaid diagram. This is one of LangGraph's
    free perks — you didn't write any of this rendering, the framework
    knows the graph structure and can output it.

    Paste the result into https://mermaid.live to view it visually,
    or into any markdown file that supports Mermaid (GitHub does).
    """
    graph = build_graph()
    mermaid = graph.get_graph().draw_mermaid()
    print(mermaid)


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph RAG experiment")
    parser.add_argument(
        "--question",
        help="Single question to ask. Omit to run both default questions.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--print-graph",
        action="store_true",
        help="Print the graph as a Mermaid diagram and exit.",
    )
    args = parser.parse_args()

    if args.print_graph:
        print_graph_diagram()
        return

    questions = [args.question] if args.question else DEFAULT_QUESTIONS
    for q in questions:
        run_one(q, top_k=args.top_k)


if __name__ == "__main__":
    main()
