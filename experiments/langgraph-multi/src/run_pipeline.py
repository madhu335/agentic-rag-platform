"""
CLI entry point for the multi-agent LangGraph experiment.

Usage:
    python -m src.run_pipeline                                    # default questions
    python -m src.run_pipeline --question "..." --doc-id "..."    # custom
    python -m src.run_pipeline --print-graph                      # Mermaid diagram
"""

import argparse
import time

from dotenv import load_dotenv
load_dotenv()

from .graph import build_graph  # noqa: E402


DEFAULT_SCENARIOS = [
    {
        "question": "What is Spring Boot auto-configuration?",
        "doc_id": "spring-boot-qa",
        "label": "Research only (Q&A)",
    },
    {
        "question": "What is Kubernetes pod autoscaling?",
        "doc_id": "spring-boot-qa",
        "label": "Research refusal test (out-of-scope)",
    },
    {
        "question": "Tell me about the BMW M3 Competition specs and performance",
        "doc_id": "bmw-m3-2025-competition",
        "label": "Vehicle fetch + summarize",
    },
    {
        "question": "Research Spring Boot auto-configuration and email the answer to hr@company.com",
        "doc_id": "spring-boot-qa",
        "label": "Research + email (multi-agent delegation)",
    },
]


def run_one(question: str, doc_id: str, top_k: int, label: str = "") -> None:
    if label:
        print(f"\n{'=' * 72}")
        print(f"SCENARIO: {label}")
    print(f"{'=' * 72}")
    print(f"Q: {question}")
    print(f"doc_id: {doc_id}")
    print(f"-" * 72)

    graph = build_graph()
    initial_state = {
        "question": question,
        "doc_id": doc_id,
        "top_k": top_k,
    }

    t0 = time.perf_counter()
    final_state = graph.invoke(initial_state)
    elapsed = time.perf_counter() - t0

    print(f"-" * 72)
    print(f"A: {final_state.get('final_answer', '<no answer>')}")
    print()
    print(f"[timing]     total: {elapsed:.2f}s")
    print(f"[citations]  {final_state.get('final_citations', [])}")
    print(f"[confidence] {final_state.get('final_confidence', 0.0)}")

    delegations = final_state.get("delegations", [])
    print(f"[delegations] {len(delegations)}: {[d['agent'] for d in delegations]}")

    results = final_state.get("agent_results", {})
    for agent_name, result in results.items():
        status = "SUCCESS" if result.get("success") else "FAILED"
        summary_preview = (result.get("summary") or "")[:100]
        print(f"  [{agent_name}] {status} — {summary_preview}...")


def print_graph_diagram() -> None:
    """Print the graph as a Mermaid diagram."""
    graph = build_graph()
    mermaid = graph.get_graph().draw_mermaid()
    print(mermaid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent LangGraph experiment")
    parser.add_argument("--question", help="Custom question")
    parser.add_argument("--doc-id", default="spring-boot-qa")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--print-graph", action="store_true")
    args = parser.parse_args()

    if args.print_graph:
        print_graph_diagram()
        return

    if args.question:
        run_one(args.question, args.doc_id, args.top_k)
    else:
        for scenario in DEFAULT_SCENARIOS:
            run_one(
                scenario["question"],
                scenario.get("doc_id", args.doc_id),
                args.top_k,
                label=scenario.get("label", ""),
            )


if __name__ == "__main__":
    main()
