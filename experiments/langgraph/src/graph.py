"""
The graph definition. THIS is where we can see what LangGraph specifically
contributes versus what we write our self.

Compare this file to a Java method that would do the same thing:

    public AskResponse handle(String question, int topK) {
        var chunks = retrieve(question, topK);
        var bestScore = chunks.isEmpty() ? null : chunks.get(0).score();

        if (!passesMinimumThreshold(chunks, bestScore)) {
            return refuse("score gate");
        }
        if (!passesKeywordGuard(question, bestScore, chunks.get(0).text())) {
            return refuse("keyword guard");
        }

        var usable = selectUsableHits(chunks);
        var answer = generate(question, usable);
        var judgeResult = judge(question, answer, usable);

        if (!judgeResult.grounded()) {
            return refuse("judge");
        }
        return new AskResponse(answer, ...);
    }

The Java version is procedural — control flow lives in `if` statements and
`return`s. The LangGraph version below is declarative — control flow is
described as a graph structure, separate from the work each node does.

Whether that's better is a judgment call. The Java version is more readable
in isolation. The LangGraph version is easier to *modify* — adding a new
node only requires adding one node and one edge, not threading new variables
through a long method. It also gives you `graph.get_graph().draw_mermaid()`
for free, which becomes valuable as the workflow grows.
"""

from langgraph.graph import END, START, StateGraph

from .nodes import (
    filter_chunks,
    finalize,
    generate,
    judge_answer,
    keyword_guard,
    refuse,
    retrieve,
    score_gate,
)
from .state import GraphState


def _route_after_score_gate(state: GraphState) -> str:
    """
    Conditional edge function. LangGraph calls this after `score_gate` runs
    and uses the returned string to pick the next node.

    This is the LangGraph equivalent of an `if` statement, but separated
    from the node logic. The node decides "did the gate pass or fail";
    the edge decides "where do we go next based on that."
    """
    if state.get("refusal_reason"):
        return "refuse"
    return "keyword_guard"


def _route_after_keyword_guard(state: GraphState) -> str:
    if state.get("refusal_reason"):
        return "refuse"
    return "filter_chunks"


def _route_after_judge(state: GraphState) -> str:
    judge_result = state.get("judge")
    if not judge_result or not judge_result.get("grounded"):
        return "refuse"
    return "finalize"


def build_graph():
    """
    Constructs and compiles the workflow graph.

    Three things to notice that LangGraph specifically contributes:

    1. `StateGraph(GraphState)` — the framework knows the shape of the
       state at build time (from the TypedDict) and can validate that
       every node returns only declared keys.

    2. `add_node` / `add_edge` / `add_conditional_edges` — declarative
       wiring. The graph structure is data, not control flow embedded in
       a procedural method. You can serialize it, visualize it, or modify
       it programmatically.

    3. `compile()` — produces an executable runtime that handles state
       merging, error propagation, and (if a checkpointer is wired in)
       persistence between nodes. None of that infrastructure code lives
       in this file — it's all in LangGraph itself.
    """
    workflow = StateGraph(GraphState)

    # --- nodes — each is a function that takes state, returns partial update ---
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("score_gate", score_gate)
    workflow.add_node("keyword_guard", keyword_guard)
    workflow.add_node("filter_chunks", filter_chunks)
    workflow.add_node("generate", generate)
    workflow.add_node("judge_answer", judge_answer)
    workflow.add_node("refuse", refuse)
    workflow.add_node("finalize", finalize)

    # --- entry point ---
    workflow.add_edge(START, "retrieve")

    # --- linear segment: retrieve → score_gate ---
    workflow.add_edge("retrieve", "score_gate")

    # --- conditional: score_gate → (refuse | keyword_guard) ---
    workflow.add_conditional_edges(
        "score_gate",
        _route_after_score_gate,
        {
            "refuse": "refuse",
            "keyword_guard": "keyword_guard",
        },
    )

    # --- conditional: keyword_guard → (refuse | filter_chunks) ---
    workflow.add_conditional_edges(
        "keyword_guard",
        _route_after_keyword_guard,
        {
            "refuse": "refuse",
            "filter_chunks": "filter_chunks",
        },
    )

    # --- linear segment: filter_chunks → generate → judge ---
    workflow.add_edge("filter_chunks", "generate")
    workflow.add_edge("generate", "judge_answer")

    # --- conditional: judge → (refuse | finalize) ---
    workflow.add_conditional_edges(
        "judge_answer",
        _route_after_judge,
        {
            "refuse": "refuse",
            "finalize": "finalize",
        },
    )

    # --- both terminal nodes go to END ---
    workflow.add_edge("refuse", END)
    workflow.add_edge("finalize", END)

    # TODO: add a checkpointer for resumable workflows:
    #   from langgraph.checkpoint.sqlite import SqliteSaver
    #   checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
    #   return workflow.compile(checkpointer=checkpointer)
    # With a checkpointer, you can pause the graph (e.g. before `judge`)
    # and resume hours or days later with the state intact. That's the
    # killer feature for human-in-the-loop or long-running workflows.

    return workflow.compile()
