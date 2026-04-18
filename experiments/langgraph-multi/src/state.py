"""
Multi-agent supervisor state.

Compared to the single-agent GraphState (which mirrors RagAnswerService),
this state mirrors SupervisorAgent — it tracks delegations and sub-agent
results rather than individual retrieval/gating steps.

Two key differences from the single-agent state:

1. `delegations` — a list of agent assignments produced by the supervisor
   planner. The single-agent state doesn't have this because there's no
   routing decision; every question goes through the same retrieve→gate→
   generate→judge pipeline.

2. `agent_results` — a dict of sub-agent outputs keyed by agent name.
   The supervisor reads this to pass content between agents (research
   result → communication content) and to synthesize the final response.
"""

from typing import TypedDict, Optional


class Delegation(TypedDict):
    """One delegation from the supervisor to a sub-agent."""
    agent: str               # "research", "vehicle", "communication"
    task: str                # natural-language task
    args: dict               # domain-specific args


class SubAgentResult(TypedDict):
    """Uniform result from any sub-agent."""
    agent: str
    summary: str
    citations: list[str]
    confidence: float
    success: bool


class MultiAgentState(TypedDict, total=False):
    """
    The state that flows through the supervisor graph.

    Mapping to Java SupervisorAgent:
        question          -> prompt (from AgentRequest)
        doc_id            -> docId
        top_k             -> topK
        delegations       -> List<Delegation> from SupervisorPlanner.plan()
        current_index     -> loop counter (which delegation are we on?)
        agent_results     -> accumulated sub-agent outputs
        last_content      -> most recent content-producing result (for passing to communication)
        final_answer      -> synthesized response
    """

    # --- inputs ---
    question: str
    doc_id: Optional[str]
    top_k: int

    # --- supervisor planner output ---
    delegations: list[Delegation]

    # --- execution loop ---
    current_index: int
    agent_results: dict[str, SubAgentResult]
    last_content: Optional[str]       # content from research/vehicle for communication to use

    # --- terminal ---
    final_answer: str
    final_citations: list[str]
    final_confidence: float
