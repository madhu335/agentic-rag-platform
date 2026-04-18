"""
The multi-agent supervisor graph.

THIS IS THE FILE TO COMPARE WITH SupervisorAgent.java.

In Java, the supervisor is a procedural method:
    for (delegation : delegations) {
        switch (delegation.agent()) {
            case "research" -> researchAgent.execute(...)
            case "vehicle"  -> vehicleAgent.execute(...)
            case "communication" -> communicationAgent.execute(...)
        }
    }

In LangGraph, the same logic is a declarative graph:

    START → plan → route_next
              ↓            ↓
         research    vehicle    communication
              ↓            ↓           ↓
         route_next ←──────←───────────←
              ↓
        (all done)
              ↓
          finalize → END

The graph handles:
    - Sequential delegation execution (via current_index counter)
    - Conditional routing to the right sub-agent
    - Content passing between agents (last_content → communication)
    - Loop termination when all delegations are done

What LangGraph gives you that Java doesn't:
    1. The loop is declarative — no manual `for` or `while`
    2. The routing is a conditional edge function, not a `switch` statement
    3. State merging is automatic — each node returns a partial dict
    4. Free Mermaid visualization via graph.get_graph().draw_mermaid()
    5. Adding a new agent = one node + one edge, not modifying a switch block

What you STILL write yourself (same as Java):
    - The supervisor planner prompt
    - The sub-agent logic (retrieval, LLM calls, keyword guard)
    - The content-passing logic (research result → communication input)
"""

import json
import os
import re

import httpx
from langgraph.graph import END, START, StateGraph

from .agents import research_agent, vehicle_agent, communication_agent
from .state import MultiAgentState, Delegation, SubAgentResult

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = "llama3.1"


def _chat(prompt: str) -> str:
    r = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": False},
        timeout=120.0,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


# ═══════════════════════════════════════════════════════════════════════
# NODES
# ═══════════════════════════════════════════════════════════════════════

def plan(state: MultiAgentState) -> dict:
    """
    Node: supervisor planner.
    Java equivalent: SupervisorPlanner.plan()

    Asks the LLM to decompose the user's question into agent delegations.
    The prompt is ~30 lines (vs ~80 lines in the single-agent PlannerService).
    """
    question = state["question"]
    doc_id = state.get("doc_id", "")

    prompt = f"""You are a supervisor that delegates tasks to specialized agents.

Available agents:
- research: retrieves information from ingested documents
- vehicle: handles vehicle specs, summaries, comparisons. MUST include "vehicleId" in args.
- communication: drafts emails or SMS messages

Rules:
- If the user asks about documents or general knowledge, delegate to research.
- If the user asks about a specific vehicle, delegate to vehicle.
- If the user asks to email or text something, delegate to communication AFTER research/vehicle.
- ONLY include communication if the user EXPLICITLY asks to email, send, or text.
- Words like "tell me", "show me", "what is" do NOT imply email.
- For comparisons, use ONE vehicle delegation with "vehicleIds" in args.

Return ONLY valid JSON:
{{"delegations": [{{"agent": "research", "task": "...", "args": {{}}}}]}}

User request: {question}"""

    raw = _chat(prompt)
    print(f"[plan] LLM response: {raw[:200]}...")

    delegations = _parse_delegations(raw, doc_id)
    print(f"[plan] {len(delegations)} delegations: {[d['agent'] for d in delegations]}")

    return {
        "delegations": delegations,
        "current_index": 0,
        "agent_results": {},
    }


def _parse_delegations(raw: str, doc_id: str) -> list[Delegation]:
    """Parse LLM output into validated delegations."""
    default = [Delegation(agent="research", task="answer the question", args={})]

    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first < 0 or last <= first:
        return default

    try:
        parsed = json.loads(cleaned[first:last + 1])
    except json.JSONDecodeError:
        return default

    raw_list = parsed.get("delegations", [])
    if not raw_list:
        return default

    valid_agents = {"research", "vehicle", "communication"}
    result = []

    for item in raw_list:
        agent = (item.get("agent") or "").strip().lower()
        if agent not in valid_agents:
            continue
        task = (item.get("task") or "").strip()
        args = dict(item.get("args") or {})

        # Override vehicleId with doc_id when it looks like a vehicleId
        if agent == "vehicle" and doc_id and re.search(r"\d{4}", doc_id):
            args["vehicleId"] = doc_id

        result.append(Delegation(agent=agent, task=task, args=args))

    return result if result else default


def dispatch(state: MultiAgentState) -> dict:
    """
    Node: dispatch current delegation to the appropriate sub-agent.

    This is the equivalent of the `switch` block inside SupervisorAgent.handle().
    In LangGraph, it's a node that reads the current delegation from state,
    calls the right sub-agent, and stores the result.
    """
    delegations = state.get("delegations", [])
    idx = state.get("current_index", 0)

    if idx >= len(delegations):
        # Should not happen — route_next should have sent us to finalize
        return {"current_index": idx}

    delegation = delegations[idx]
    agent_name = delegation["agent"]
    task = delegation.get("task", "")
    args = dict(delegation.get("args", {}))

    print(f"[dispatch] agent='{agent_name}' task='{task}'")

    result: SubAgentResult

    if agent_name == "research":
        result = research_agent(
            task,
            state.get("doc_id"),
            state.get("top_k", 5),
        )

    elif agent_name == "vehicle":
        result = vehicle_agent(task, args)

    elif agent_name == "communication":
        content = state.get("last_content", state.get("question", ""))
        result = communication_agent(task, content, args)

    else:
        result = SubAgentResult(
            agent=agent_name, summary=f"Unknown agent: {agent_name}",
            citations=[], confidence=0.0, success=False,
        )

    # Accumulate results
    results = dict(state.get("agent_results", {}))
    results[agent_name] = result

    # Track last content (for passing to communication)
    last_content = state.get("last_content")
    if result["success"] and agent_name != "communication":
        last_content = result["summary"]

    return {
        "agent_results": results,
        "last_content": last_content,
        "current_index": idx + 1,
    }


def finalize(state: MultiAgentState) -> dict:
    """
    Node: synthesize the final response from accumulated sub-agent results.
    Java equivalent: SupervisorAgent.buildResponse()
    """
    results = state.get("agent_results", {})

    # Prefer research or vehicle result over communication
    for agent_name in ["research", "vehicle", "communication"]:
        if agent_name in results and results[agent_name]["success"]:
            r = results[agent_name]
            if agent_name != "communication":
                return {
                    "final_answer": r["summary"],
                    "final_citations": r.get("citations", []),
                    "final_confidence": r.get("confidence", 0.0),
                }

    # Fall back to whatever we have
    for r in results.values():
        if r["success"]:
            return {
                "final_answer": r["summary"],
                "final_citations": r.get("citations", []),
                "final_confidence": r.get("confidence", 0.0),
            }

    return {
        "final_answer": "No agents produced a result.",
        "final_citations": [],
        "final_confidence": 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════
# ROUTING (conditional edges)
# ═══════════════════════════════════════════════════════════════════════

def route_after_dispatch(state: MultiAgentState) -> str:
    """
    Conditional edge: after dispatching one delegation, decide whether
    to dispatch the next one or finalize.

    This is the LangGraph equivalent of the `for` loop in SupervisorAgent.
    Instead of an explicit loop, the graph cycles: dispatch → route → dispatch
    until all delegations are done, then route → finalize.
    """
    delegations = state.get("delegations", [])
    idx = state.get("current_index", 0)

    if idx < len(delegations):
        return "dispatch"   # more delegations to process
    return "finalize"       # all done


# ═══════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Build the supervisor graph.

    Compare this to SupervisorAgent.handle():
        Java: 200 lines of procedural code (for loop + switch + try/catch)
        LangGraph: ~15 lines of graph wiring below

    The graph shape:

        START → plan → dispatch ←─┐
                          │       │
                    route_after_dispatch
                          │       │
                     (more?) ──yes─┘
                          │
                     (done?) ──→ finalize → END
    """
    workflow = StateGraph(MultiAgentState)

    # Nodes
    workflow.add_node("plan", plan)
    workflow.add_node("dispatch", dispatch)
    workflow.add_node("finalize", finalize)

    # Edges
    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "dispatch")

    # Dispatch loops until all delegations are processed
    workflow.add_conditional_edges(
        "dispatch",
        route_after_dispatch,
        {
            "dispatch": "dispatch",    # more delegations → loop back
            "finalize": "finalize",    # all done → finalize
        },
    )

    workflow.add_edge("finalize", END)

    return workflow.compile()
