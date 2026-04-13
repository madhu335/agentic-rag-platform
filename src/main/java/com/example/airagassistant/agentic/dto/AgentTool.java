package com.example.airagassistant.agentic.dto;

import com.example.airagassistant.agentic.state.AgentSessionState;

public interface AgentTool {
    String name();

    AgentSessionState execute(AgentSessionState state, PlanStep step);
}