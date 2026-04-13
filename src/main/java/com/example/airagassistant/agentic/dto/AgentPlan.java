package com.example.airagassistant.agentic.dto;

import java.util.List;

public record AgentPlan(
        List<PlanStep> steps
) {}