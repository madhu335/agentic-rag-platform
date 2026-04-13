package com.example.airagassistant.agentic.dto;

import java.util.Map;

public record PlanStep(
        String step,            // "research", "email"
        Map<String, Object> args
) {}