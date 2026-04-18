package com.example.airagassistant.agentic.multi;

import java.util.Map;

/**
 * A single delegation from the supervisor to a sub-agent.
 *
 * The supervisor planner produces a list of these. Each one says
 * "call this agent with this task and these args."
 *
 * Example:
 *   Delegation("research", "Spring Boot auto-configuration", {})
 *   Delegation("vehicle", "compare performance", {vehicleIds: "bmw-m3-2025,porsche-911-2025"})
 *   Delegation("communication", "email the research result", {type: "email", recipient: "hr@company.com"})
 */
public record Delegation(
        String agent,           // "research", "vehicle", "communication"
        String task,            // natural-language task description
        Map<String, Object> args // domain-specific args
) {}
