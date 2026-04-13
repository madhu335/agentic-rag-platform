package com.example.airagassistant.agentic.state;

import java.time.Instant;
import java.util.Map;

public record StepHistoryEntry(
        String step,
        Map<String, Object> args,
        String status, // STARTED, SUCCESS, FAILED
        String resultSummary,
        String error,
        Instant timestamp
) {}