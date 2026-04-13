package com.example.airagassistant.domain.vehicle;

// ── Review snippets ───────────────────────────────────────────────────────────
public record ReviewSnippet(
        String source,           // "Car and Driver", "Motor Trend"
        int    score,            // out of 10
        String summary,          // 1-2 sentence editorial summary
        String pros,
        String cons,
        int    year
) {}
