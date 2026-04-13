package com.example.airagassistant.domain.vehicle;

// ── Rankings ─────────────────────────────────────────────────────────────────
public record RankingRecord(
        String category,     // "sports sedans", "luxury cars", "family SUVs"
        int    rank,
        int    total,        // ranked Nth of total
        int    score,        // out of 100
        String source,       // "US News", "Consumer Reports", "JD Power"
        int    year,
        String notes
) {}
