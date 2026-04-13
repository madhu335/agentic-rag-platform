package com.example.airagassistant.domain.vehicle;

// ── Recalls ──────────────────────────────────────────────────────────────────
public record RecallRecord(
        String recallId,
        String date,
        String component,        // "Fuel system", "Airbag", "Brakes"
        String description,
        String remedy,
        boolean remedyAvailable
) {}
