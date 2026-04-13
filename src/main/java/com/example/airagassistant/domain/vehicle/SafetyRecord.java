package com.example.airagassistant.domain.vehicle;

// ── Safety ───────────────────────────────────────────────────────────────────
public record SafetyRecord(
        String nhtsaOverall,          // "5-star", "4-star"
        String nhtsaFrontal,
        String nhtsaSide,
        String nhtsaRollover,
        String iihsOverall,           // "Good", "Acceptable", "Marginal", "Poor"
        String iihsSmallOverlap,
        String iihsHeadlights,
        boolean hasAutomaticEmergencyBraking,
        boolean hasLaneDepartureWarning,
        boolean hasBlindSpotMonitoring
) {}
