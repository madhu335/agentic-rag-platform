package com.example.airagassistant.domain.vehicle;

// ── Total ownership cost (5-year) ─────────────────────────────────────────────
public record OwnershipCost(
        int    fuelCostPerYear,
        int    insurancePerYear,
        int    maintenancePerYear,
        int    depreciationFiveYear,
        int    totalFiveYear,
        double resaleValuePct,    // % of MSRP retained after 3 years
        String resaleRating       // "excellent", "good", "average", "poor"
) {}
