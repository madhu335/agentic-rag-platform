package com.example.airagassistant.domain.vehicle;

// ── Maintenance schedule ──────────────────────────────────────────────────────
public record MaintenanceInterval(
        int    mileage,
        String service,          // human-readable: "Oil change + filter + tire rotation"
        int    partsCost,        // USD
        double laborHours,
        int    dealerAvg,        // USD total at dealership
        int    diyAvg,           // USD doing it yourself
        String notes
) {}
