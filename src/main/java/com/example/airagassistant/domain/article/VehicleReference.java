package com.example.airagassistant.domain.article;

// ── Vehicle reference ─────────────────────────────────────────────────────────
// One article can reference multiple vehicles (comparison test, buyers guide)
public record VehicleReference(
        String vehicleId,        // links to vehicle domain: "bmw-m3-2025-competition"
        int    year,
        String make,
        String model,
        String trim,
        String role             // "primary", "competitor", "mentioned"
) {}
