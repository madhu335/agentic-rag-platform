package com.example.airagassistant.domain.vehicle;

import java.util.List;

// ── Trim levels ───────────────────────────────────────────────────────────────
public record TrimLevel(
        String trim,
        String msrp,
        List<String> addedFeatures,
        Integer extraHorsepower     // delta vs base
) {}
