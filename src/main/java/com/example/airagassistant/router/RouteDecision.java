package com.example.airagassistant.router;

public record RouteDecision(
        QueryRouter.Route route,
        double confidence,
        String reason
) {}