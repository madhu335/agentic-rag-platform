package com.example.airagassistant.agentic.dto;

public record ContinueRequest(
        String sessionId,
        String instruction
) {}