package com.example.airagassistant.agentic.dto;

public record EmailToolRequest(
        String sessionId,
        String to,
        String subject
) {}