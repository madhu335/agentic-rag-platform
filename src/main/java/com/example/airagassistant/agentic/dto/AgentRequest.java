package com.example.airagassistant.agentic.dto;

public record AgentRequest(
        String docId,
        String prompt,
        String recipientEmail,
        Integer topK
) {}