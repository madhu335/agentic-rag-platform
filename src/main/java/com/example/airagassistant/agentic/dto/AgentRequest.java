package com.example.airagassistant.agentic.dto;

public record AgentRequest(
        String prompt,
        String docType,   // ✅ ADD THIS
        String docId,
        Integer topK
) {}