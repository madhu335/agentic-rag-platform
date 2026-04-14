package com.example.airagassistant.agentic.dto;

import java.time.Instant;
import java.util.List;

public record AgentSessionSummaryDto(
        String sessionId,
        String flowType,
        String status,
        String originalUserRequest,
        String currentUserRequest,
        String docId,
        int researchAttempts,
        Double confidenceScore,
        Double judgeScore,
        Boolean judgeGrounded,
        String emailStatus,
        String smsStatus,
        int historyCount,
        List<String> stepNames,
        Instant createdAt,
        Instant updatedAt
) {}