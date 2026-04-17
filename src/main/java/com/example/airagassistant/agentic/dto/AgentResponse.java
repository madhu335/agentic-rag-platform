package com.example.airagassistant.agentic.dto;

import com.example.airagassistant.judge.JudgeResult;

import java.util.List;

public record AgentResponse(
        String sessionId,
        String answer,
        List<String> citations,
        Double retrievalScore,
        JudgeResult judge,
        String emailStatus
) {}