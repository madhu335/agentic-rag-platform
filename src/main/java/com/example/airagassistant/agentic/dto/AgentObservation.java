package com.example.airagassistant.agentic.dto;

import com.example.airagassistant.judge.JudgeResult;

public record AgentObservation(
        String lastStep,
        String summary,
        Double retrievalScore,
        JudgeResult judge
) {}