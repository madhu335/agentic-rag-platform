package com.example.airagassistant.judge;

public record JudgeResult(
        boolean grounded,
        boolean correct,
        boolean complete,
        double score,
        String reason
) {}