package com.example.airagassistant.eval;

public record EvalSummary(
        int total,
        double avgPrecisionAtK,
        double avgRecallAtK,
        double hitRateAtK,
        double answerPassRate
) {}