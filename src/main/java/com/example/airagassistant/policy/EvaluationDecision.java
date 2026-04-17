package com.example.airagassistant.policy;

public record EvaluationDecision(
        boolean shouldJudge,
        boolean acceptable,
        boolean shouldRetry,
        String reason
) {
    public static EvaluationDecision accepted(boolean shouldJudge, String reason) {
        return new EvaluationDecision(shouldJudge, true, false, reason);
    }

    public static EvaluationDecision rejected(boolean shouldJudge, boolean shouldRetry, String reason) {
        return new EvaluationDecision(shouldJudge, false, shouldRetry, reason);
    }
}