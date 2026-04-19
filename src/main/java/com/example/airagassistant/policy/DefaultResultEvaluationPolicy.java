package com.example.airagassistant.policy;

import com.example.airagassistant.agentic.dto.ResearchResult;
import com.example.airagassistant.judge.JudgeResult;
import com.example.airagassistant.router.OrchestratorResult;
import org.springframework.stereotype.Service;

@Service
public class DefaultResultEvaluationPolicy implements ResultEvaluationPolicy {

    // Mode-aware thresholds for orchestrator attempts
    private static final double MIN_VECTOR_SCORE = 0.55;
    private static final double MIN_BM25_SCORE = 0.10;
    private static final double MIN_HYBRID_SCORE = 0.08;
    private static final double MIN_HYBRID_RERANK_SCORE = 0.20;

    // Judge thresholds
    private static final double MIN_JUDGE_SCORE = 0.70;
    private static final double MIN_INCOMPLETE_JUDGE_SCORE = 0.75;

    // Only judge non-agent attempts if score is not obviously too weak
    private static final double MIN_SCORE_TO_JUDGE = 0.15;

    @Override
    public EvaluationDecision evaluateOrchestratorResult(OrchestratorResult result) {
        if (result == null) {
            return EvaluationDecision.rejected(false, true, "result_null");
        }

        Double bestScore = result.bestScore();
        String routeUsed = result.routeUsed();

        if (bestScore == null) {
            return EvaluationDecision.rejected(false, true, "best_score_null");
        }

        double minScore = minScoreForRoute(routeUsed);
        boolean shouldJudge = shouldJudgeForRoute(routeUsed, bestScore);

        if (bestScore < minScore) {
            return EvaluationDecision.rejected(
                    shouldJudge,
                    true,
                    "below_route_threshold:" + routeUsed
            );
        }

        JudgeResult judge = result.judge();

        if (judge == null) {
            return EvaluationDecision.accepted(shouldJudge, "judge_missing_but_score_passed");
        }

        if (!judge.grounded()) {
            return EvaluationDecision.rejected(true, true, "judge_not_grounded");
        }

        if (!judge.correct()) {
            return EvaluationDecision.rejected(true, true, "judge_not_correct");
        }

        if ("judge_unavailable".equalsIgnoreCase(judge.reason())) {
            return bestScore >= fallbackJudgeUnavailableThreshold(routeUsed)
                    ? EvaluationDecision.accepted(true, "judge_unavailable_score_passed")
                    : EvaluationDecision.rejected(true, true, "judge_unavailable_score_too_low");
        }

        if (judge.score() < MIN_JUDGE_SCORE) {
            return EvaluationDecision.rejected(true, true, "judge_score_too_low");
        }

        if (!judge.complete() && judge.score() < MIN_INCOMPLETE_JUDGE_SCORE) {
            return EvaluationDecision.rejected(true, true, "judge_incomplete_and_low_score");
        }

        return EvaluationDecision.accepted(true, "accepted");
    }

    @Override
    public EvaluationDecision evaluateResearchResult(ResearchResult result) {
        if (result == null) {
            return EvaluationDecision.rejected(false, true, "research_result_null");
        }

        Double retrievalScore  = result.retrievalScore();
        JudgeResult judge = result.judge();

        // In your actual runtime, retrievalScore  comes from orchestrator bestScore,
        // and practical runtime is centered on HYBRID_RERANK-style scoring.
        if (retrievalScore  == null) {
            return EvaluationDecision.rejected(false, true, "confidence_null");
        }

        if (retrievalScore  < MIN_HYBRID_RERANK_SCORE) {
            return EvaluationDecision.rejected(false, true, "confidence_below_hybrid_rerank_threshold");
        }

        if (judge == null) {
            return EvaluationDecision.rejected(true, true, "judge_null");
        }

        if (!judge.grounded()) {
            return EvaluationDecision.rejected(true, true, "judge_not_grounded");
        }

        if (!judge.correct()) {
            return EvaluationDecision.rejected(true, true, "judge_not_correct");
        }

        if (judge.score() < MIN_JUDGE_SCORE) {
            return EvaluationDecision.rejected(true, true, "judge_score_too_low");
        }

        if (!judge.complete() && judge.score() < MIN_INCOMPLETE_JUDGE_SCORE) {
            return EvaluationDecision.rejected(true, true, "judge_incomplete_and_low_score");
        }

        return EvaluationDecision.accepted(true, "accepted");
    }

    private boolean shouldJudgeForRoute(String routeUsed, double bestScore) {
        if ("AGENT".equalsIgnoreCase(routeUsed) || "AGENT_FALLBACK".equalsIgnoreCase(routeUsed)) {
            return true;
        }
        return bestScore >= MIN_SCORE_TO_JUDGE;
    }

    private double minScoreForRoute(String routeUsed) {
        if (routeUsed == null) {
            return MIN_HYBRID_RERANK_SCORE;
        }

        return switch (routeUsed.toUpperCase()) {
            case "VECTOR" -> MIN_VECTOR_SCORE;
            case "BM25" -> MIN_BM25_SCORE;
            case "HYBRID" -> MIN_HYBRID_SCORE;
            case "HYBRID_RERANK" -> MIN_HYBRID_RERANK_SCORE;
            case "AGENT", "AGENT_FALLBACK" -> 0.0;
            default -> MIN_HYBRID_RERANK_SCORE;
        };
    }

    private double fallbackJudgeUnavailableThreshold(String routeUsed) {
        if (routeUsed == null) {
            return MIN_HYBRID_RERANK_SCORE;
        }

        return switch (routeUsed.toUpperCase()) {
            case "VECTOR" -> 0.75;
            case "BM25" -> 0.20;
            case "HYBRID" -> 0.15;
            case "HYBRID_RERANK" -> 0.25;
            case "AGENT", "AGENT_FALLBACK" -> 0.0;
            default -> 0.25;
        };
    }
    // In DefaultResultEvaluationPolicy.java
    public double getMinJudgeScore() {
        return MIN_JUDGE_SCORE;
    }
}