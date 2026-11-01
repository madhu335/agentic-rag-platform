package com.example.airagassistant.policy;

import com.example.airagassistant.agentic.dto.ResearchResult;
import com.example.airagassistant.judge.JudgeResult;
import com.example.airagassistant.router.OrchestratorResult;
import org.springframework.stereotype.Service;

/**
 * Accept/reject decisions for orchestrator attempts and research results.
 *
 * Judge decisions are driven by {@link JudgeResult#verdict()} (2026-11-01), not by the grounded/correct
 * booleans, which could contradict each other and the score:
 *   PASS         accept (unless score is below MIN_JUDGE_SCORE)
 *   INCOMPLETE   accept if score >= MIN_INCOMPLETE_JUDGE_SCORE, else reject + retry
 *   UNSUPPORTED  reject + retry
 *   INCORRECT    reject + retry
 *   UNAVAILABLE  fall back to the retrieval score (route-specific threshold)
 *
 * Fix: UNAVAILABLE used to arrive as grounded=false and was rejected as "judge_not_grounded" before the
 * judge_unavailable branch could run, so a judge outage rejected every attempt and the orchestrator fell
 * through to AGENT_FALLBACK. It is now checked first.
 */
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

        Double bestScore = result.retrievalScore();
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

        if (judge.isUnavailable()) {
            return bestScore >= fallbackJudgeUnavailableThreshold(routeUsed)
                    ? EvaluationDecision.accepted(true, "judge_unavailable_score_passed")
                    : EvaluationDecision.rejected(true, true, "judge_unavailable_score_too_low");
        }

        return decideOnVerdict(judge);
    }

    @Override
    public EvaluationDecision evaluateResearchResult(ResearchResult result) {
        if (result == null) {
            return EvaluationDecision.rejected(false, true, "research_result_null");
        }

        Double retrievalScore = result.retrievalScore();
        JudgeResult judge = result.judge();

        // In the actual runtime, retrievalScore comes from orchestrator bestScore,
        // and practical runtime is centered on HYBRID_RERANK-style scoring.
        if (retrievalScore == null) {
            return EvaluationDecision.rejected(false, true, "confidence_null");
        }

        if (retrievalScore < MIN_HYBRID_RERANK_SCORE) {
            return EvaluationDecision.rejected(false, true, "confidence_below_hybrid_rerank_threshold");
        }

        if (judge == null) {
            return EvaluationDecision.rejected(true, true, "judge_null");
        }

        if (judge.isUnavailable()) {
            return EvaluationDecision.rejected(true, true, "judge_unavailable");
        }

        return decideOnVerdict(judge);
    }

    private EvaluationDecision decideOnVerdict(JudgeResult judge) {
        return switch (judge.verdict()) {
            case PASS -> judge.score() >= MIN_JUDGE_SCORE
                    ? EvaluationDecision.accepted(true, "accepted")
                    : EvaluationDecision.rejected(true, true, "judge_pass_but_score_too_low");
            case INCOMPLETE -> judge.score() >= MIN_INCOMPLETE_JUDGE_SCORE
                    ? EvaluationDecision.accepted(true, "accepted_incomplete")
                    : EvaluationDecision.rejected(true, true, "judge_incomplete_and_low_score");
            case UNSUPPORTED -> EvaluationDecision.rejected(true, true, "judge_unsupported");
            case INCORRECT -> EvaluationDecision.rejected(true, true, "judge_incorrect");
            case UNAVAILABLE -> EvaluationDecision.rejected(true, true, "judge_unavailable");
        };
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

    public double getMinJudgeScore() {
        return MIN_JUDGE_SCORE;
    }
}
