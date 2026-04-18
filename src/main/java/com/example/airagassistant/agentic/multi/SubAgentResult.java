package com.example.airagassistant.agentic.multi;

import com.example.airagassistant.judge.JudgeResult;

import java.util.List;
import java.util.Map;

/**
 * Uniform result type returned by every sub-agent.
 *
 * Design choice: every sub-agent returns the same shape regardless of domain.
 * The supervisor doesn't need to know whether the result came from research,
 * a vehicle comparison, or an email draft — it reads `summary` for the answer
 * and `metadata` for domain-specific details.
 *
 * This is the multi-agent equivalent of your existing `RagResult` and
 * `AgentResponse` — but designed for agent-to-agent communication, not
 * agent-to-user. The supervisor translates SubAgentResult into the final
 * user-facing response.
 */
public record SubAgentResult(
        String agentName,
        String summary,
        List<String> citations,
        Double confidence,
        JudgeResult judge,
        Map<String, Object> metadata,
        boolean success
) {

    /**
     * Quick factory for successful results.
     */
    public static SubAgentResult success(String agentName, String summary,
                                          List<String> citations, Double confidence,
                                          JudgeResult judge) {
        return new SubAgentResult(agentName, summary, citations, confidence, judge, Map.of(), true);
    }

    public static SubAgentResult success(String agentName, String summary,
                                          List<String> citations, Double confidence,
                                          JudgeResult judge, Map<String, Object> metadata) {
        return new SubAgentResult(agentName, summary, citations, confidence, judge, metadata, true);
    }

    /**
     * Quick factory for failures — summary carries the error reason.
     */
    public static SubAgentResult failure(String agentName, String reason) {
        return new SubAgentResult(agentName, reason, List.of(), 0.0, null, Map.of(), false);
    }
}
