package com.example.airagassistant.agentic.multi.agents;

import com.example.airagassistant.agentic.divergence.DivergenceCheckService;
import com.example.airagassistant.agentic.divergence.QueryRefinementService;
import com.example.airagassistant.agentic.dto.ResearchResult;
import com.example.airagassistant.agentic.multi.SubAgentResult;
import com.example.airagassistant.agentic.tools.ResearchTool;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Sub-agent: Research
 * <p>
 * Owns: document retrieval, quality checking, retry with refined queries.
 * <p>
 * This agent wraps the existing ResearchTool (which calls OrchestratorService,
 * which runs the BM25→HYBRID→RERANK→VECTOR→AGENT cascade). It adds
 * divergence detection and bounded retry on top — the same logic that
 * AgentExecutorService does for research steps, extracted into a
 * self-contained agent.
 * <p>
 * The key difference from the single-agent approach: this agent has NO
 * knowledge of email, SMS, or vehicles. It does one thing well — retrieve
 * and validate information.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class ResearchSubAgent {

    private static final int MAX_RETRIES = 2;
    private static final int DEFAULT_TOP_K = 5;

    private final ResearchTool researchTool;
    private final DivergenceCheckService divergenceCheckService;
    private final QueryRefinementService queryRefinementService;

    public SubAgentResult execute(String task, String docId, Integer topK) {
        int k = (topK != null && topK > 0) ? topK : DEFAULT_TOP_K;
        String query = (task != null && !task.isBlank()) ? task : "answer the user's question";

        log.info("ResearchSubAgent — query='{}' docId='{}' topK={}", query, docId, k);

        ResearchResult bestResult = null;
        int attempts = 0;

        while (attempts < MAX_RETRIES) {
            attempts++;
            ResearchResult result = null;
            try {
                result = researchTool.research(docId, query, k);
                bestResult = result;

                boolean diverged = divergenceCheckService.isDiverged(result);
                boolean lowQuality = shouldRetry(result);

                if (!diverged && !lowQuality) {
                    log.info("ResearchSubAgent — accepted on attempt {}: confidence={}",
                            attempts, result.retrievalScore());

                    return SubAgentResult.success(
                            "research",
                            result.answer(),
                            result.citations(),
                            result.retrievalScore(),
                            result.judge()
                    );
                }

                log.info("ResearchSubAgent — attempt {} diverged/low-quality, retrying. " +
                        "diverged={} lowQuality={}", attempts, diverged, lowQuality);

                // Refine query for next attempt
                query = refineQuery(query, result);

            } catch (Exception e) {
                log.warn("ResearchSubAgent — attempt {} failed: {}", attempts, e.getMessage());
                query = refineQuery(query, result);
            }
        }

        // Max retries reached — return best available
        if (bestResult != null) {
            log.info("ResearchSubAgent — max retries reached, returning best available.");
            return SubAgentResult.success(
                    "research",
                    bestResult.answer(),
                    bestResult.citations(),
                    bestResult.retrievalScore(),
                    bestResult.judge()
            );
        }

        return SubAgentResult.failure("research", "Research failed after " + MAX_RETRIES + " attempts.");
    }

    private boolean shouldRetry(ResearchResult result) {
        if (result == null) return true;

        String answer = result.answer() != null ? result.answer().toLowerCase() : "";
        boolean saysIDontKnow = answer.contains("i don't know based on the ingested documents");
        boolean hasChunks = result.chunks() != null && !result.chunks().isEmpty();

        double judgeScore = result.judge() != null ? result.judge().score() : 0.0;

        return (saysIDontKnow && hasChunks) || judgeScore < 0.60;
    }

    private String refineQuery(String query, ResearchResult result) {
        String refinedQuery = queryRefinementService.refineQuery(
                query,
                result != null ? result.answer() : null,
                result != null ? result.chunks() : null,
                result != null ? result.judge() : null
        );
        if (refinedQuery == null || refinedQuery.isBlank()) {
            refinedQuery = query + " detailed explanation with examples";
        }
        return refinedQuery;
    }
}
