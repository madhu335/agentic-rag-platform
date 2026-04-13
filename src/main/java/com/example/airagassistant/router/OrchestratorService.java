package com.example.airagassistant.router;

import com.example.airagassistant.agent.AgentService;
import com.example.airagassistant.judge.JudgeResult;
import com.example.airagassistant.judge.JudgeService;
import com.example.airagassistant.rag.RagAnswerService;
import com.example.airagassistant.rag.RetrievalMode;
import com.example.airagassistant.trace.TraceHelper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.LinkedHashMap;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class OrchestratorService {

    private final QueryRouter queryRouter;
    private final RagAnswerService ragAnswerService;
    private final AgentService agentService;
    private final JudgeService judgeService;
    private final TraceHelper traceHelper;

    public OrchestratorResult handle(String question, String docId, int k) {
        RouteDecision decision = queryRouter.route(question);

        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.span.kind", "chain");
        attrs.put("langsmith.metadata.doc_id", docId);
        attrs.put("langsmith.metadata.route", decision.route().name());
        attrs.put("langsmith.metadata.route_reason", decision.reason());
        attrs.put("langsmith.metadata.route_confidence", decision.confidence());

        return traceHelper.run("orchestrator-flow", attrs, () -> {
            if (decision.route() == QueryRouter.Route.RAG) {

                var bm25 = runAttempt(
                        "BM25",
                        () -> ragAnswerService.answerWithMode(docId, question, k, RetrievalMode.BM25),
                        decision.reason(),
                        "bm25_success",
                        question
                );
                if (isAcceptable(bm25)) {
                    traceHelper.addAttributes(Map.of(
                            "langsmith.metadata.final_mode", "BM25",
                            "langsmith.metadata.final_outcome", bm25.outcome()
                    ));
                    return bm25;
                }

                var hybrid = runAttempt(
                        "HYBRID",
                        () -> ragAnswerService.answerWithMode(docId, question, k, RetrievalMode.HYBRID),
                        decision.reason(),
                        "hybrid_success",
                        question
                );
                if (isAcceptable(hybrid)) {
                    traceHelper.addAttributes(Map.of(
                            "langsmith.metadata.final_mode", "HYBRID",
                            "langsmith.metadata.final_outcome", hybrid.outcome()
                    ));
                    return hybrid;
                }

                var hybridRerank = runAttempt(
                        "HYBRID_RERANK",
                        () -> ragAnswerService.answerWithMode(docId, question, k, RetrievalMode.HYBRID_RERANK),
                        decision.reason(),
                        "hybrid_rerank_success",
                        question
                );
                if (isAcceptable(hybridRerank)) {
                    traceHelper.addAttributes(Map.of(
                            "langsmith.metadata.final_mode", "HYBRID_RERANK",
                            "langsmith.metadata.final_outcome", hybridRerank.outcome()
                    ));
                    return hybridRerank;
                }

                var vector = runAttempt(
                        "VECTOR",
                        () -> ragAnswerService.answerWithMode(docId, question, k, RetrievalMode.VECTOR),
                        decision.reason(),
                        "vector_success",
                        question
                );
                if (isAcceptable(vector)) {
                    traceHelper.addAttributes(Map.of(
                            "langsmith.metadata.final_mode", "VECTOR",
                            "langsmith.metadata.final_outcome", vector.outcome()
                    ));
                    return vector;
                }

                var agentFallback = runAttempt(
                        "AGENT_FALLBACK",
                        () -> agentService.answer(docId, question, k),
                        decision.reason(),
                        "all_retrievers_failed",
                        question
                );

                traceHelper.addAttributes(Map.of(
                        "langsmith.metadata.final_mode", "AGENT_FALLBACK",
                        "langsmith.metadata.final_outcome", agentFallback.outcome()
                ));
                return agentFallback;
            }

            var agent = runAttempt(
                    "AGENT",
                    () -> agentService.answer(docId, question, k),
                    decision.reason(),
                    "agent_direct",
                    question
            );

            traceHelper.addAttributes(Map.of(
                    "langsmith.metadata.final_mode", "AGENT",
                    "langsmith.metadata.final_outcome", agent.outcome()
            ));
            return agent;
        });
    }

    private OrchestratorResult runAttempt(
            String routeUsed,
            java.util.function.Supplier<RagAnswerService.RagResult> supplier,
            String reason,
            String outcome,
            String question
    ) {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.span.kind", "chain");
        attrs.put("langsmith.metadata.attempt_mode", routeUsed);
        attrs.put("langsmith.metadata.attempt_reason", reason);

        return traceHelper.run("retrieval-attempt-" + routeUsed.toLowerCase(), attrs, () -> {
            OrchestratorResult result = buildRagResult(routeUsed, supplier.get(), reason, outcome, question);

            Map<String, Object> resultAttrs = new LinkedHashMap<>();
            resultAttrs.put("langsmith.metadata.attempt_mode", routeUsed);
            resultAttrs.put("langsmith.metadata.accepted", isAcceptable(result));
            resultAttrs.put("langsmith.metadata.outcome", result.outcome());

            if (result.bestScore() != null) {
                resultAttrs.put("langsmith.metadata.best_score", result.bestScore());
            }

            resultAttrs.put("langsmith.metadata.used_chunks", result.usedChunks());
            resultAttrs.put("langsmith.metadata.retrieved_chunk_count", result.retrievedChunkIds() != null ? result.retrievedChunkIds().size() : 0);

            if (result.judge() != null) {
                resultAttrs.put("langsmith.metadata.judge.score", result.judge().score());
                resultAttrs.put("langsmith.metadata.judge.grounded", result.judge().grounded());
                resultAttrs.put("langsmith.metadata.judge.correct", result.judge().correct());
                resultAttrs.put("langsmith.metadata.judge.complete", result.judge().complete());
                resultAttrs.put("langsmith.metadata.judge.reason", result.judge().reason());
            } else {
                resultAttrs.put("langsmith.metadata.judge.skipped", true);
            }

            traceHelper.addAttributes(resultAttrs);
            return result;
        });
    }

    private boolean isAcceptable(OrchestratorResult result) {
        if (result == null || result.bestScore() == null) {
            return false;
        }

        if (result.bestScore() < 0.55) {
            return false;
        }

        if (result.judge() == null) {
            return true;
        }

        if (!result.judge().grounded()) {
            return false;
        }

        if ("judge_unavailable".equals(result.judge().reason())) {
            return result.bestScore() >= 0.75;
        }

        return result.judge().score() >= 0.70;
    }

    private OrchestratorResult buildRagResult(
            String routeUsed,
            RagAnswerService.RagResult result,
            String reason,
            String outcome,
            String question
    ) {
        var chunks = result.retrievedChunks().stream()
                .map(c -> new OrchestratorResult.Chunk(c.id(), c.text()))
                .toList();

        JudgeResult judge = null;
        boolean shouldJudge =
                "AGENT".equals(routeUsed) ||
                        "AGENT_FALLBACK".equals(routeUsed) ||
                        (result.bestScore() != null && result.bestScore() >= 0.15);

        if (shouldJudge) {
            judge = judgeService.evaluate(
                    question,
                    result.answer(),
                    chunks.stream().map(OrchestratorResult.Chunk::text).toList()
            );
        }


        return new OrchestratorResult(
                routeUsed,
                result.answer(),
                result.retrievedChunkIds(),
                result.citedChunkIds(),
                chunks,
                result.usedChunks(),
                result.bestScore(),
                reason,
                outcome,
                judge
        );
    }
}