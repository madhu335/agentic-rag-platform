package com.example.airagassistant.router;

import com.example.airagassistant.agent.AgentService;
import com.example.airagassistant.judge.JudgeResult;
import com.example.airagassistant.judge.JudgeService;
import com.example.airagassistant.policy.EvaluationDecision;
import com.example.airagassistant.policy.ResultEvaluationPolicy;
import com.example.airagassistant.rag.RagAnswerService;
import com.example.airagassistant.rag.RetrievalMode;
import com.example.airagassistant.trace.TraceHelper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class OrchestratorService {

    private final QueryRouter queryRouter;
    private final RagAnswerService ragAnswerService;
    private final AgentService agentService;
    private final JudgeService judgeService;
    private final TraceHelper traceHelper;
    private final ResultEvaluationPolicy resultEvaluationPolicy;
    private final com.example.airagassistant.domain.article.service.ArticleRagService articleRagService;
    private final com.example.airagassistant.LlmClient llmClient;

    public OrchestratorResult handle(String question, String docType, String docId, int k) {
        if (("article".equalsIgnoreCase(docType) || "article_all".equalsIgnoreCase(docType))
                && "articles".equalsIgnoreCase(docId)) {

            log.info("Routing global article query to ArticleRagService");

            var hits = articleRagService.searchAllArticles(question, k);

            if (hits.isEmpty()) {
                return new OrchestratorResult(
                        "ARTICLE_GLOBAL",
                        "I don't know based on the ingested documents.",
                        List.of(),
                        List.of(),
                        List.of(),
                        List.of(),
                        0,
                        null,
                        "global_article_route",
                        "article_global_no_hits",
                        null
                );
            }

            var contextChunks = hits.stream()
                    .map(h -> "[" + h.chunkId() + "] " + h.excerpt())
                    .limit(4)   // 🔥 reduce tokens
                    .toList();
            String groundedPrompt = buildGroundedPrompt(question, contextChunks);
            String answer = llmClient.answer(groundedPrompt, contextChunks);

            var citedChunkIds = hits.stream()
                    .map(com.example.airagassistant.domain.article.service.ArticleRagService.ArticleSearchHit::chunkId)
                    .toList();

            var chunks = hits.stream()
                    .map(h -> new OrchestratorResult.Chunk(h.chunkId(), h.excerpt()))
                    .toList();

            JudgeResult judge = null;

            if (hits.get(0).score() < 0.12) {
                judge = judgeService.evaluate(
                        question,
                        answer,
                        chunks.stream().map(OrchestratorResult.Chunk::text).toList()
                );
            }

            return new OrchestratorResult(
                    "ARTICLE_GLOBAL",
                    answer,
                    citedChunkIds,
                    citedChunkIds,
                    List.of(),
                    chunks,
                    contextChunks.size(),
                    hits.get(0).score(),
                    "global_article_route",
                    "article_global_success",
                    judge
            );
        }

        RouteDecision decision = queryRouter.route(question);

        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.span.kind", "chain");
        attrs.put("langsmith.metadata.doc_type", docType);
        attrs.put("langsmith.metadata.doc_id", docId);
        attrs.put("langsmith.metadata.route", decision.route().name());
        attrs.put("langsmith.metadata.route_reason", decision.reason());
        attrs.put("langsmith.metadata.route_confidence", decision.confidence());

        return traceHelper.run("orchestrator-flow", attrs, () -> {
            if (decision.route() == QueryRouter.Route.RAG) {

                var bm25 = runAttempt(
                        "BM25",
                        () -> ragAnswerService.answerWithMode(docType, docId, question, k, RetrievalMode.BM25),
                        decision.reason(),
                        "bm25_success",
                        question
                );
                EvaluationDecision bm25Decision = resultEvaluationPolicy.evaluateOrchestratorResult(bm25);
                if (bm25Decision.acceptable()) {
                    traceHelper.addAttributes(Map.of(
                            "langsmith.metadata.final_mode", "BM25",
                            "langsmith.metadata.final_outcome", bm25.outcome(),
                            "langsmith.metadata.final_reason", bm25Decision.reason()
                    ));
                    return bm25;
                }

                var hybrid = runAttempt(
                        "HYBRID",
                        () -> ragAnswerService.answerWithMode(docType, docId, question, k, RetrievalMode.HYBRID),
                        decision.reason(),
                        "hybrid_success",
                        question
                );
                EvaluationDecision hybridDecision = resultEvaluationPolicy.evaluateOrchestratorResult(hybrid);
                if (hybridDecision.acceptable()) {
                    traceHelper.addAttributes(Map.of(
                            "langsmith.metadata.final_mode", "HYBRID",
                            "langsmith.metadata.final_outcome", hybrid.outcome(),
                            "langsmith.metadata.final_reason", hybridDecision.reason()
                    ));
                    return hybrid;
                }

                var hybridRerank = runAttempt(
                        "HYBRID_RERANK",
                        () -> ragAnswerService.answerWithMode(docType, docId, question, k, RetrievalMode.HYBRID_RERANK),
                        decision.reason(),
                        "hybrid_rerank_success",
                        question
                );
                EvaluationDecision hybridRerankDecision = resultEvaluationPolicy.evaluateOrchestratorResult(hybridRerank);
                if (hybridRerankDecision.acceptable()) {
                    traceHelper.addAttributes(Map.of(
                            "langsmith.metadata.final_mode", "HYBRID_RERANK",
                            "langsmith.metadata.final_outcome", hybridRerank.outcome(),
                            "langsmith.metadata.final_reason", hybridRerankDecision.reason()
                    ));
                    return hybridRerank;
                }

                var vector = runAttempt(
                        "VECTOR",
                        () -> ragAnswerService.answerWithMode(docType, docId, question, k, RetrievalMode.VECTOR),
                        decision.reason(),
                        "vector_success",
                        question
                );
                EvaluationDecision vectorDecision = resultEvaluationPolicy.evaluateOrchestratorResult(vector);
                if (vectorDecision.acceptable()) {
                    traceHelper.addAttributes(Map.of(
                            "langsmith.metadata.final_mode", "VECTOR",
                            "langsmith.metadata.final_outcome", vector.outcome(),
                            "langsmith.metadata.final_reason", vectorDecision.reason()
                    ));
                    return vector;
                }

                var agentFallback = runAttempt(
                        "AGENT_FALLBACK",
                        () -> agentService.answer(docType, docId, question, k),
                        decision.reason(),
                        "all_retrievers_failed",
                        question
                );

                EvaluationDecision agentFallbackDecision = resultEvaluationPolicy.evaluateOrchestratorResult(agentFallback);

                traceHelper.addAttributes(Map.of(
                        "langsmith.metadata.final_mode", "AGENT_FALLBACK",
                        "langsmith.metadata.final_outcome", agentFallback.outcome(),
                        "langsmith.metadata.final_reason", agentFallbackDecision.reason()
                ));
                return agentFallback;
            }

            var agent = runAttempt(
                    "AGENT",
                    () -> agentService.answer(docType, docId, question, k),
                    decision.reason(),
                    "agent_direct",
                    question
            );

            EvaluationDecision agentDecision = resultEvaluationPolicy.evaluateOrchestratorResult(agent);

            traceHelper.addAttributes(Map.of(
                    "langsmith.metadata.final_mode", "AGENT",
                    "langsmith.metadata.final_outcome", agent.outcome(),
                    "langsmith.metadata.final_reason", agentDecision.reason()
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
            EvaluationDecision decision = resultEvaluationPolicy.evaluateOrchestratorResult(result);

            Map<String, Object> resultAttrs = new LinkedHashMap<>();
            resultAttrs.put("langsmith.metadata.attempt_mode", routeUsed);
            resultAttrs.put("langsmith.metadata.accepted", decision.acceptable());
            resultAttrs.put("langsmith.metadata.retry", decision.shouldRetry());
            resultAttrs.put("langsmith.metadata.decision_reason", decision.reason());
            resultAttrs.put("langsmith.metadata.outcome", result.outcome());

            if (result.retrievalScore() != null) {
                resultAttrs.put("langsmith.metadata.best_score", result.retrievalScore());
            }

            resultAttrs.put("langsmith.metadata.used_chunks", result.usedChunks());
            resultAttrs.put("langsmith.metadata.retrieved_chunk_count",
                    result.retrievedChunkIds() != null ? result.retrievedChunkIds().size() : 0);

            if (result.judge() != null) {
                resultAttrs.put("langsmith.metadata.judge.score", result.judge().score());
                resultAttrs.put("langsmith.metadata.judge.grounded", result.judge().grounded());
                resultAttrs.put("langsmith.metadata.judge.correct", result.judge().correct());
                resultAttrs.put("langsmith.metadata.judge.complete", result.judge().complete());
                resultAttrs.put("langsmith.metadata.judge.reason", result.judge().reason());
            } else {
                resultAttrs.put("langsmith.metadata.judge.skipped", true);
            }

            log.info("MODE={}, retrievalScore={}, accepted={}, retry={}, reason={}",
                    result.routeUsed(),
                    result.retrievalScore(),
                    decision.acceptable(),
                    decision.shouldRetry(),
                    decision.reason());

            traceHelper.addAttributes(resultAttrs);
            return result;
        });
    }

    private OrchestratorResult buildRagResult(
            String routeUsed,
            RagAnswerService.RagResult result,
            String reason,
            String outcome,
            String question
    ) {
        var chunks = result.chunks().stream()
                .map(c -> new OrchestratorResult.Chunk(c.id(), c.text()))
                .toList();

        JudgeResult judge = null;
        boolean shouldJudge =
                "AGENT".equals(routeUsed) ||
                        "AGENT_FALLBACK".equals(routeUsed) ||
                        (result.retrievalScore() != null && result.retrievalScore() >= 0.15);

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
                result.cards(),
                chunks,
                result.usedChunks(),
                result.retrievalScore(),
                reason,
                outcome,
                judge
        );
    }

    private String buildGroundedPrompt(String question, List<String> contextChunks) {
        StringBuilder sb = new StringBuilder();

        sb.append("You are a precise automotive assistant.\n");
        sb.append("Answer ONLY from the provided context.\n");
        sb.append("Every claim must include at least one chunk citation like [chunk_id].\n");
        sb.append("Every sentence must end with at least one citation like [chunk_id].\n");
        sb.append("Be specific. If the question asks for the best models, name them directly and explain why using the cited evidence.\n");
        sb.append("When listing best vehicles, give one short evidence-backed reason for each vehicle.\n");
        sb.append("Do not use vague phrases like 'part of the comparison' without saying what that implies.\n");
        sb.append("Do not say 'based on the articles' or similar filler unless necessary.\n\n");

        sb.append("Format:\n");
        sb.append("1. Vehicle name — why it stands out [chunk_id]\n");
        sb.append("2. Vehicle name — why it stands out [chunk_id]\n");
        sb.append("3. Vehicle name — why it stands out [chunk_id]\n\n");

        sb.append("Context:\n");
        for (String chunk : contextChunks) {
            sb.append(chunk).append("\n");
        }

        sb.append("\nQuestion:\n").append(question).append("\n\n");
        sb.append("Answer:\n");

        return sb.toString();
    }
}