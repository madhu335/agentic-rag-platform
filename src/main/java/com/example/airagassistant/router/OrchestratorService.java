package com.example.airagassistant.router;

import com.example.airagassistant.LlmClient;
import com.example.airagassistant.agent.AgentService;
import com.example.airagassistant.domain.article.service.ArticleRagService;
import com.example.airagassistant.judge.AnswerChecks;
import com.example.airagassistant.judge.AsyncJudgeService;
import com.example.airagassistant.judge.JudgeResult;
import com.example.airagassistant.judge.JudgeService;
import com.example.airagassistant.judge.QuestionRisk;
import com.example.airagassistant.policy.EvaluationDecision;
import com.example.airagassistant.policy.ResultEvaluationPolicy;
import com.example.airagassistant.rag.RagAnswerService;
import com.example.airagassistant.rag.RetrievalMode;
import com.example.airagassistant.trace.TraceHelper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Routes a question, produces an answer and decides whether it is good enough to return.
 *
 * <h3>Judge placement (orchestrator.judge.sync, 2026-11-01)</h3>
 * <b>risky</b> (default): an answer gets the synchronous THOROUGH judge only when it is risky -
 * AGENT / AGENT_FALLBACK route, a numeric / threshold / ranking question ({@link QuestionRisk}), or a deterministic
 * answer check fired ({@link AnswerChecks}: number or quoted text not in the context, trivially short answer).
 * Every other answer is returned immediately and judged in the background by {@link AsyncJudgeService}.
 * <p>
 * RAG route: attempts still run in order BM25 -> HYBRID -> HYBRID_RERANK -> VECTOR -> AGENT_FALLBACK. An attempt
 * below its route's retrieval threshold is skipped without judging; the first attempt above it is returned
 * (non-risky) or judged (risky). A risky attempt the judge rejects moves on to the next mode, as before.
 * Before this change every attempt above 0.15 was judged synchronously, so one hard question could cost up to
 * 5 generations + 5 judge calls before the user saw anything.
 * <p>
 * <b>all</b>: previous behavior, every attempt judged synchronously with JudgeService.evaluate (rollback switch).
 */
@Slf4j
@Service
public class OrchestratorService {

    private static final String SYNC_ALL = "all";

    private final QueryRouter queryRouter;
    private final RagAnswerService ragAnswerService;
    private final AgentService agentService;
    private final JudgeService judgeService;
    private final AsyncJudgeService asyncJudgeService;
    private final TraceHelper traceHelper;
    private final ResultEvaluationPolicy resultEvaluationPolicy;
    private final ArticleRagService articleRagService;
    private final LlmClient llmClient;

    /**
     * Extra AGENT generations when the judge rejects an answer and the policy asks for a retry.
     * Nemotron's answers vary between runs even for an identical prompt, so a fresh generation
     * usually passes. Keeps the best-judged answer if all attempts are rejected. 0 disables.
     */
    private final int agentMaxRetries;
    /** "risky" (default): sync thorough judge only for risky answers, async for the rest. "all": old behavior. */
    private final String judgeSyncMode;

    public OrchestratorService(
            QueryRouter queryRouter,
            RagAnswerService ragAnswerService,
            AgentService agentService,
            JudgeService judgeService,
            AsyncJudgeService asyncJudgeService,
            TraceHelper traceHelper,
            ResultEvaluationPolicy resultEvaluationPolicy,
            ArticleRagService articleRagService,
            LlmClient llmClient,
            @Value("${orchestrator.agent.max-retries:1}") int agentMaxRetries,
            @Value("${orchestrator.judge.sync:risky}") String judgeSyncMode
    ) {
        this.queryRouter = queryRouter;
        this.ragAnswerService = ragAnswerService;
        this.agentService = agentService;
        this.judgeService = judgeService;
        this.asyncJudgeService = asyncJudgeService;
        this.traceHelper = traceHelper;
        this.resultEvaluationPolicy = resultEvaluationPolicy;
        this.articleRagService = articleRagService;
        this.llmClient = llmClient;
        this.agentMaxRetries = agentMaxRetries;
        this.judgeSyncMode = judgeSyncMode == null ? "risky" : judgeSyncMode.trim().toLowerCase();
        log.info("OrchestratorService: judge sync mode={} agentMaxRetries={}", this.judgeSyncMode, agentMaxRetries);
    }

    private record RagMode(String route, RetrievalMode mode, String outcome) {}

    private static final List<RagMode> RAG_CHAIN = List.of(
            new RagMode("BM25", RetrievalMode.BM25, "bm25_success"),
            new RagMode("HYBRID", RetrievalMode.HYBRID, "hybrid_success"),
            new RagMode("HYBRID_RERANK", RetrievalMode.HYBRID_RERANK, "hybrid_rerank_success"),
            new RagMode("VECTOR", RetrievalMode.VECTOR, "vector_success")
    );

    public OrchestratorResult handle(String question, String docType, String docId, int k) {
        if (("article".equalsIgnoreCase(docType) || "article_all".equalsIgnoreCase(docType))
                && "articles".equalsIgnoreCase(docId)) {
            return handleGlobalArticles(question, k);
        }

        RouteDecision decision = queryRouter.route(question);

        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.span.kind", "chain");
        attrs.put("langsmith.metadata.doc_type", docType);
        attrs.put("langsmith.metadata.doc_id", docId);
        attrs.put("langsmith.metadata.route", decision.route().name());
        attrs.put("langsmith.metadata.route_reason", decision.reason());
        attrs.put("langsmith.metadata.route_confidence", decision.confidence());
        attrs.put("langsmith.metadata.judge_sync_mode", judgeSyncMode);

        return traceHelper.run("orchestrator-flow", attrs, () -> {
            if (decision.route() == QueryRouter.Route.RAG) {
                for (RagMode m : RAG_CHAIN) {
                    OrchestratorResult attempt = runAttempt(
                            m.route(),
                            () -> ragAnswerService.answerWithMode(docType, docId, question, k, m.mode()),
                            decision.reason(),
                            m.outcome()
                    );

                    // Retrieval gate first (judge == null here): below the route threshold -> next mode, no judge.
                    EvaluationDecision retrievalDecision = resultEvaluationPolicy.evaluateOrchestratorResult(attempt);
                    if (!retrievalDecision.acceptable()) {
                        log.info("MODE={} skipped: {}", m.route(), retrievalDecision.reason());
                        continue;
                    }

                    OrchestratorResult judged = judgeOrDefer(attempt, question);
                    EvaluationDecision finalDecision = resultEvaluationPolicy.evaluateOrchestratorResult(judged);
                    log.info("MODE={} retrievalScore={} judge={} accepted={} reason={}",
                            m.route(), judged.retrievalScore(), describe(judged.judge()),
                            finalDecision.acceptable(), finalDecision.reason());
                    if (finalDecision.acceptable()) {
                        markFinal(m.route(), judged, finalDecision);
                        return judged;
                    }
                }

                OrchestratorResult agentFallback = judgeSync(runAttempt(
                        "AGENT_FALLBACK",
                        () -> agentService.answer(docType, docId, question, k),
                        decision.reason(),
                        "all_retrievers_failed"
                ), question, "agent_route");
                EvaluationDecision agentFallbackDecision = resultEvaluationPolicy.evaluateOrchestratorResult(agentFallback);
                markFinal("AGENT_FALLBACK", agentFallback, agentFallbackDecision);
                return agentFallback;
            }

            OrchestratorResult agent = judgeSync(runAttempt(
                    "AGENT",
                    () -> agentService.answer(docType, docId, question, k),
                    decision.reason(),
                    "agent_direct"
            ), question, "agent_route");
            EvaluationDecision agentDecision = resultEvaluationPolicy.evaluateOrchestratorResult(agent);

            int attempt = 1;
            while (!agentDecision.acceptable() && agentDecision.shouldRetry() && attempt <= agentMaxRetries) {
                attempt++;
                log.info("AGENT answer rejected ({}, judge={}); regenerating, attempt {}/{}",
                        agentDecision.reason(), judgeScore(agent), attempt, agentMaxRetries + 1);
                OrchestratorResult retry = judgeSync(runAttempt(
                        "AGENT",
                        () -> agentService.answer(docType, docId, question, k),
                        agentDecision.reason(),
                        "agent_retry"
                ), question, "agent_route");
                EvaluationDecision retryDecision = resultEvaluationPolicy.evaluateOrchestratorResult(retry);
                if (retryDecision.acceptable() || judgeScore(retry) > judgeScore(agent)) {
                    agent = retry;
                    agentDecision = retryDecision;
                }
            }

            markFinal("AGENT", agent, agentDecision);
            return agent;
        });
    }

    // ── judge placement ──────────────────────────────────────────────────────────────────────────────

    /**
     * Risky (or sync mode "all") -> synchronous judge attached to the result.
     * Otherwise -> background judge, result returned unjudged (policy accepts it on retrieval score).
     */
    private OrchestratorResult judgeOrDefer(OrchestratorResult result, String question) {
        if (SYNC_ALL.equals(judgeSyncMode)) {
            return judgeSync(result, question, "sync_mode_all");
        }
        List<String> riskReasons = riskReasons(result, question);
        if (!riskReasons.isEmpty()) {
            return judgeSync(result, question, String.join("+", riskReasons));
        }
        asyncJudgeService.evaluateAsync(question, result.answer(), judgeContext(result.chunks()), result.routeUsed());
        traceHelper.addAttributes(Map.of(
                "langsmith.metadata.judge.placement", "async",
                "langsmith.metadata.judge.risk", "none"
        ));
        return result;
    }

    private List<String> riskReasons(OrchestratorResult result, String question) {
        List<String> reasons = new ArrayList<>();
        String route = result.routeUsed();
        if ("AGENT".equals(route) || "AGENT_FALLBACK".equals(route)) {
            reasons.add("agent_route");
        }
        String questionRisk = QuestionRisk.reason(question);
        if (questionRisk != null) {
            reasons.add(questionRisk);
        }
        AnswerChecks.Result checks = AnswerChecks.check(question, result.answer(), judgeContext(result.chunks()));
        if (checks.fired()) {
            reasons.add(checks.reason());
        }
        return reasons;
    }

    /** Synchronous judge: thorough in "risky" mode, JudgeService.evaluate (old behavior) in "all" mode. */
    private OrchestratorResult judgeSync(OrchestratorResult result, String question, String riskReason) {
        List<String> context = judgeContext(result.chunks());
        JudgeResult judge = SYNC_ALL.equals(judgeSyncMode)
                ? judgeService.evaluate(question, result.answer(), context)
                : judgeService.evaluateThorough(question, result.answer(), context);

        Map<String, Object> judgeAttrs = new LinkedHashMap<>();
        judgeAttrs.put("langsmith.metadata.judge.placement", "sync");
        judgeAttrs.put("langsmith.metadata.judge.risk", riskReason);
        judgeAttrs.put("langsmith.metadata.judge.verdict", judge.verdict().name());
        judgeAttrs.put("langsmith.metadata.judge.score", judge.score());
        judgeAttrs.put("langsmith.metadata.judge.stage", judge.stage());
        judgeAttrs.put("langsmith.metadata.judge.reason", judge.reason());
        traceHelper.addAttributes(judgeAttrs);
        log.info("Sync judge: route={} risk={} verdict={} score={} stage={}",
                result.routeUsed(), riskReason, judge.verdict(), judge.score(), judge.stage());

        return withJudge(result, judge);
    }

    private static OrchestratorResult withJudge(OrchestratorResult r, JudgeResult judge) {
        return new OrchestratorResult(r.routeUsed(), r.answer(), r.retrievedChunkIds(), r.citedChunkIds(), r.cards(),
                r.chunks(), r.usedChunks(), r.retrievalScore(), r.reason(), r.outcome(), judge);
    }

    private static double judgeScore(OrchestratorResult result) {
        return result.judge() == null ? -1.0 : result.judge().score();
    }

    private static String describe(JudgeResult judge) {
        return judge == null ? "async" : judge.verdict() + "/" + judge.score();
    }

    private void markFinal(String mode, OrchestratorResult result, EvaluationDecision decision) {
        traceHelper.addAttributes(Map.of(
                "langsmith.metadata.final_mode", mode,
                "langsmith.metadata.final_outcome", result.outcome(),
                "langsmith.metadata.final_reason", decision.reason()
        ));
    }

    // ── attempts ─────────────────────────────────────────────────────────────────────────────────────

    /** Generates one answer (no judge) and traces retrieval metadata. */
    private OrchestratorResult runAttempt(
            String routeUsed,
            Supplier<RagAnswerService.RagResult> supplier,
            String reason,
            String outcome
    ) {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.span.kind", "chain");
        attrs.put("langsmith.metadata.attempt_mode", routeUsed);
        attrs.put("langsmith.metadata.attempt_reason", reason);

        return traceHelper.run("retrieval-attempt-" + routeUsed.toLowerCase(), attrs, () -> {
            OrchestratorResult result = buildRagResult(routeUsed, supplier.get(), reason, outcome);

            Map<String, Object> resultAttrs = new LinkedHashMap<>();
            resultAttrs.put("langsmith.metadata.attempt_mode", routeUsed);
            resultAttrs.put("langsmith.metadata.outcome", result.outcome());
            if (result.retrievalScore() != null) {
                resultAttrs.put("langsmith.metadata.best_score", result.retrievalScore());
            }
            resultAttrs.put("langsmith.metadata.used_chunks", result.usedChunks());
            resultAttrs.put("langsmith.metadata.retrieved_chunk_count",
                    result.retrievedChunkIds() != null ? result.retrievedChunkIds().size() : 0);
            traceHelper.addAttributes(resultAttrs);

            log.info("MODE={} generated: retrievalScore={} usedChunks={}",
                    routeUsed, result.retrievalScore(), result.usedChunks());
            return result;
        });
    }

    private OrchestratorResult buildRagResult(
            String routeUsed,
            RagAnswerService.RagResult result,
            String reason,
            String outcome
    ) {
        var chunks = result.chunks().stream()
                .map(c -> new OrchestratorResult.Chunk(c.id(), c.text()))
                .toList();

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
                null
        );
    }

    /**
     * Context for the judge keeps each chunk's id next to its text, so the judge can
     * verify [chunk-id] citations in the answer instead of flagging them as missing.
     */
    private List<String> judgeContext(List<OrchestratorResult.Chunk> chunks) {
        if (chunks == null) {
            return List.of();
        }
        return chunks.stream()
                .map(c -> "[" + c.id() + "] " + c.text())
                .toList();
    }

    // ── global article route ─────────────────────────────────────────────────────────────────────────

    private OrchestratorResult handleGlobalArticles(String question, int k) {
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
                .limit(4)   // reduce tokens
                .toList();
        String groundedPrompt = buildGroundedPrompt(question, contextChunks);
        String answer = llmClient.answer(groundedPrompt, contextChunks);

        var citedChunkIds = hits.stream()
                .map(ArticleRagService.ArticleSearchHit::chunkId)
                .toList();

        var chunks = hits.stream()
                .map(h -> new OrchestratorResult.Chunk(h.chunkId(), h.excerpt()))
                .toList();

        OrchestratorResult result = new OrchestratorResult(
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
                null
        );

        // Weak retrieval (as before: best score < 0.12) -> judge synchronously; otherwise same risk rules.
        if (hits.get(0).score() < 0.12) {
            return judgeSync(result, question, "weak_article_retrieval");
        }
        return judgeOrDefer(result, question);
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
