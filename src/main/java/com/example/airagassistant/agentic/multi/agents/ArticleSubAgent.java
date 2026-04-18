package com.example.airagassistant.agentic.multi.agents;

import com.example.airagassistant.LlmClient;
import com.example.airagassistant.agentic.multi.SubAgentResult;
import com.example.airagassistant.agentic.tools.vehicle.FetchVehicleSpecsTool;
import com.example.airagassistant.domain.article.service.ArticleRagService;
import com.example.airagassistant.judge.JudgeResult;
import com.example.airagassistant.judge.JudgeService;
import com.example.airagassistant.rag.RagAnswerService;
import com.example.airagassistant.rag.RetrievalMode;
import com.example.airagassistant.rag.SearchHit;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Sub-agent: Article
 * <p>
 * Owns: article search, article Q&A, vehicle-enriched article content.
 * <p>
 * This agent handles three kinds of requests:
 * <p>
 * 1. Single-article ask — "What did MotorTrend say about the M3's performance?"
 * → Scoped to one articleId, uses ArticleRagService.askArticle
 * (which internally calls RagAnswerService → LLM for synthesis)
 * <p>
 * 2. Cross-article search — "Which car was rated best for value in 2025?"
 * → Searches across all articles, collects chunks, then calls LLM
 * to synthesize a grounded answer from the retrieved content
 * <p>
 * 3. Vehicle-enriched article — "Write a top-ranked vehicle article with specs"
 * → Searches articles, then calls VehicleSubAgent's FetchVehicleSpecsTool
 * to pull in spec data, merges both into context chunks, and calls LLM
 * to synthesize a unified answer
 * <p>
 * LLM usage:
 * Every path calls the LLM for answer synthesis — no template-based responses.
 * This is consistent with VehicleSubAgent (GenerateVehicleSummaryTool calls LLM)
 * and ResearchSubAgent (ResearchTool calls RagAnswerService → LLM).
 * <p>
 * Agent-to-agent communication:
 * The article agent calls FetchVehicleSpecsTool directly — same tool the
 * VehicleSubAgent uses, shared via Spring DI. No circular dependency;
 * both agents depend on the tool, not on each other. This is the "shared
 * tool" pattern for agent-to-agent communication.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class ArticleSubAgent {

    private final ArticleRagService articleRagService;
    private final RagAnswerService ragAnswerService;
    private final FetchVehicleSpecsTool fetchSpecsTool;
    private final LlmClient llm;
    private final JudgeService judgeService;

    private static final int DEFAULT_TOP_K = 5;

    // Must match DefaultResultEvaluationPolicy.MIN_JUDGE_SCORE (0.70)
    // and DivergenceCheckService.MIN_JUDGE_SCORE (0.70)
    // If you change those, change this too.
    private static final double JUDGE_RETRY_THRESHOLD = 0.70;

    public SubAgentResult execute(String task, Map<String, Object> args) {
        String lowerTask = task != null ? task.toLowerCase() : "";

        // Normalize LLM key variations: article_id → articleId, vehicle_query → vehicleQuery
        // LLMs sometimes emit snake_case despite being told camelCase in the prompt
        args = normalizeArgs(args);

        // Route based on task + args
        if (hasArg(args, "articleId")) {
            return executeAskArticle(task, args);
        }

        if (lowerTask.contains("top ranked") || lowerTask.contains("top-ranked")
                || lowerTask.contains("best rated") || lowerTask.contains("highest rated")
                || lowerTask.contains("with specs") || lowerTask.contains("with vehicle")) {
            return executeVehicleEnrichedSearch(task, args);
        }

        if (hasArg(args, "vehicleQuery") || lowerTask.contains("articles about")
                || lowerTask.contains("reviews for") || lowerTask.contains("articles for")) {
            return executeVehicleScopedSearch(task, args);
        }

        // Default: cross-article search
        return executeCrossArticleSearch(task, args);
    }

    /**
     * LLMs sometimes return snake_case keys despite the prompt saying camelCase.
     * Normalize common variations so routing doesn't silently fail.
     *
     * Observed in production logs:
     *   Planner returned {"article_id": "motortrend-bmw-m3-2025-review"}
     *   but hasArg(args, "articleId") returned false → fell through to
     *   executeVehicleEnrichedSearch instead of executeAskArticle.
     */
    private Map<String, Object> normalizeArgs(Map<String, Object> args) {
        if (args == null) return Map.of();
        var normalized = new LinkedHashMap<>(args);
        // article_id → articleId
        if (normalized.containsKey("article_id") && !normalized.containsKey("articleId")) {
            normalized.put("articleId", normalized.remove("article_id"));
        }
        // vehicle_query → vehicleQuery
        if (normalized.containsKey("vehicle_query") && !normalized.containsKey("vehicleQuery")) {
            normalized.put("vehicleQuery", normalized.remove("vehicle_query"));
        }
        // vehicle_ids → vehicleIds
        if (normalized.containsKey("vehicle_ids") && !normalized.containsKey("vehicleIds")) {
            normalized.put("vehicleIds", normalized.remove("vehicle_ids"));
        }
        // vehicle_name → vehicleQuery (supervisor sometimes uses this)
        if (normalized.containsKey("vehicle_name") && !normalized.containsKey("vehicleQuery")) {
            normalized.put("vehicleQuery", normalized.remove("vehicle_name"));
        }
        return normalized;
    }

    // ─── 1. Single-article ask ────────────────────────────────────────────
    //
    //   Uses ArticleRagService.askArticle which internally calls
    //   RagAnswerService.answerWithMode → LLM synthesis. Already LLM-backed.

    private SubAgentResult executeAskArticle(String task, Map<String, Object> args) {
        String articleId = getStringArg(args, "articleId", null);
        if (articleId == null || articleId.isBlank()) {
            return SubAgentResult.failure("article", "articleId is required for single-article ask");
        }

        String question = getStringArg(args, "question", task);
        int topK = getIntArg(args, "topK", DEFAULT_TOP_K);
        RetrievalMode mode = parseMode(getStringArg(args, "mode", "HYBRID"));

        log.info("ArticleSubAgent — asking article '{}': '{}'", articleId, question);

        try {
            RagAnswerService.RagResult result = articleRagService.askArticle(
                    articleId, question, topK, mode);

            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("articleId", articleId);
            metadata.put("operation", "ask_article");
            metadata.put("usedChunks", result.usedChunks());
            metadata.put("bestScore", result.bestScore());

            return SubAgentResult.success(
                    "article", result.answer(),
                    result.citedChunkIds(), result.bestScore(),
                    null, metadata
            );

        } catch (Exception e) {
            log.error("ArticleSubAgent — ask article failed for '{}': {}", articleId, e.getMessage());
            return SubAgentResult.failure("article",
                    "Failed to query article " + articleId + ": " + e.getMessage());
        }
    }

    // ─── 2. Cross-article search ──────────────────────────────────────────
    //
    //   Retrieves article chunks, then passes them to the LLM for synthesis.

    private SubAgentResult executeCrossArticleSearch(String task, Map<String, Object> args) {
        String question = getStringArg(args, "question", task);
        int topK = getIntArg(args, "topK", DEFAULT_TOP_K);

        log.info("ArticleSubAgent — cross-article search: '{}'", question);

        try {
            List<ArticleRagService.ArticleSearchHit> hits =
                    articleRagService.searchAllArticles(question, topK);

            if (hits.isEmpty()) {
                return SubAgentResult.failure("article",
                        "No articles found for: " + question);
            }

            // Build context chunks for LLM synthesis
            List<String> contextChunks = hits.stream()
                    .map(h -> "[" + h.chunkId() + "] " + h.excerpt())
                    .toList();

            // LLM synthesizes a grounded answer from retrieved article chunks
            // Judge evaluates quality; retry if below threshold (same pattern as ResearchSubAgent)
            String synthesizedAnswer = null;
            JudgeResult judgeResult = null;
            String currentQuestion = question;
            int maxAttempts = 2;

            for (int attempt = 1; attempt <= maxAttempts; attempt++) {
                synthesizedAnswer = llm.answer(currentQuestion, contextChunks);
                log.info("ArticleSubAgent — LLM synthesized answer from {} article chunks (attempt {})",
                        hits.size(), attempt);

                judgeResult = judgeService.evaluate(question, synthesizedAnswer, contextChunks);
                log.info("ArticleSubAgent — judge: grounded={} score={} reason='{}'",
                        judgeResult.grounded(), judgeResult.score(), judgeResult.reason());

                if (judgeResult.grounded() && judgeResult.score() >= JUDGE_RETRY_THRESHOLD) {
                    break;
                }

                if (attempt < maxAttempts) {
                    String judgeReason = judgeResult.reason() != null ? judgeResult.reason() : "low quality";

                    // Validate the judge's reason before feeding it back
                    if (validateJudgeReason(judgeReason, contextChunks, synthesizedAnswer)) {
                        log.warn("ArticleSubAgent — judge score {} below threshold {}, reason validated, retrying with: '{}'",
                                judgeResult.score(), JUDGE_RETRY_THRESHOLD, judgeReason);
                        currentQuestion = question
                                + "\n\nJUDGE FEEDBACK ON YOUR PREVIOUS ANSWER: " + judgeReason
                                + "\nFix the issues identified above. Only state facts directly supported by the context chunks.";
                    } else {
                        log.info("ArticleSubAgent — judge reason '{}' not supported by context, skipping retry",
                                judgeReason);
                        break;  // Don't retry with a bad reason — it'll make the answer worse
                    }
                }
            }

            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("operation", "cross_article_search");
            metadata.put("resultCount", hits.size());
            metadata.put("topArticle", hits.get(0).articleId());
            metadata.put("topScore", hits.get(0).score());
            metadata.put("judgeScore", judgeResult != null ? judgeResult.score() : null);
            metadata.put("judgeGrounded", judgeResult != null ? judgeResult.grounded() : null);
            metadata.put("judgeReason", judgeResult != null ? judgeResult.reason() : null);
            // Context chunks for dashboard debugging — truncate each to keep payload manageable
            metadata.put("contextChunks", contextChunks.stream()
                    .map(c -> c.length() > 300 ? c.substring(0, 300) + "…" : c)
                    .toList());
            metadata.put("retrievedArticles", hits.stream()
                    .map(h -> Map.of(
                            "chunkId", h.chunkId(),
                            "articleId", h.articleId(),
                            "score", h.score(),
                            "excerpt", h.excerpt().length() > 150
                                    ? h.excerpt().substring(0, 150) + "…" : h.excerpt()
                    ))
                    .toList());

            List<String> citations = hits.stream()
                    .map(ArticleRagService.ArticleSearchHit::chunkId)
                    .toList();

            return SubAgentResult.success(
                    "article", synthesizedAnswer, citations,
                    hits.get(0).score(), judgeResult, metadata
            );

        } catch (Exception e) {
            log.error("ArticleSubAgent — cross-article search failed: {}", e.getMessage());
            return SubAgentResult.failure("article",
                    "Article search failed: " + e.getMessage());
        }
    }

    // ─── 3. Vehicle-scoped search ─────────────────────────────────────────
    //
    //   Finds articles about a specific vehicle, then LLM-synthesizes.

    private SubAgentResult executeVehicleScopedSearch(String task, Map<String, Object> args) {
        String vehicleQuery = getStringArg(args, "vehicleQuery",
                getStringArg(args, "vehicle", task));
        int topK = getIntArg(args, "topK", DEFAULT_TOP_K);

        log.info("ArticleSubAgent — vehicle-scoped search: '{}'", vehicleQuery);

        try {
            Map<String, List<SearchHit>> articlesByVehicle =
                    articleRagService.searchArticlesByVehicle(vehicleQuery, topK);

            if (articlesByVehicle.isEmpty()) {
                return SubAgentResult.failure("article",
                        "No articles found for vehicle: " + vehicleQuery);
            }

            // Build context from all matched article chunks
            List<String> contextChunks = new ArrayList<>();
            articlesByVehicle.forEach((articleId, articleHits) -> {
                for (SearchHit hit : articleHits) {
                    String text = hit.record().text();
                    if (text.length() > 500) text = text.substring(0, 500);
                    contextChunks.add("[" + hit.record().id() + "] " + text);
                }
            });

            // LLM synthesizes
            String question = "Summarize the articles and reviews about: " + vehicleQuery;
            String synthesizedAnswer = llm.answer(question, contextChunks);
            log.info("ArticleSubAgent — LLM synthesized from {} vehicle-scoped chunks", contextChunks.size());

            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("operation", "vehicle_scoped_search");
            metadata.put("vehicleQuery", vehicleQuery);
            metadata.put("articleCount", articlesByVehicle.size());

            List<String> citations = articlesByVehicle.values().stream()
                    .flatMap(List::stream)
                    .map(h -> h.record().id())
                    .toList();

            return SubAgentResult.success(
                    "article", synthesizedAnswer, citations,
                    1.0, null, metadata
            );

        } catch (Exception e) {
            log.error("ArticleSubAgent — vehicle-scoped search failed: {}", e.getMessage());
            return SubAgentResult.failure("article",
                    "Vehicle article search failed: " + e.getMessage());
        }
    }

    // ─── 4. Vehicle-enriched article search ───────────────────────────────
    //
    //   Three-phase flow:
    //     Phase 1: Search articles (ArticleRagService)
    //     Phase 2: Fetch vehicle specs (FetchVehicleSpecsTool — shared with VehicleSubAgent)
    //     Phase 3: Merge article + spec chunks → LLM synthesis
    //
    //   The supervisor doesn't need to know phases 2-3 happen internally.

    // Define a dedicated executor (typically in a @Configuration class)
// private final ExecutorService agentExecutor = Executors.newVirtualThreadPerTaskExecutor();

    private SubAgentResult executeVehicleEnrichedSearch(String task, Map<String, Object> args) {
        String question = getStringArg(args, "question", task);
        int topK = getIntArg(args, "topK", DEFAULT_TOP_K);

        // Track performance & state for debugging
        long startTime = System.currentTimeMillis();
        List<String> errorLogs = new ArrayList<>();

        try {
            // Phase 1: Search Articles
            List<ArticleRagService.ArticleSearchHit> articleHits = articleRagService.searchAllArticles(question, topK);
            if (articleHits.isEmpty()) {
                return SubAgentResult.failure("article", "No articles found for: " + question);
            }

            // Phase 2: Extraction & Parallel Fetching
            List<String> vehicleIds = extractVehicleIds(articleHits, args);
            Map<String, CompletableFuture<List<FetchVehicleSpecsTool.SpecChunk>>> futures = new LinkedHashMap<>();

            for (String vehicleId : vehicleIds) {
                futures.put(vehicleId, CompletableFuture.supplyAsync(() -> {
                    try {
                        log.debug("Fetching specs for ID: {}", vehicleId);
                        // Refinement: Pass the ID as a strict filter to the tool to avoid cross-pollination
                        List<FetchVehicleSpecsTool.SpecChunk> result = fetchSpecsTool.execute(new FetchVehicleSpecsTool.Input(vehicleId, "specifications engine performance", 3));
                        log.info("Tool returned {} chunks for {}", result.size(), vehicleId);
                        return result;
                    } catch (Exception e) {
                        log.warn("Spec fetch error for {}: {}", vehicleId, e.getMessage());
                        return List.<FetchVehicleSpecsTool.SpecChunk>of();
                    }
                }, Executors.newVirtualThreadPerTaskExecutor())); // Use Virtual Threads for I/O
            }

            // Collect Results with detailed timeout handling
            Map<String, List<FetchVehicleSpecsTool.SpecChunk>> specsByVehicle = new LinkedHashMap<>();
            for (var entry : futures.entrySet()) {
                try {
                    // Per-vehicle timeout (Circuit Breaker)
                    List<FetchVehicleSpecsTool.SpecChunk> specs = entry.getValue().get(10, TimeUnit.SECONDS);
                    if (!specs.isEmpty()) {
                        specsByVehicle.put(entry.getKey(), specs);
                    }
                } catch (TimeoutException e) {
                    errorLogs.add("Timeout: " + entry.getKey());
                    log.error("Ollama/PGVector timeout on vehicle: {}", entry.getKey());
                } catch (Exception e) {
                    errorLogs.add("Error: " + entry.getKey());
                }
            }
            // Prepare the "Source of Truth" list before Synthesis
            List<String> citationChecklist = extractAllCitations(articleHits, specsByVehicle);

            // Phase 3: Merge & Synthesis
            List<String> contextChunks = new ArrayList<>();

            // Add Article Content
            articleHits.forEach(hit -> contextChunks.add(String.format("[%s] %s", hit.chunkId(), hit.excerpt())));

            // Add Specs with "Safe Trimming"
            for (var specEntry : specsByVehicle.entrySet()) {
                for (FetchVehicleSpecsTool.SpecChunk chunk : specEntry.getValue()) {
                    String text = chunk.text();
                    // Improvement: Don't break words. Find the last period before 500 chars.
                    if (text.length() > 500) {
                        int lastPeriod = text.lastIndexOf('.', 500);
                        text = (lastPeriod > 100) ? text.substring(0, lastPeriod + 1) : text.substring(0, 500) + "...";
                    }
                    contextChunks.add(String.format("[%s] (Spec for %s): %s", chunk.chunkId(), specEntry.getKey(), text));
                }
            }

            // Synthesis
            String enrichedQuestion = String.format("""
                    QUESTION: %s
                    
                    INSTRUCTIONS:
                    1. Answer using ONLY the provided context chunks.
                    2. Every fact must be followed by its [ID] tag.
                    3. If specifications are missing, explicitly acknowledge it.
                    
                    STRICT RULES:
                    - You are FORBIDDEN from inventing or guessing [ID] tags.
                    - Only use the IDs explicitly provided in the context.
                    - If a fact is not in a chunk, DO NOT cite it.
                    
                    CRITICAL CORRECTION: 
                    You previously cited [bmw-m3-2025-competition:8] which DOES NOT EXIST in the provided context. 
                    Review your source list carefully and only use IDs like [:2, :4, :7].
                    """, question);

            enrichedQuestion += "\nCRITICAL: Use the FULL ID including the text, like [motortrend-bmw-m3-2025-review:1]. " +
                    "DO NOT shorten it to just [:1] or [1].";


//            String enrichedQuestion = question + "\nInstruction: Use the provided [ID] tags for citations. If specifications are missing for a vehicle, acknowledge it based on the context.";
//            enrichedQuestion += "\nCRITICAL: You cited [bmw-m3-2025-competition:8] which does not exist. " +
//                    "ONLY use IDs provided in the context. If the data is not in a specific chunk, " +
//                    "do not cite it.";

            int maxRetries = 2;
            String synthesizedAnswer = "";

            for (int i = 0; i <= maxRetries; i++) {
                synthesizedAnswer = llm.answer(enrichedQuestion, contextChunks);
                // Inside your synthesis loop
                log.info("LLM Answer: {}", synthesizedAnswer);
                log.info("Valid Citation IDs: {}", citationChecklist);

                // Simple Validation: Did it actually cite anything?
                // And are those citations real?
                if (isValidSynthesis(synthesizedAnswer, citationChecklist)) {
                    break;
                }

                log.warn("Synthesis failed validation (attempt {}/{}). Retrying...", i + 1, maxRetries);
                // Subtle prompt adjustment for the retry
                enrichedQuestion += "\nIMPORTANT: Only use the provided citation tags [ID]. Do not invent IDs.";
            }


            // Judge evaluation — runs AFTER citation validation passes
            // This catches factual errors that citation validation can't see
            // (e.g. LLM says "503hp" but the chunk says "473hp")
            JudgeResult judgeResult = null;
            try {
                judgeResult = judgeService.evaluate(question, synthesizedAnswer, contextChunks);
                log.info("ArticleSubAgent — enriched judge: grounded={} score={} reason='{}'",
                        judgeResult.grounded(), judgeResult.score(), judgeResult.reason());

                // If judge fails, validate the reason before retrying
                if (!judgeResult.grounded() || judgeResult.score() < JUDGE_RETRY_THRESHOLD) {
                    String judgeReason = judgeResult.reason() != null ? judgeResult.reason() : "";

                    // Validate the judge's reason against the actual context chunks.
                    // The judge sometimes says "missing X" when X is present in the chunks.
                    // Blindly feeding a wrong reason back makes the LLM second-guess a correct answer.
                    boolean reasonIsValid = validateJudgeReason(judgeReason, contextChunks, synthesizedAnswer);

                    if (reasonIsValid) {
                        log.warn("ArticleSubAgent — judge score {} below threshold {}, reason validated, retrying",
                                judgeResult.score(), JUDGE_RETRY_THRESHOLD);

                        String retryQuestion = enrichedQuestion
                                + "\n\nJUDGE FEEDBACK ON YOUR PREVIOUS ANSWER: " + judgeReason
                                + "\nFix the issues identified above. Only state facts directly supported by the context chunks."
                                + " If information is not in the context, say so — do not speculate.";

                        log.info("ArticleSubAgent — retry with validated judge feedback: '{}'", judgeReason);
                        synthesizedAnswer = llm.answer(retryQuestion, contextChunks);
                        log.info("ArticleSubAgent — retry LLM answer: {}", synthesizedAnswer);

                        judgeResult = judgeService.evaluate(question, synthesizedAnswer, contextChunks);
                        log.info("ArticleSubAgent — enriched judge retry: grounded={} score={} reason='{}'",
                                judgeResult.grounded(), judgeResult.score(), judgeResult.reason());
                    } else {
                        // Judge reason is contradicted by the context — keep the original answer
                        // but log it so we can tune the judge prompt later
                        log.info("ArticleSubAgent — judge reason '{}' not supported by context, skipping retry. "
                                + "Keeping original answer with score {}",
                                judgeReason, judgeResult.score());
                    }
                }
            } catch (Exception e) {
                log.warn("ArticleSubAgent — judge evaluation failed, proceeding without: {}",
                        e.getMessage());
                // Judge failure is non-fatal — answer is still usable, just unscored
            }

            // Final Metadata & Result
            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("operation", "vehicle_enriched_search");
            metadata.put("latency_ms", System.currentTimeMillis() - startTime);
            metadata.put("specs_attempted", vehicleIds.size());
            metadata.put("specs_resolved", specsByVehicle.size());
            metadata.put("vehicleIds", vehicleIds);
            metadata.put("failed_ids", errorLogs);
            metadata.put("judgeScore", judgeResult != null ? judgeResult.score() : null);
            metadata.put("judgeGrounded", judgeResult != null ? judgeResult.grounded() : null);
            metadata.put("judgeReason", judgeResult != null ? judgeResult.reason() : null);
            // Context chunks for dashboard debugging — truncate to keep payload manageable
            metadata.put("contextChunks", contextChunks.stream()
                    .map(c -> c.length() > 300 ? c.substring(0, 300) + "…" : c)
                    .toList());
            // Article hits with scores
            metadata.put("retrievedArticles", articleHits.stream()
                    .map(h -> Map.of(
                            "chunkId", h.chunkId(),
                            "articleId", h.articleId(),
                            "score", h.score(),
                            "excerpt", h.excerpt().length() > 150
                                    ? h.excerpt().substring(0, 150) + "…" : h.excerpt()
                    ))
                    .toList());
            // Vehicle specs fetched via shared tool
            Map<String, Object> specsDetail = new LinkedHashMap<>();
            specsByVehicle.forEach((vid, chunks) -> {
                specsDetail.put(vid, chunks.stream()
                        .map(c -> Map.of(
                                "chunkId", c.chunkId(),
                                "score", c.score(),
                                "text", c.text().length() > 200
                                        ? c.text().substring(0, 200) + "…" : c.text()
                        ))
                        .toList());
            });
            metadata.put("vehicleSpecs", specsDetail);

            return SubAgentResult.success(
                    "article",
                    synthesizedAnswer,
                    extractAllCitations(articleHits, specsByVehicle),
                    articleHits.get(0).score(),
                    judgeResult,
                    metadata
            );

        } catch (Exception e) {
            log.error("Critical failure in enriched search", e);
            return SubAgentResult.failure("article", "System error: " + e.getMessage());
        }
    }

    private boolean isValidSynthesis(String answer, List<String> validIds) {
        Pattern pattern = Pattern.compile("\\[([^\\]]+)\\]");
        Matcher matcher = pattern.matcher(answer);

        boolean foundAtLeastOneCitation = false;

        while (matcher.find()) {
            foundAtLeastOneCitation = true;
            String citedId = matcher.group(1);

            // Check if the cited ID exists in our source list
            if (!validIds.contains(citedId)) {
                log.warn("Hallucination detected: [{}] is not in valid list", citedId);
                return false;
            }
        }

        // Optional: Fail if the LLM provided zero citations
        // (Uncomment if you want to force at least one citation)
    /*
    if (!foundAtLeastOneCitation && !validIds.isEmpty()) {
        log.warn("No citations found in the answer despite having source data.");
        return false;
    }
    */

        return true;
    }

    /**
     * Validates whether the judge's reason is actually supported by the evidence.
     *
     * The judge is an LLM too — it can hallucinate reasons. Common failure modes:
     *
     *   1. "Missing specifications" — but specs ARE in the context chunks
     *   2. "Incomplete ranking" — but ranking data IS in the chunks
     *   3. "Not enough vehicles" — legitimate, the corpus only has N vehicles
     *   4. "Factually incorrect" — legitimate, the answer contradicts the chunks
     *
     * Cases 3 & 4 are valid reasons to retry. Cases 1 & 2 are false negatives
     * where retrying with bad feedback makes the answer worse.
     *
     * Approach: check if the "missing" keywords from the judge's reason are
     * actually present in the context chunks. If they are, the judge is wrong.
     *
     * This is deliberately conservative — we only skip the retry when we can
     * prove the judge is wrong. If we're unsure, we retry (err on the side
     * of quality).
     */
    private boolean validateJudgeReason(String reason, List<String> contextChunks,
                                         String currentAnswer) {
        if (reason == null || reason.isBlank()) {
            return true;  // No reason given — default to trusting the judge
        }

        String lowerReason = reason.toLowerCase();
        String contextConcat = String.join(" ", contextChunks).toLowerCase();

        // Pattern 1: "missing X" / "no information about X" / "X not found"
        // Check if X is actually in the context
        List<String> missingClaims = extractMissingClaims(lowerReason);
        if (!missingClaims.isEmpty()) {
            int claimsFoundInContext = 0;
            for (String claim : missingClaims) {
                if (contextConcat.contains(claim)) {
                    log.debug("ArticleSubAgent — judge claims '{}' is missing but found in context", claim);
                    claimsFoundInContext++;
                }
            }

            // If ALL "missing" claims are actually present, the judge is wrong
            if (claimsFoundInContext == missingClaims.size()) {
                log.info("ArticleSubAgent — judge reason invalidated: all {} 'missing' claims " +
                        "are present in context", claimsFoundInContext);
                return false;
            }
        }

        // Pattern 2: "insufficient information" when the answer already covers the question well
        if (lowerReason.contains("insufficient") || lowerReason.contains("incomplete")) {
            boolean answerHasCitations = currentAnswer != null && currentAnswer.contains("[");
            boolean answerIsSubstantive = currentAnswer != null && currentAnswer.length() > 200;

            if (answerHasCitations && answerIsSubstantive) {
                log.debug("ArticleSubAgent — judge says 'insufficient' but answer looks substantive, "
                        + "proceeding with retry anyway (conservative)");
            }
        }

        // Default: trust the judge's reason
        return true;
    }

    /**
     * Extracts the subjects of "missing" claims from the judge's reason.
     *
     * Examples:
     *   "Missing specifications" → ["specifications"]
     *   "No information about pricing or ranking" → ["pricing", "ranking"]
     *   "Specifications are missing for a complete answer" → ["specifications"]
     *   "Overall ranking not provided" → ["ranking"]
     */
    private List<String> extractMissingClaims(String lowerReason) {
        List<String> claims = new ArrayList<>();

        // "missing X" pattern
        Matcher m1 = Pattern.compile("missing\\s+(\\w+(?:\\s+\\w+)?)").matcher(lowerReason);
        while (m1.find()) claims.add(m1.group(1).trim());

        // "no information about X" / "no X provided"
        Matcher m2 = Pattern.compile("no\\s+(\\w+(?:\\s+\\w+)?)\\s+(?:provided|found|available|about)")
                .matcher(lowerReason);
        while (m2.find()) claims.add(m2.group(1).trim());

        // "X not provided" / "X not found"
        Matcher m3 = Pattern.compile("(\\w+(?:\\s+\\w+)?)\\s+not\\s+(?:provided|found|mentioned|included)")
                .matcher(lowerReason);
        while (m3.find()) claims.add(m3.group(1).trim());

        // Filter noise words that would match everything
        List<String> noiseWords = List.of("information", "data", "details", "answer",
                "complete", "enough", "sufficient", "it", "the", "a");
        return claims.stream()
                .filter(c -> !noiseWords.contains(c))
                .distinct()
                .toList();
    }


    private List<String> extractAllCitations(
            List<ArticleRagService.ArticleSearchHit> articleHits,
            Map<String, List<FetchVehicleSpecsTool.SpecChunk>> specsByVehicle) {

        List<String> citations = new ArrayList<>();

        // 1. Collect IDs from Article Search Hits
        articleHits.stream()
                .map(ArticleRagService.ArticleSearchHit::chunkId)
                .forEach(citations::add);

        // 2. Collect IDs from Spec Chunks (Flattening the Map)
        specsByVehicle.values().stream()
                .flatMap(List::stream)
                .map(FetchVehicleSpecsTool.SpecChunk::chunkId)
                .forEach(citations::add);

        // 3. Return distinct values in case of overlap
        return citations.stream().distinct().toList();
    }

    // ─── Vehicle ID extraction ────────────────────────────────────────────

    /**
     * Extracts vehicle IDs for spec fetching. Three sources checked in priority order:
     *
     *   1. Explicit args from supervisor (e.g. {"vehicleIds": "bmw-m3-2025-competition"})
     *   2. Chunk text — parses "vehicleId:xxx" tokens embedded in article chunk anchors
     *      by ArticleChunkBuilder. This is the cleanest source because the vehicleId
     *      was written at ingestion time from the actual VehicleReference record.
     *   3. Article ID convention — strips "motortrend-" prefix and article type suffix.
     *      This is a fragile fallback that fails on comparison articles
     *      (e.g. "motortrend-sports-sedan-comparison-2025" → "sports-sedan-comparison-2025"
     *      which is not a vehicle ID). Only used if sources 1 and 2 find nothing.
     *
     * After the ArticleChunkBuilder update, every chunk contains:
     *   "Vehicles: vehicleId:bmw-m3-2025-competition, vehicleId:mercedes-amg-c63-2025."
     * so source 2 will find the correct IDs for both single-vehicle and comparison articles.
     */
    private List<String> extractVehicleIds(
            List<ArticleRagService.ArticleSearchHit> articleHits,
            Map<String, Object> args) {

        List<String> ids = new ArrayList<>();

        // Source 1: Explicit args from supervisor
        Object rawIds = args.get("vehicleIds");
        if (rawIds instanceof List<?> list) {
            list.forEach(id -> ids.add(String.valueOf(id)));
        } else if (rawIds instanceof String s && !s.isBlank()) {
            for (String id : s.split(",")) {
                String trimmed = id.trim();
                if (!trimmed.isEmpty()) ids.add(trimmed);
            }
        }

        if (!ids.isEmpty()) {
            log.debug("ArticleSubAgent — vehicleIds from explicit args: {}", ids);
            return ids.stream().distinct().limit(5).toList();
        }

        // Source 2: Parse "vehicleId:xxx" from chunk text (written by ArticleChunkBuilder)
        Pattern vehicleIdPattern = Pattern.compile("vehicleId:([a-zA-Z0-9\\-]+)");
        for (ArticleRagService.ArticleSearchHit hit : articleHits) {
            String text = hit.excerpt();
            if (text == null) continue;
            Matcher m = vehicleIdPattern.matcher(text);
            while (m.find()) {
                String vid = m.group(1);
                if (!ids.contains(vid)) {
                    ids.add(vid);
                }
            }
        }

        if (!ids.isEmpty()) {
            log.info("ArticleSubAgent — vehicleIds from chunk text: {}", ids);
            return ids.stream().distinct().limit(5).toList();
        }

        // Source 3: Fallback — extract from articleId naming convention
        // Only reaches here for articles ingested before the ArticleChunkBuilder update
        log.debug("ArticleSubAgent — no vehicleIds in chunk text, falling back to articleId parsing");
        for (ArticleRagService.ArticleSearchHit hit : articleHits) {
            String extracted = extractVehicleIdFromArticleId(hit.articleId());
            if (extracted != null && !ids.contains(extracted)) {
                ids.add(extracted);
            }
        }

        return ids.stream().distinct().limit(5).toList();
    }

    /**
     * Fallback: extract vehicle slug from articleId naming convention.
     * Only used when chunk text doesn't contain "vehicleId:xxx" tokens
     * (i.e. articles ingested before the ArticleChunkBuilder update).
     */
    private String extractVehicleIdFromArticleId(String articleId) {
        if (articleId == null || articleId.isBlank()) return null;
        if (articleId.startsWith("motortrend-")) {
            String rest = articleId.substring("motortrend-".length());
            rest = rest.replaceAll("-(review|comparison|first.drive|long.term|buyers.guide)$", "");
            return rest.isBlank() ? null : rest;
        }
        return articleId;
    }

    // ─── Helpers ──────────────────────────────────────────────────────────

    private RetrievalMode parseMode(String mode) {
        try {
            return RetrievalMode.valueOf(mode.toUpperCase());
        } catch (Exception e) {
            return RetrievalMode.HYBRID;
        }
    }

    private boolean hasArg(Map<String, Object> args, String key) {
        if (args == null) return false;
        Object v = args.get(key);
        if (v == null) return false;
        if (v instanceof String s) return !s.isBlank();
        return true;
    }

    private String getStringArg(Map<String, Object> args, String key, String fallback) {
        if (args == null) return fallback;
        Object v = args.get(key);
        if (v == null) return fallback;
        String s = String.valueOf(v).trim();
        return s.isEmpty() ? fallback : s;
    }

    private int getIntArg(Map<String, Object> args, String key, int fallback) {
        if (args == null) return fallback;
        Object v = args.get(key);
        if (v == null) return fallback;
        try {
            return Integer.parseInt(String.valueOf(v).trim());
        } catch (NumberFormatException e) {
            return fallback;
        }
    }
}
