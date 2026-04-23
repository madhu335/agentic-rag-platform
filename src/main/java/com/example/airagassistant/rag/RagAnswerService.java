package com.example.airagassistant.rag;

import com.example.airagassistant.LlmClient;
import com.example.airagassistant.rag.VehicleCardDto;
import com.example.airagassistant.agentic.dto.ChunkDto;
import com.example.airagassistant.agentic.mapper.VehicleSummaryMapper;
import com.example.airagassistant.rag.retrieval.VehicleSummaryService;
import com.example.airagassistant.trace.TraceHelper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.function.Consumer;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class RagAnswerService {

    private static final double COSINE_LOW = 0.40;
    private static final double COSINE_HIGH = 0.70;

    private static final double RRF_LOW = 0.04;
    private static final double RRF_HIGH = 0.08;

    private static final String FALLBACK = "I don't know based on the ingested documents.";

    private final RagRetriever ragRetriever;
    private final ReRankService reRankService;
    private final LlmClient llm;
    private final TraceHelper traceHelper;
    private final VehicleSummaryService vehicleSummaryService;
    private final VehicleSummaryMapper summaryMapper;

    public RagResult answerWithMode(String docType, String docId, String question, int topK, RetrievalMode mode) {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.metadata.doc_type", docType);
        attrs.put("langsmith.metadata.doc_id", docId);
        attrs.put("gen_ai.prompt.0.content", question);
        attrs.put("langsmith.metadata.top_k", topK);
        attrs.put("langsmith.metadata.retrieval_mode", mode.name());

        return traceHelper.run("rag-answer-" + mode.name().toLowerCase(), attrs, () -> {
            List<SearchHit> hits = traceHelper.run(
                    "retrieve-chunks-" + mode.name().toLowerCase(),
                    Map.of(
                            "langsmith.metadata.doc_type", docType,
                            "langsmith.metadata.doc_id", docId,
                            "langsmith.metadata.top_k", topK,
                            "langsmith.metadata.retrieval_mode", mode.name()
                    ),
                    () -> {
                        List<SearchHit> result = ragRetriever.retrieve(docId, question, topK, mode);
                        traceHelper.addAttributes(Map.of("retrieved.count", result.size()));
                        return result;
                    }
            );

            List<SearchHit> rankedHits = hits;

            if (mode != RetrievalMode.HYBRID_RERANK) {
                rankedHits = traceHelper.run(
                        "rerank",
                        Map.of("langsmith.metadata.retrieval_mode", mode.name()),
                        () -> {
                            List<SearchHit> result = reRankService.rerank(question, hits);
                            traceHelper.addAttributes(Map.of(
                                    "input.count", hits.size(),
                                    "output.count", result.size()
                            ));
                            return result;
                        }
                );
            }

            Double retrievalScore = rankedHits.isEmpty() ? null : rankedHits.get(0).score();
            addBestScore(retrievalScore);

            if (rankedHits.isEmpty()) {
                addFallbackAttributes("no_hits");
                return fallback(retrievalScore);
            }

            String topText = rankedHits.get(0).record().text();

            if (!passesKeywordGuard(question, retrievalScore, topText)) {
                addFallbackAttributes("keyword_guard");
                return fallback(retrievalScore);
            }

            List<SearchHit> usableHits = rankedHits.stream()
                    .limit(3)
                    .toList();

            String citedQuestion = buildStrictPrompt(question);

            ContextBuildResult contextResult = traceHelper.run(
                    "build-context",
                    null,
                    () -> {
                        List<String> contextChunks = buildCitedContext(usableHits);
                        List<String> retrievedChunkIds = extractChunkIds(usableHits);

                        Map<String, Object> resultAttrs = new LinkedHashMap<>();
                        resultAttrs.put("gen_ai.completion.0.usable.count", usableHits.size());
                        resultAttrs.put("gen_ai.completion.0.context.count", contextChunks.size());
                        traceHelper.addAttributes(resultAttrs);

                        return new ContextBuildResult(contextChunks, retrievedChunkIds);
                    }
            );

            String answer = traceHelper.run(
                    "llm-call",
                    buildLlmAttributes(question, citedQuestion, contextResult.contextChunks().size()),
                    () -> {
                        String result = llm.answer(citedQuestion, contextResult.contextChunks());

                        Map<String, Object> llmAttrs = new LinkedHashMap<>();
                        llmAttrs.put("gen_ai.completion.0.response.length", result != null ? result.length() : 0);
                        traceHelper.addAttributes(llmAttrs);

                        return result;
                    }
            );

            answer = removeInvalidCitations(answer, contextResult.retrievedChunkIds());
            answer = cleanAnswer(answer);
            List<String> citedChunkIds = filterValidCitations(answer, contextResult.retrievedChunkIds());

            Map<String, Object> finalAttrs = new LinkedHashMap<>();
            finalAttrs.put("gen_ai.completion.0.content", answer);
            finalAttrs.put("gen_ai.completion.0.used.chunks", contextResult.contextChunks().size());
            finalAttrs.put("gen_ai.completion.0.cited.count", citedChunkIds.size());
            traceHelper.addAttributes(finalAttrs);

            return new RagResult(
                    answer,
                    buildVehicleCards(docType, docId, question, topK),
                    toChunkDtos(usableHits),
                    citedChunkIds,
                    contextResult.retrievedChunkIds(),
                    contextResult.contextChunks().size(),
                    retrievalScore
            );
        });
    }

    public RagResult answer(String docType, String docId, String question, int topK) {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.metadata.doc_type", docType);
        attrs.put("langsmith.metadata.doc_id", docId);
        attrs.put("gen_ai.prompt.0.content", question);
        attrs.put("langsmith.metadata.top_k", topK);

        return traceHelper.run("rag-answer", attrs, () -> {
            List<SearchHit> hits = traceHelper.run(
                    "retrieve-chunks",
                    Map.of(
                            "langsmith.metadata.doc_type", docType,
                            "langsmith.metadata.doc_id", docId,
                            "langsmith.metadata.top_k", topK
                    ),
                    () -> {
                        List<SearchHit> result = ragRetriever.retrieve(docId, question, topK);
                        traceHelper.addAttributes(Map.of("retrieved.count", result.size()));
                        return result;
                    }
            );

            List<SearchHit> reranked = traceHelper.run(
                    "rerank",
                    null,
                    () -> {
                        List<SearchHit> result = reRankService.rerank(question, hits);
                        traceHelper.addAttributes(Map.of(
                                "input.count", hits.size(),
                                "output.count", result.size()
                        ));
                        return result;
                    }
            );

            Double retrievalScore = reranked.isEmpty() ? null : reranked.get(0).score();
            addBestScore(retrievalScore);

            if (reranked.isEmpty()) {
                addFallbackAttributes("no_hits");
                return fallback(retrievalScore);
            }

            String topText = reranked.get(0).record().text();

            if (!passesKeywordGuard(question, retrievalScore, topText)) {
                addFallbackAttributes("keyword_guard");
                return fallback(retrievalScore);
            }

            List<SearchHit> usableHits = reranked.stream()
                    .limit(3)
                    .toList();

            String citedQuestion = buildDefaultPrompt(question);

            ContextBuildResult contextResult = traceHelper.run(
                    "build-context",
                    null,
                    () -> {
                        List<String> contextChunks = buildCitedContext(usableHits);
                        List<String> retrievedChunkIds = extractChunkIds(usableHits);

                        Map<String, Object> resultAttrs = new LinkedHashMap<>();
                        resultAttrs.put("gen_ai.completion.0.usable.count", usableHits.size());
                        resultAttrs.put("gen_ai.completion.0.context.count", contextChunks.size());
                        traceHelper.addAttributes(resultAttrs);

                        return new ContextBuildResult(contextChunks, retrievedChunkIds);
                    }
            );

            String answer = traceHelper.run(
                    "llm-call",
                    buildLlmAttributes(question, citedQuestion, contextResult.contextChunks().size()),
                    () -> {
                        String result = llm.answer(citedQuestion, contextResult.contextChunks());

                        Map<String, Object> llmAttrs = new LinkedHashMap<>();
                        llmAttrs.put("gen_ai.completion.0.response.length", result != null ? result.length() : 0);
                        traceHelper.addAttributes(llmAttrs);

                        return result;
                    }
            );

            answer = removeInvalidCitations(answer, contextResult.retrievedChunkIds());
            answer = cleanAnswer(answer);
            List<String> citedChunkIds = filterValidCitations(answer, contextResult.retrievedChunkIds());

            Map<String, Object> finalAttrs = new LinkedHashMap<>();
            finalAttrs.put("gen_ai.completion.0.content", answer);
            finalAttrs.put("gen_ai.completion.0.used.chunks", contextResult.contextChunks().size());
            finalAttrs.put("gen_ai.completion.0.cited.count", citedChunkIds.size());
            traceHelper.addAttributes(finalAttrs);

            return new RagResult(
                    answer,
                    buildVehicleCards(docType, docId, question, topK),
                    toChunkDtos(usableHits),
                    citedChunkIds,
                    contextResult.retrievedChunkIds(),
                    contextResult.contextChunks().size(),
                    retrievalScore
            );
        });
    }

    public void streamAnswer(String docType, String docId, String question, int topK, Consumer<String> onToken) {
        streamAnswerEvents(docType, docId, question, topK, event -> {
            if ("token".equals(event.type()) && event.payload() instanceof TokenPayload tokenPayload) {
                onToken.accept(tokenPayload.value());
            }
        });
    }

    public void streamAnswerEvents(String docType, String docId, String question, int topK, Consumer<StreamEvent> onEvent) {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.metadata.doc_type", docType);
        attrs.put("langsmith.metadata.doc_id", docId);
        attrs.put("gen_ai.prompt.0.content", question);
        attrs.put("langsmith.metadata.top_k", topK);

        onEvent.accept(new StreamEvent("status", new StatusPayload("retrieving", "Retrieving relevant chunks")));

        traceHelper.run("rag-answer-stream", attrs, () -> {
            List<VehicleCardDto> cards = buildVehicleCards(docType, docId, question, topK);
            if (cards == null) {
                cards = List.of();
            }

            Map<String, Object> sourcesPayload = new LinkedHashMap<>();
            sourcesPayload.put("cards", cards);
            onEvent.accept(new StreamEvent("sources", sourcesPayload));

            List<SearchHit> hits = traceHelper.run(
                    "retrieve-chunks",
                    Map.of(
                            "langsmith.metadata.doc_type", docType,
                            "langsmith.metadata.doc_id", docId,
                            "langsmith.metadata.top_k", topK
                    ),
                    () -> {
                        List<SearchHit> result = ragRetriever.retrieve(docId, question, topK);
                        traceHelper.addAttributes(Map.of("retrieved.count", result.size()));
                        return result;
                    }
            );

            onEvent.accept(new StreamEvent(
                    "status",
                    new StatusPayload("reranking", "Ranking retrieved chunks")
            ));

            List<SearchHit> reranked = traceHelper.run(
                    "rerank",
                    null,
                    () -> {
                        List<SearchHit> result = reRankService.rerank(question, hits);
                        traceHelper.addAttributes(Map.of(
                                "input.count", hits.size(),
                                "output.count", result.size()
                        ));
                        return result;
                    }
            );

            Double retrievalScore = reranked.isEmpty() ? null : reranked.get(0).score();
            addBestScore(retrievalScore);

            if (reranked.isEmpty()) {
                addFallbackAttributes("no_hits");
                RagResult fallbackResult = fallback(retrievalScore);

                onEvent.accept(new StreamEvent("token", new TokenPayload(fallbackResult.answer())));

                Map<String, Object> payload = new LinkedHashMap<>();
                payload.put("answer", fallbackResult.answer());
                payload.put("citedChunkIds", List.of());
                payload.put("retrievedChunkIds", List.of());
                payload.put("usedChunks", 0);
                payload.put("retrievalScore", retrievalScore);
                payload.put("cards", cards);

                onEvent.accept(new StreamEvent("done", payload));
                return null;
            }

            String topText = reranked.get(0).record().text();

            if (!passesKeywordGuard(question, retrievalScore, topText)) {
                addFallbackAttributes("keyword_guard");
                RagResult fallbackResult = fallback(retrievalScore);

                onEvent.accept(new StreamEvent("token", new TokenPayload(fallbackResult.answer())));

                Map<String, Object> payload = new LinkedHashMap<>();
                payload.put("answer", fallbackResult.answer());
                payload.put("citedChunkIds", List.of());
                payload.put("retrievedChunkIds", List.of());
                payload.put("usedChunks", 0);
                payload.put("retrievalScore", retrievalScore);
                payload.put("cards", cards);

                onEvent.accept(new StreamEvent("done", payload));
                return null;
            }

            List<SearchHit> usableHits = reranked.stream()
                    .limit(3)
                    .toList();

            String citedQuestion = buildDefaultPrompt(question);

            ContextBuildResult contextResult = traceHelper.run(
                    "build-context",
                    null,
                    () -> {
                        List<String> contextChunks = buildCitedContext(usableHits);
                        List<String> retrievedChunkIds = extractChunkIds(usableHits);

                        Map<String, Object> resultAttrs = new LinkedHashMap<>();
                        resultAttrs.put("gen_ai.completion.0.usable.count", usableHits.size());
                        resultAttrs.put("gen_ai.completion.0.context.count", contextChunks.size());
                        traceHelper.addAttributes(resultAttrs);

                        return new ContextBuildResult(contextChunks, retrievedChunkIds);
                    }
            );

            onEvent.accept(new StreamEvent(
                    "status",
                    new StatusPayload("generating", "Generating answer")
            ));

            StringBuilder fullAnswer = new StringBuilder();

            traceHelper.run(
                    "llm-call-stream",
                    buildLlmAttributes(question, citedQuestion, contextResult.contextChunks().size()),
                    () -> {
                        llm.streamAnswer(citedQuestion, contextResult.contextChunks(), token -> {
                            fullAnswer.append(token);
                            onEvent.accept(new StreamEvent("token", new TokenPayload(token)));
                        });
                        return null;
                    }
            );

            String finalAnswer = removeInvalidCitations(fullAnswer.toString(), contextResult.retrievedChunkIds());
            finalAnswer = cleanAnswer(finalAnswer);
            List<String> citedChunkIds = filterValidCitations(finalAnswer, contextResult.retrievedChunkIds());

            Map<String, Object> finalAttrs = new LinkedHashMap<>();
            finalAttrs.put("gen_ai.completion.0.content", finalAnswer);
            finalAttrs.put("gen_ai.completion.0.used.chunks", contextResult.contextChunks().size());
            finalAttrs.put("gen_ai.completion.0.cited.count", citedChunkIds.size());
            traceHelper.addAttributes(finalAttrs);

            Map<String, Object> donePayload = new LinkedHashMap<>();
            donePayload.put("answer", finalAnswer);
            donePayload.put("citedChunkIds", citedChunkIds);
            donePayload.put("retrievedChunkIds", contextResult.retrievedChunkIds());
            donePayload.put("usedChunks", contextResult.contextChunks().size());
            donePayload.put("retrievalScore", retrievalScore);
            donePayload.put("cards", cards);

            onEvent.accept(new StreamEvent("done", donePayload));
            return null;
        });
    }

    private List<VehicleCardDto> buildVehicleCards(String docType, String docId, String question, int topK) {
        if (!shouldBuildVehicleCards(docType, docId)) {
            return List.of();
        }

        try {
            return vehicleSummaryService.searchSummaries(question, topK).stream()
                    .map(summaryMapper::toCard)
                    .toList();
        } catch (Exception e) {
            log.warn("Failed to build vehicle cards for docType='{}', docId='{}': {}", docType, docId, e.getMessage());
            return List.of();
        }
    }

    private boolean shouldBuildVehicleCards(String docType, String docId) {
        if (docType == null || docId == null || docId.isBlank()) {
            log.info("VehicleCards: docType/docId invalid -> FALSE");
            return false;
        }

        String type = docType.trim().toLowerCase(Locale.ROOT);
        String id = docId.trim().toLowerCase(Locale.ROOT);

        if (!type.equals("vehicle")) {
            log.info("VehicleCards: docType='{}' is not vehicle -> FALSE", docType);
            return false;
        }

        boolean result = id.equals("fleet")
                || id.equals("vehicles")
                || id.equals("all-vehicles")
                || id.equals("*");

        log.info("VehicleCards: docType='{}', docId='{}', normalizedDocId='{}', result={}",
                docType, docId, id, result);
        return result;
    }

    private List<ChunkDto> toChunkDtos(List<SearchHit> hits) {
        return hits.stream()
                .map(hit -> new ChunkDto(
                        hit.record().id(),
                        hit.record().text()
                ))
                .toList();
    }

    private void addBestScore(Double bestScore) {
        if (bestScore == null) {
            return;
        }
        traceHelper.addAttributes(Map.of("gen_ai.completion.0.best.score", bestScore));
    }

    private void addFallbackAttributes(String reason) {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("gen_ai.completion.0.fallback", true);
        attrs.put("gen_ai.completion.0.fallback.reason", reason);
        traceHelper.addAttributes(attrs);
    }

    private Map<String, Object> buildLlmAttributes(String question, String prompt, int contextCount) {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("gen_ai.system", "ollama");
        attrs.put("gen_ai.request.model", "llama3.1");
        attrs.put("gen_ai.request.temperature", 0.0);
        attrs.put("gen_ai.prompt.0.role", "user");
        attrs.put("gen_ai.prompt.0.content", question);
        attrs.put("gen_ai.system.prompt.count", contextCount);
        attrs.put("gen_ai.completion.0.length", prompt.length());
        return attrs;
    }

    private String buildStrictPrompt(String question) {
        return """
                You are a strict RAG assistant.
                
                Answer the question using ONLY the provided context.
                
                Rules:
                - Be concise, direct, and clear.
                - Write the answer in natural, professional language.
                - Paraphrase the source instead of copying raw note fragments.
                - Do NOT add explanations, notes, or meta commentary.
                - Do NOT mention "context", "chunks", or these instructions.
                - Do NOT repeat phrases.
                - Do NOT start the answer with a citation.
                - Do NOT copy raw code unless the user explicitly asks for code.
                - Do NOT include the fallback sentence unless the answer is completely unsupported.
                - If the answer is supported, return ONLY the answer.
                
                Citations:
                - Every sentence that contains a fact MUST end with a citation.
                - The answer should be 1 to 3 sentences if sufficient information exists.
                - Do NOT combine multiple sentences under one citation.
                - Each sentence must have its own citation.
                - Use the chunk id exactly as provided.
                - Use EXACT format: [docId:chunkNumber]
                - Do NOT use parentheses.
                - Do NOT say "according to", "based on the context", or "see".
                - Do NOT place citations on a new line.
                - Do NOT place a citation at the beginning of the answer.
                - Do NOT add notes, commentary, or sentences starting with "Note:", "However," or "I must point out".
                - Do NOT say "the information is not available" if ANY related fact is present — use what is available.
                - If the question asks about "engine and speed", answer with engine AND 0-60 AND top speed from the context.
                
                Fallback:
                - If and only if the answer is not supported by the provided context, reply exactly:
                I don't know based on the ingested documents.
                
                Question:
                """ + question;
    }

    private String buildDefaultPrompt(String question) {
        return """
                You are a strict RAG assistant.
                
                Answer the question using ONLY the provided context.
                
                Rules:
                - Be concise, direct, and clear.
                - Write the answer in natural, professional language.
                - Paraphrase the source instead of copying raw note fragments.
                - Do NOT add explanations, notes, or meta commentary.
                - Do NOT mention "context", "chunks", or these instructions.
                - Do NOT repeat phrases.
                - Do NOT include a "Sources" section
                - Do NOT summarize citations at the end
                
                Citations:
                - Every sentence that contains a fact MUST end with a citation.
                - The answer must be 2 to 3 sentences if sufficient information exists.
                - Do NOT combine multiple sentences under one citation.
                - Each sentence must have its own citation.
                - Use the chunk id exactly as provided.
                - Use EXACT format: [docId:chunkNumber]
                - Do NOT use parentheses.
                - Do NOT say "according to", "based on the context", or "see".
                - Do NOT place citations on a new line.
                
                Fallback:
                - If the answer is not supported by the provided context, reply exactly:
                I don't know based on the ingested documents.
                
                Question:
                """ + question;
    }

    private String cleanAnswer(String answer) {
        if (answer == null || answer.isBlank()) {
            return answer;
        }

        String cleaned = answer
                .replaceFirst("(?s)^\\s*\\[[a-zA-Z0-9\\-]+:\\d+]\\s*\\n?", "")
                .replaceAll("\\n{2,}", "\n")
                .trim();

        String fallback = "I don't know based on the ingested documents.";
        if (!cleaned.equals(fallback) && cleaned.contains(fallback)) {
            cleaned = cleaned.replace(fallback, "").trim();
        }

        return cleaned;
    }

    private boolean passesKeywordGuard(String question, Double bestScore, String topText) {
        if (bestScore == null) {
            return false;
        }

        boolean isRrfScore = bestScore < 0.5;

        double low = isRrfScore ? RRF_LOW : COSINE_LOW;
        double high = isRrfScore ? RRF_HIGH : COSINE_HIGH;

        if (bestScore < low) return false;
        if (bestScore >= high) return true;
        return hasKeywordOverlap(question, topText);
    }

    private List<String> extractChunkIds(List<SearchHit> hits) {
        return hits.stream()
                .map(hit -> hit.record().id())
                .toList();
    }

    private RagResult fallback(Double retrievalScore) {
        return new RagResult(
                FALLBACK,
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                0,
                retrievalScore
        );
    }

    private boolean hasKeywordOverlap(String question, String chunkText) {
        String q = question.toLowerCase().replaceAll("[^a-z0-9 ]", " ");
        String c = chunkText.toLowerCase().replaceAll("[^a-z0-9 ]", " ");

        String[] qTokens = q.split("\\s+");
        String[] cTokens = c.split("\\s+");

        Set<String> chunkWords = new HashSet<>(List.of(cTokens));

        int hits = 0;
        int total = 0;

        for (String t : qTokens) {
            if (t.length() < 3) continue;
            total++;
            if (chunkWords.contains(t)) hits++;
        }

        return total > 0 && hits >= 1;
    }

    private List<String> buildCitedContext(List<SearchHit> hits) {
        return hits.stream()
                .map(hit -> {
                    String id = hit.record().id();
                    String text = hit.record().text();

                    String cleaned = compress(text);

                    return "[" + id + "] " + cleaned;
                })
                .toList();
    }

    private String compress(String text) {
        if (text == null || text.isBlank()) return text;

        if (isVehicleChunk(text)) {
            return text;
        }

        List<String> lines = Arrays.stream(text.split("\n"))
                .map(String::trim)
                .filter(line -> !line.isEmpty())
                .filter(line ->
                        !line.startsWith("SELECT") &&
                                !line.contains("===") &&
                                !line.contains(" ns ") &&
                                !line.contains(" ms ")
                )
                .toList();

        for (int i = 0; i < lines.size(); i++) {
            if (lines.get(i).toLowerCase().contains("interview answer")) {
                return lines.stream()
                        .skip(i)
                        .limit(3)
                        .collect(Collectors.joining(" "));
            }
        }

        Optional<String> definition = lines.stream()
                .filter(line -> {
                    String l = line.toLowerCase();
                    return l.contains(" is a ") || l.contains(" is an ");
                })
                .findFirst();

        if (definition.isPresent()) {
            return definition.get();
        }

        return lines.stream()
                .filter(line -> line.length() < 400)
                .limit(10)
                .collect(Collectors.joining(" "));
    }

    private boolean isVehicleChunk(String text) {
        if (text == null || text.isBlank()) return false;
        String first = text.strip().lines().findFirst().orElse("").toLowerCase();

        if (first.contains("vehicle specification")) return true;

        if (first.matches(".*\\b(performance specs|ownership cost|5-year ownership"
                + "|rankings and awards|safety ratings|features and trim"
                + "|expert reviews|maintenance at \\d|recall [a-z]"
                + "|industry rankings).*")) return true;

        if (first.startsWith("motortrend ")) return true;
        if (first.matches(".*\\b(article:|section:|motortrend ratings|expert assessment"
                + "|vehicles featured|article excerpt).*")) return true;

        return false;
    }

    private List<String> filterValidCitations(String answer, List<String> validIds) {
        if (answer == null || answer.isBlank()) {
            return List.of();
        }

        Set<String> valid = new HashSet<>(validIds);

        return java.util.regex.Pattern.compile("\\[([a-zA-Z0-9\\-]+:\\d+)]")
                .matcher(answer)
                .results()
                .map(m -> m.group(1))
                .filter(valid::contains)
                .distinct()
                .toList();
    }

    private String removeInvalidCitations(String answer, List<String> validIds) {
        if (answer == null || answer.isBlank()) {
            return answer;
        }

        Set<String> valid = new HashSet<>(validIds);

        return java.util.regex.Pattern.compile("\\[([a-zA-Z0-9\\-]+:\\d+)]")
                .matcher(answer)
                .replaceAll(matchResult -> {
                    String id = matchResult.group(1);
                    return valid.contains(id) ? "[" + id + "]" : "";
                });
    }

    private record ContextBuildResult(
            List<String> contextChunks,
            List<String> retrievedChunkIds
    ) {
    }

    public record RagResult(
            String answer,
            List<VehicleCardDto> cards,
            List<ChunkDto> chunks,
            List<String> citedChunkIds,
            List<String> retrievedChunkIds,
            int usedChunks,
            Double retrievalScore
    ) {
    }

    public record StreamEvent(
            String type,
            Object payload
    ) {
    }

    public record StatusPayload(
            String stage,
            String message
    ) {
    }

    public record TokenPayload(
            String value
    ) {
    }
}