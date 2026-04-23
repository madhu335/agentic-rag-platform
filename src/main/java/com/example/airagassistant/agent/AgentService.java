package com.example.airagassistant.agent;

import com.example.airagassistant.LlmClient;
import com.example.airagassistant.agentic.dto.ChunkDto;
import com.example.airagassistant.agentic.mapper.VehicleSummaryMapper;
import com.example.airagassistant.rag.RagAnswerService;
import com.example.airagassistant.rag.RagRetriever;
import com.example.airagassistant.rag.SearchHit;
import com.example.airagassistant.rag.VehicleCardDto;
import com.example.airagassistant.rag.retrieval.VehicleSummaryService;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Scope;
import org.springframework.stereotype.Service;

import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

@Service
public class AgentService {

    private static final String FALLBACK = "I don't know based on the ingested documents.";

    private final RagRetriever ragRetriever;
    private final LlmClient llm;
    private final VehicleSummaryService vehicleSummaryService;
    private final VehicleSummaryMapper summaryMapper;
    private final Tracer tracer = GlobalOpenTelemetry.getTracer("ai-rag");

    public AgentService(RagRetriever ragRetriever,
                        LlmClient llm,
                        VehicleSummaryService vehicleSummaryService,
                        VehicleSummaryMapper summaryMapper) {
        this.ragRetriever = ragRetriever;
        this.llm = llm;
        this.vehicleSummaryService = vehicleSummaryService;
        this.summaryMapper = summaryMapper;
    }

    public RagAnswerService.RagResult answer(String docType, String docId, String question, int topK) {
        Span agentSpan = tracer.spanBuilder("agent-flow").startSpan();

        try (Scope scope = agentSpan.makeCurrent()) {
            agentSpan.setAttribute("langsmith.span.kind", "chain");
            agentSpan.setAttribute("gen_ai.prompt.0.role", "user");
            agentSpan.setAttribute("gen_ai.prompt.0.content", question);
            agentSpan.setAttribute("langsmith.metadata.doc_type", docType);
            agentSpan.setAttribute("langsmith.metadata.doc_id", docId);
            agentSpan.setAttribute("langsmith.metadata.top_k", topK);

            List<VehicleCardDto> cards = buildVehicleCards(docType, docId, question, topK);

            List<SearchHit> hits = ragRetriever.retrieve(docId, question, Math.max(topK, 3));
            agentSpan.setAttribute("langsmith.metadata.hit_count", hits.size());

            Double retrievalScore = hits.isEmpty() ? null : hits.get(0).score();
            if (retrievalScore != null) {
                agentSpan.setAttribute("langsmith.metadata.best_score", retrievalScore);
            }

            if (hits.isEmpty()) {
                agentSpan.setAttribute("langsmith.metadata.fallback", true);
                agentSpan.setAttribute("langsmith.metadata.fallback_reason", "no_hits");

                agentSpan.setAttribute("gen_ai.completion.0.role", "assistant");
                agentSpan.setAttribute("gen_ai.completion.0.content", FALLBACK);

                return new RagAnswerService.RagResult(
                        FALLBACK,
                        cards,
                        List.of(),
                        List.of(),
                        List.of(),
                        0,
                        retrievalScore
                );
            }

            List<SearchHit> usableHits = hits.stream()
                    .limit(3)
                    .toList();

            boolean hasUsableContext = !usableHits.isEmpty()
                    && usableHits.get(0).record().text() != null
                    && !usableHits.get(0).record().text().isBlank();

            if (!hasUsableContext) {
                agentSpan.setAttribute("langsmith.metadata.fallback", true);
                agentSpan.setAttribute("langsmith.metadata.fallback_reason", "empty_context");

                agentSpan.setAttribute("gen_ai.completion.0.role", "assistant");
                agentSpan.setAttribute("gen_ai.completion.0.content", FALLBACK);

                return new RagAnswerService.RagResult(
                        FALLBACK,
                        cards,
                        List.of(),
                        List.of(),
                        List.of(),
                        0,
                        retrievalScore
                );
            }

            agentSpan.setAttribute("langsmith.metadata.used_chunks", usableHits.size());

            List<String> contextChunks;
            Span contextSpan = tracer.spanBuilder("build-context").startSpan();
            try (Scope ctxScope = contextSpan.makeCurrent()) {
                contextSpan.setAttribute("langsmith.span.kind", "chain");
                contextChunks = buildCitedContext(usableHits);
                contextSpan.setAttribute("langsmith.metadata.context_size", contextChunks.size());
            } catch (Exception e) {
                contextSpan.recordException(e);
                throw e;
            } finally {
                contextSpan.end();
            }

            List<String> retrievedChunkIds = usableHits.stream()
                    .map(h -> h.record().id())
                    .collect(Collectors.toList());

            List<ChunkDto> chunks = usableHits.stream()
                    .map(h -> new ChunkDto(
                            h.record().id(),
                            h.record().text()
                    ))
                    .toList();

            String agentQuestion = buildGroundedQuestion(question);

            String answer;
            Span llmSpan = tracer.spanBuilder("llm-call").startSpan();
            try (Scope llmScope = llmSpan.makeCurrent()) {
                llmSpan.setAttribute("langsmith.span.kind", "llm");
                llmSpan.setAttribute("langsmith.metadata.provider", "vllm");
                llmSpan.setAttribute("gen_ai.prompt.0.role", "user");
                llmSpan.setAttribute("gen_ai.prompt.0.content", agentQuestion);
                llmSpan.setAttribute("langsmith.metadata.context_size", contextChunks.size());

                answer = llm.answer(agentQuestion, contextChunks);

                llmSpan.setAttribute("gen_ai.completion.0.role", "assistant");
                llmSpan.setAttribute("gen_ai.completion.0.content", answer);
                llmSpan.setAttribute("langsmith.metadata.answer_length", answer.length());
            } catch (Exception e) {
                llmSpan.recordException(e);
                throw e;
            } finally {
                llmSpan.end();
            }

            answer = removeInvalidCitations(answer, retrievedChunkIds);

            List<String> citedChunkIds = filterValidCitations(answer, retrievedChunkIds);

            if (citedChunkIds.isEmpty()) {
                agentSpan.setAttribute("langsmith.metadata.fallback", true);
                agentSpan.setAttribute("langsmith.metadata.fallback_reason", "no_valid_citations");

                agentSpan.setAttribute("gen_ai.completion.0.role", "assistant");
                agentSpan.setAttribute("gen_ai.completion.0.content", FALLBACK);

                return new RagAnswerService.RagResult(
                        FALLBACK,
                        cards,
                        chunks,
                        List.of(),
                        retrievedChunkIds,
                        contextChunks.size(),
                        retrievalScore
                );
            }

            String answerWithCitations = answer;

            if (!citedChunkIds.isEmpty()) {
                String citations = citedChunkIds.stream()
                        .map(id -> "[" + id + "]")
                        .reduce((a, b) -> a + " " + b)
                        .orElse("");

                answerWithCitations = answer + "\n\nSources: " + citations;
            }

            agentSpan.setAttribute("gen_ai.completion.0.role", "assistant");
            agentSpan.setAttribute("gen_ai.completion.0.content", answerWithCitations);
            agentSpan.setAttribute("langsmith.metadata.cited_count", citedChunkIds.size());
            agentSpan.setAttribute("langsmith.metadata.answer_length", answerWithCitations.length());

            return new RagAnswerService.RagResult(
                    answerWithCitations,
                    cards,
                    chunks,
                    citedChunkIds,
                    retrievedChunkIds,
                    contextChunks.size(),
                    retrievalScore
            );

        } catch (Exception e) {
            agentSpan.recordException(e);
            throw e;
        } finally {
            agentSpan.end();
        }
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
            return List.of();
        }
    }

    private boolean shouldBuildVehicleCards(String docType, String docId) {
        if (docType == null || docId == null || docId.isBlank()) {
            return false;
        }

        String type = docType.trim().toLowerCase(Locale.ROOT);
        String id = docId.trim().toLowerCase(Locale.ROOT);

        if (!type.equals("vehicle")) {
            return false;
        }

        return id.equals("fleet")
                || id.equals("vehicles")
                || id.equals("all-vehicles")
                || id.equals("*");
    }

    private List<String> buildCitedContext(List<SearchHit> hits) {
        return hits.stream()
                .map(hit -> "Chunk ID: [" + hit.record().id() + "]\n" + hit.record().text())
                .toList();
    }

    private List<String> filterValidCitations(String answer, List<String> validIds) {
        if (answer == null || answer.isBlank()) {
            return List.of();
        }

        Set<String> valid = new HashSet<>(validIds);

        return Pattern.compile("\\[([a-zA-Z0-9\\-]+:\\d+)]")
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

        return Pattern.compile("\\[([a-zA-Z0-9\\-]+:\\d+)]")
                .matcher(answer)
                .replaceAll(matchResult -> {
                    String id = matchResult.group(1);
                    return valid.contains(id) ? "[" + id + "]" : "";
                });
    }

    private String buildGroundedQuestion(String question) {
        return """
                You are a strict retrieval-based assistant.

                You are given context chunks. Each chunk has an ID like:
                [spring-boot-qa:6]

                RULES:
                - You MUST use these chunk IDs in your answer
                - EVERY sentence MUST include at least one chunk citation
                - DO NOT answer without citations
                - DO NOT invent citations
                - ONLY use IDs that appear in the context

                If you cannot cite, respond EXACTLY:
                "I don't know based on the ingested documents."

                Example:
                Lazy loading delays fetching until accessed [spring-boot-qa:6].

                Answer concisely.

                Question:
                """ + question;
    }
}