package com.example.airagassistant.rag;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Scope;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

@Service
public class ReRankService {

    private final Tracer tracer = GlobalOpenTelemetry.getTracer("ai-rag");

    public List<SearchHit> rerank(String question, List<SearchHit> hits) {
        Span span = tracer.spanBuilder("rerank-score").startSpan();

        try (Scope scope = span.makeCurrent()) {
            span.setAttribute("langsmith.span.kind", "chain");
            span.setAttribute("gen_ai.prompt.0.role", "user");
            span.setAttribute("gen_ai.prompt.0.content", question);
            span.setAttribute("langsmith.metadata.input_count", hits.size());

            String normalizedQuestion = normalize(question);
            Set<String> qTokens = tokenize(normalizedQuestion);

            List<SearchHit> reranked = hits.stream()
                    .map(hit -> {
                        String content = normalize(hit.record().text());
                        Set<String> chunkTokens = tokenize(content);

                        double baseScore = hit.score() * 0.85;

                        double exactPhraseBoost = content.contains(normalizedQuestion) ? 0.15 : 0.0;

                        long overlapCount = qTokens.stream()
                                .filter(chunkTokens::contains)
                                .count();
                        double overlapBoost = qTokens.isEmpty()
                                ? 0.0
                                : ((double) overlapCount / qTokens.size()) * 0.12;

                        double importantTermBoost = 0.0;
                        for (String token : qTokens) {
                            if (token.startsWith("@") || token.length() > 8) {
                                if (content.contains(token)) {
                                    importantTermBoost += 0.03;
                                }
                            }
                        }

                        double finalScore = baseScore + exactPhraseBoost + overlapBoost + importantTermBoost;

                        return new SearchHit(hit.record(), finalScore);
                    })
                    .sorted(Comparator.comparingDouble(SearchHit::score).reversed())
                    .collect(Collectors.toList());

            span.setAttribute("langsmith.metadata.output_count", reranked.size());
            if (!reranked.isEmpty()) {
                span.setAttribute("langsmith.metadata.top_chunk_id", reranked.get(0).record().id());
                span.setAttribute("langsmith.metadata.top_score", reranked.get(0).score());
            }

            return reranked;
        } catch (Exception e) {
            span.recordException(e);
            throw e;
        } finally {
            span.end();
        }
    }

    private String normalize(String text) {
        if (text == null) {
            return "";
        }
        return text.toLowerCase()
                .replaceAll("[^a-z0-9@ ]", " ")
                .replaceAll("\\s+", " ")
                .trim();
    }

    private Set<String> tokenize(String text) {
        if (text == null || text.isBlank()) {
            return Set.of();
        }

        return Arrays.stream(text.split("\\s+"))
                .filter(t -> t.length() > 2 || t.startsWith("@"))
                .collect(Collectors.toSet());
    }
}