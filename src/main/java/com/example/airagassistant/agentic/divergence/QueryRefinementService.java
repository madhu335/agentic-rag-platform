package com.example.airagassistant.agentic.divergence;

import com.example.airagassistant.LlmClient;
import com.example.airagassistant.judge.JudgeResult;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class QueryRefinementService {

    private final LlmClient llmClient;

    public String refineQuery(String originalQuery,
                              String previousAnswer,
                              List<String> retrievedChunks,
                              JudgeResult judge) {

        String prompt = buildPrompt(originalQuery, previousAnswer, retrievedChunks, judge);

        String rewritten = llmClient.answer(prompt, List.of());

        if (rewritten == null || rewritten.isBlank()) {
            return fallbackRewrite(originalQuery, judge);
        }

        return cleanSingleLine(rewritten);
    }

    private String buildPrompt(String originalQuery,
                               String previousAnswer,
                               List<String> retrievedChunks,
                               JudgeResult judge) {

        String judgeReason = judge != null ? nullSafe(judge.reason()) : "";
        String grounded = judge != null ? String.valueOf(judge.grounded()) : "";
        String correct = judge != null ? String.valueOf(judge.correct()) : "";
        String complete = judge != null ? String.valueOf(judge.complete()) : "";

        String chunkSummary = summarizeChunks(retrievedChunks);

        return """
                You are improving a retrieval query for a second-pass RAG search.

                Your task:
                Rewrite the query so retrieval becomes more specific, grounded, and complete.

                Rules:
                - Return ONLY the improved query.
                - Do not explain anything.
                - Keep it concise.
                - Make it more specific than the original.
                - Use retrieved chunk content as hints for missing concepts or entities.
                - Use judge feedback to correct vagueness, incompleteness, or weak grounding.
                - Do not copy full chunk text into the query.
                - Prefer retrieval-friendly phrasing.

                Original query:
                %s

                Previous answer:
                %s

                Retrieved chunk hints:
                %s

                Judge grounded:
                %s

                Judge correct:
                %s

                Judge complete:
                %s

                Judge feedback:
                %s
                """.formatted(
                nullSafe(originalQuery),
                nullSafe(previousAnswer),
                nullSafe(chunkSummary),
                grounded,
                correct,
                complete,
                judgeReason
        );
    }

    private String summarizeChunks(List<String> chunks) {
        if (chunks == null || chunks.isEmpty()) {
            return "";
        }

        return chunks.stream()
                .limit(3)
                .map(this::compress)
                .collect(Collectors.joining("\n- ", "- ", ""));
    }

    private String compress(String text) {
        if (text == null) {
            return "";
        }

        String cleaned = text.replaceAll("\\s+", " ").trim();
        return cleaned.length() <= 220 ? cleaned : cleaned.substring(0, 220) + "...";
    }

    private String fallbackRewrite(String originalQuery, JudgeResult judge) {
        StringBuilder sb = new StringBuilder(originalQuery);

        if (judge != null) {
            if (!judge.complete()) {
                sb.append(" including missing details");
            }
            if (!judge.grounded()) {
                sb.append(" with explicit supporting facts");
            }
            if (!judge.correct()) {
                sb.append(" with verified accurate information");
            }
        } else {
            sb.append(" with more specific supporting details");
        }

        return sb.toString().trim();
    }

    private String cleanSingleLine(String text) {
        String cleaned = text.replaceAll("[\\r\\n]+", " ").trim();
        cleaned = cleaned.replaceAll("^\"|\"$", "").trim();
        cleaned = cleaned.replaceAll("^'+|'+$", "").trim();
        return cleaned;
    }

    private String nullSafe(String value) {
        return value == null ? "" : value;
    }
}