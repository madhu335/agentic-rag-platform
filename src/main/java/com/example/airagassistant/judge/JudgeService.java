package com.example.airagassistant.judge;

import com.example.airagassistant.trace.TraceHelper;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
public class JudgeService {

    private final JudgeClient judgeClient;
    private final ObjectMapper objectMapper;
    private final TraceHelper traceHelper;

    public JudgeResult evaluate(String question, String answer, List<String> contextChunks) {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.span.kind", "chain");
        attrs.put("langsmith.metadata.context_chunk_count", contextChunks != null ? contextChunks.size() : 0);
        attrs.put("langsmith.metadata.answer_length", answer != null ? answer.length() : 0);

        return traceHelper.run("judge-evaluation", attrs, () -> {
            String prompt = buildPrompt(question, answer);

            Exception lastException = null;

            for (int i = 0; i < 2; i++) {
                try {
                    traceHelper.addAttributes(Map.of(
                            "langsmith.metadata.judge_attempt", i + 1
                    ));

                    String raw = judgeClient.evaluate(prompt, contextChunks);
                    String json = extractJsonObject(raw);

                    JudgeResult result = objectMapper.readValue(json, JudgeResult.class);

                    double score = result.score();
                    if (score > 1.0) {
                        score = score / 10.0;
                    }

                    JudgeResult normalized = new JudgeResult(
                            result.grounded(),
                            result.correct(),
                            result.complete(),
                            score,
                            result.reason()
                    );

                    traceHelper.addAttributes(Map.of(
                            "gen_ai.completion.0.content", result.reason(),
                            "gen_ai.completion.0.score", result.score()
                    ));
                    traceHelper.addAttributes(buildJudgeResultAttributes(normalized, false));
                    return normalized;

                } catch (Exception e) {
                    lastException = e;
                    log.warn("Judge attempt {} failed: {}", i + 1, e.getMessage());

                    Map<String, Object> errorAttrs = new LinkedHashMap<>();
                    errorAttrs.put("langsmith.metadata.judge_attempt", i + 1);
                    errorAttrs.put("langsmith.metadata.judge_attempt_failed", true);
                    errorAttrs.put("langsmith.metadata.judge_error", e.getMessage());
                    traceHelper.addAttributes(errorAttrs);
                }
            }

            log.warn("Judge failed after retries", lastException);

            JudgeResult fallback = new JudgeResult(false, false, false, 0.0, "judge_unavailable");
            traceHelper.addAttributes(buildJudgeResultAttributes(fallback, true));
            return fallback;
        });
    }

    private String extractJsonObject(String text) {
        if (text == null || text.isBlank()) {
            return "{}";
        }

        String trimmed = text.trim();

        if (trimmed.startsWith("```")) {
            trimmed = trimmed
                    .replaceFirst("^```json\\s*", "")
                    .replaceFirst("^```\\s*", "")
                    .replaceFirst("\\s*```$", "")
                    .trim();
        }

        int firstBrace = trimmed.indexOf('{');
        int lastBrace = trimmed.lastIndexOf('}');

        if (firstBrace >= 0 && lastBrace > firstBrace) {
            return trimmed.substring(firstBrace, lastBrace + 1);
        }

        return trimmed;
    }

    private String buildPrompt(String question, String answer) {
        return """
                You are an evaluator.

                Evaluate the answer using only the provided context.

                Question:
                %s

                Answer:
                %s

                Return JSON only in this exact format:
                {
                  "grounded": true,
                  "correct": true,
                  "complete": true,
                  "score": 0.7,
                  "reason": "short explanation"
                }

                Rules:
                - score must be a decimal between 0.0 and 1.0
                - do NOT use a 1-10 scale
                - do NOT use percentages
                - do NOT return markdown
                - do NOT return any text outside the JSON
                """.formatted(question, answer);
    }

    private Map<String, Object> buildJudgeResultAttributes(JudgeResult result, boolean unavailable) {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.metadata.judge.grounded", result.grounded());
        attrs.put("langsmith.metadata.judge.correct", result.correct());
        attrs.put("langsmith.metadata.judge.complete", result.complete());
        attrs.put("langsmith.metadata.judge.score", result.score());
        attrs.put("langsmith.metadata.judge.reason", result.reason());
        attrs.put("langsmith.metadata.judge.unavailable", unavailable);
        return attrs;
    }
}