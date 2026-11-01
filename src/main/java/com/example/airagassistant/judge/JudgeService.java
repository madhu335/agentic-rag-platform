package com.example.airagassistant.judge;

import com.example.airagassistant.trace.TraceHelper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Judges a RAG answer against its context and returns one {@link JudgeVerdict}.
 *
 * <h3>Contract (judge_compare.py, 2026-11-01)</h3>
 * One verdict field with defined categories, placeholders instead of example values. The old prompt's example
 * JSON (true/true/true/0.7) was copied verbatim by a non-reasoning judge, and its "grounded" boolean was
 * ambiguous. With this prompt: Nemotron thinking-on and gpt-oss-120b both 24/24 decisions, 0 conflicts.
 *
 * <h3>Cascade (only when the client has a fast mode, i.e. judge.provider=triton)</h3>
 * <pre>
 *   fast judge (reasoning off, ~0.2 s)  +  NumbersCheck (no LLM)
 *     fast REJECT                          -> return it              (stage fast_reject; 0 false rejects in tests)
 *     fast PASS and every number in context -> return it             (stage fast_pass)
 *     fast PASS but a number is not in context,
 *     fast output unusable, or verdict/score conflict
 *                                          -> thorough judge (reasoning on, ~1.2 s)  (stage escalated:&lt;why&gt;)
 * </pre>
 * The fast judge misses answers that look right (an extra invented claim, a wrong sum); the numbers check misses
 * numbers attached to the wrong quantity. Together they caught every bad case in the test set.
 * Providers without a fast mode (halo, claude, openai, default) always use the thorough judge.
 *
 * <h3>Consistency guard</h3>
 * PASS with score &lt; 0.5, or UNSUPPORTED/INCORRECT with score &gt;= 0.8, is treated as unreliable: the fast stage
 * escalates, the thorough stage retries once and then trusts the verdict.
 */
@Service
@Slf4j
public class JudgeService {

    private static final double CONFLICT_PASS_BELOW = 0.5;
    private static final double CONFLICT_FAIL_AT_OR_ABOVE = 0.8;

    private final JudgeClient judgeClient;
    private final ObjectMapper objectMapper;
    private final TraceHelper traceHelper;
    private final boolean cascadeEnabled;
    private final boolean verifyFastRejects;
    private final int thoroughAttempts;

    public JudgeService(
            JudgeClient judgeClient,
            ObjectMapper objectMapper,
            TraceHelper traceHelper,
            @Value("${judge.cascade.enabled:true}") boolean cascadeEnabled,
            @Value("${judge.cascade.verify-fast-rejects:false}") boolean verifyFastRejects,
            @Value("${judge.thorough-attempts:2}") int thoroughAttempts
    ) {
        this.judgeClient = judgeClient;
        this.objectMapper = objectMapper;
        this.traceHelper = traceHelper;
        this.cascadeEnabled = cascadeEnabled;
        this.verifyFastRejects = verifyFastRejects;
        this.thoroughAttempts = Math.max(1, thoroughAttempts);
        log.info("JudgeService: cascadeEnabled={} (active only for clients with a fast mode) verifyFastRejects={} "
                + "thoroughAttempts={}", cascadeEnabled, verifyFastRejects, this.thoroughAttempts);
    }

    public JudgeResult evaluate(String question, String answer, List<String> contextChunks) {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.span.kind", "chain");
        attrs.put("langsmith.metadata.context_chunk_count", contextChunks != null ? contextChunks.size() : 0);
        attrs.put("langsmith.metadata.answer_length", answer != null ? answer.length() : 0);

        return traceHelper.run("judge-evaluation", attrs, () -> {
            String prompt = buildPrompt(question, answer);
            JudgeResult result = (cascadeEnabled && judgeClient.supportsFastMode())
                    ? cascade(prompt, answer, contextChunks)
                    : thorough(prompt, contextChunks, "thorough");

            traceHelper.addAttributes(buildJudgeResultAttributes(result));
            log.info("Judge: verdict={} score={} stage={} reason={}",
                    result.verdict(), result.score(), result.stage(), result.reason());
            return result;
        });
    }

    // ── cascade ──────────────────────────────────────────────────────────────────────────────────────

    private JudgeResult cascade(String prompt, String answer, List<String> contextChunks) {
        JudgeResult fast = attempt(() -> judgeClient.evaluateFast(prompt, contextChunks), "fast");
        List<String> missingNumbers = NumbersCheck.missingNumbers(answer, contextChunks);

        String escalation;
        if (fast == null) {
            escalation = "fast_unusable";
        } else if (isConflict(fast)) {
            escalation = "fast_conflict";
        } else if (!fast.accepted()) {
            if (!verifyFastRejects) {
                return fast.withStage("fast_reject");
            }
            escalation = "verify_fast_reject";
        } else if (!missingNumbers.isEmpty()) {
            escalation = "numbers_not_in_context:" + String.join(",", missingNumbers);
        } else {
            return fast.withStage("fast_pass");
        }

        traceHelper.addAttributes(Map.of(
                "langsmith.metadata.judge.escalated", true,
                "langsmith.metadata.judge.escalation_reason", escalation
        ));
        log.info("Judge escalating to thorough: {} (fast verdict={})",
                escalation, fast == null ? "none" : fast.verdict());
        return thorough(prompt, contextChunks, "escalated:" + escalation);
    }

    private JudgeResult thorough(String prompt, List<String> contextChunks, String stage) {
        JudgeResult last = null;
        for (int i = 1; i <= thoroughAttempts; i++) {
            traceHelper.addAttributes(Map.of("langsmith.metadata.judge_attempt", i));
            JudgeResult result = attempt(() -> judgeClient.evaluate(prompt, contextChunks), "thorough#" + i);
            if (result == null) {
                continue;
            }
            last = result;
            if (!isConflict(result)) {
                return result.withStage(stage);
            }
            log.warn("Judge verdict/score conflict on attempt {}: verdict={} score={}", i, result.verdict(), result.score());
        }
        if (last != null) {
            // Still conflicting after retries: the verdict is the field validated as reliable, keep it.
            return last.withStage(stage + ":conflict_kept_verdict");
        }
        log.warn("Judge unavailable after {} attempt(s)", thoroughAttempts);
        traceHelper.addAttributes(Map.of("langsmith.metadata.judge.unavailable", true));
        return JudgeResult.unavailable();
    }

    /** One client call + parse. Returns null (and logs) on any failure. */
    private JudgeResult attempt(Supplier<String> call, String label) {
        try {
            String raw = call.get();
            JudgeResult parsed = parse(raw);
            if (parsed == null) {
                log.warn("Judge {} output unusable: {}", label, truncate(raw, 200));
            }
            return parsed;
        } catch (Exception e) {
            log.warn("Judge {} failed: {}", label, e.getMessage());
            traceHelper.addAttributes(Map.of(
                    "langsmith.metadata.judge_attempt_failed", true,
                    "langsmith.metadata.judge_error", String.valueOf(e.getMessage())
            ));
            return null;
        }
    }

    static boolean isConflict(JudgeResult r) {
        return switch (r.verdict()) {
            case PASS -> r.score() < CONFLICT_PASS_BELOW;
            case UNSUPPORTED, INCORRECT -> r.score() >= CONFLICT_FAIL_AT_OR_ABOVE;
            default -> false;   // INCOMPLETE can legitimately score high; UNAVAILABLE never reaches here
        };
    }

    // ── parsing ──────────────────────────────────────────────────────────────────────────────────────

    /**
     * Parses {"verdict","score","reason"}. Falls back to the old {"grounded","correct","complete",...} shape so a
     * model that ignores the new format still yields a usable result. Returns null when nothing usable is found.
     */
    JudgeResult parse(String raw) {
        String json = extractJsonObject(raw);
        JsonNode node;
        try {
            node = objectMapper.readTree(json);
        } catch (Exception e) {
            return null;
        }
        if (node == null || !node.isObject()) {
            return null;
        }

        double score = normalizeScore(node.path("score").asDouble(Double.NaN));
        String reason = node.path("reason").asText("");

        JudgeVerdict verdict = JudgeVerdict.parse(node.path("verdict").asText(null));
        if (verdict != null) {
            if (Double.isNaN(score)) {
                score = verdict.accepted() ? 1.0 : 0.0;
            }
            return new JudgeResult(verdict, score, reason, "parsed");
        }

        if (node.has("grounded") && node.has("correct")) {
            JudgeResult legacy = new JudgeResult(
                    node.path("grounded").asBoolean(false),
                    node.path("correct").asBoolean(false),
                    node.path("complete").asBoolean(true),
                    Double.isNaN(score) ? 0.0 : score,
                    reason);
            return legacy.withStage("parsed_legacy");
        }
        return null;
    }

    private static double normalizeScore(double score) {
        if (Double.isNaN(score)) {
            return score;
        }
        if (score > 1.0) {
            score = score / 10.0;          // a 1-10 scale slipped through
        }
        return Math.max(0.0, Math.min(1.0, score));
    }

    static String extractJsonObject(String text) {
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

    // ── prompt ───────────────────────────────────────────────────────────────────────────────────────

    /** Validated verbatim in judge_compare.py --prompt verdict. Keep the two in sync. */
    static String buildPrompt(String question, String answer) {
        return """
                You are an evaluator.

                Evaluate the answer using only the provided context.

                Question:
                %s

                Answer:
                %s

                Choose exactly one verdict:
                - UNSUPPORTED: the answer states a fact, number or detail the context does not state,
                  or attaches a number from the context to the wrong quantity.
                - INCORRECT: the facts used are in the context, but the answer derives something wrongly
                  (a wrong total, comparison or conclusion).
                - INCOMPLETE: everything stated is supported and correct, but the answer leaves out part of
                  what the question asked although the context contains it.
                - PASS: every claim is supported and correct, and the question is answered. An answer that
                  honestly says the context does not contain the information is PASS when that is true.

                Check each claim in the answer against the context once, then decide.

                Return JSON only, in this format:
                {"verdict": "<PASS|UNSUPPORTED|INCORRECT|INCOMPLETE>", "score": <0.0 to 1.0, where 1.0 is fully correct and supported>, "reason": "<at most 12 words; name the problem claim if any>"}

                Rules:
                - do NOT return markdown
                - do NOT return any text outside the JSON
                """.formatted(question, answer);
    }

    // ── tracing ──────────────────────────────────────────────────────────────────────────────────────

    private Map<String, Object> buildJudgeResultAttributes(JudgeResult result) {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.metadata.judge.verdict", result.verdict().name());
        attrs.put("langsmith.metadata.judge.stage", result.stage());
        attrs.put("langsmith.metadata.judge.grounded", result.grounded());
        attrs.put("langsmith.metadata.judge.correct", result.correct());
        attrs.put("langsmith.metadata.judge.complete", result.complete());
        attrs.put("langsmith.metadata.judge.score", result.score());
        attrs.put("langsmith.metadata.judge.reason", result.reason());
        attrs.put("langsmith.metadata.judge.unavailable", result.isUnavailable());
        attrs.put("gen_ai.completion.0.content", result.reason());
        attrs.put("gen_ai.completion.0.score", result.score());
        return attrs;
    }

    private static String truncate(String text, int max) {
        if (text == null) {
            return "";
        }
        return text.length() <= max ? text : text.substring(0, max) + "...";
    }
}
