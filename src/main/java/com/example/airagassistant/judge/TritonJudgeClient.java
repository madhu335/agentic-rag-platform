package com.example.airagassistant.judge;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Judge on Nemotron-3-Nano-30B-A3B served by Triton (vLLM backend) on the 5090, native generate API.
 *
 * Two modes, used by the JudgeService cascade:
 *   fast      reasoning OFF (empty think block)  ~0.2 s, max fast-max-tokens
 *   thorough  reasoning ON                       ~1.2 s, max thorough-max-tokens
 * Validated with scripts/judge-compare/judge_compare.py (2026-11-01, verdict prompt):
 *   thorough 24/24 decisions, 0 conflicts; fast 6/8 (misses were "looks right" answers the numbers check flags).
 *
 * Reasoning and the JSON share max_tokens. With reasoning on, 1024 tokens cut the reasoning off on hard cases
 * (no closing think tag, no JSON); 3072 never did. A cut-off is reported as an exception so JudgeService retries.
 *
 * Triton's generate endpoint does not apply the chat template, so the Nemotron ChatML template is rendered here
 * (same format as TritonVllmClient). The system prompt is identical to the one used in judge_compare.py.
 */
@Slf4j
@Component
@ConditionalOnProperty(name = "judge.provider", havingValue = "triton")
public class TritonJudgeClient implements JudgeClient {

    private static final String IM_START = "<|im_start|>";
    private static final String IM_END = "<|im_end|>";
    private static final String THINK_END = "</think>";
    /** Same system prompt as judge_compare.py / HaloJudgeClient ("Reasoning:" is a no-op for Nemotron). */
    private static final String SYSTEM_PROMPT = "Reasoning: low\n"
            + "You are a strict evaluator for a retrieval-augmented QA system. "
            + "Judge the answer only against the provided context. "
            + "Respond with a single JSON object and nothing else.";

    private final HttpClient http;
    private final ObjectMapper om = new ObjectMapper();
    private final String generateUrl;
    private final int fastMaxTokens;
    private final int thoroughMaxTokens;
    private final double temperature;
    private final Duration readTimeout;

    public TritonJudgeClient(
            @Value("${judge.triton.base-url:${llm.triton.base-url:http://localhost:8000}}") String baseUrl,
            @Value("${judge.triton.model-name:nemotron}") String modelName,
            @Value("${judge.triton.fast-max-tokens:512}") int fastMaxTokens,
            @Value("${judge.triton.thorough-max-tokens:3072}") int thoroughMaxTokens,
            @Value("${judge.triton.temperature:0.1}") double temperature,
            @Value("${judge.triton.connect-timeout-seconds:5}") int connectTimeoutSeconds,
            @Value("${judge.triton.read-timeout-seconds:60}") int readTimeoutSeconds
    ) {
        String cleanBaseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.generateUrl = cleanBaseUrl + "/v2/models/" + modelName + "/generate";
        this.fastMaxTokens = fastMaxTokens;
        this.thoroughMaxTokens = thoroughMaxTokens;
        this.temperature = temperature;
        this.readTimeout = Duration.ofSeconds(readTimeoutSeconds);
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(connectTimeoutSeconds))
                .build();

        log.info("Triton judge enabled: url={} fastMaxTokens={} thoroughMaxTokens={} temperature={} readTimeout={}s",
                generateUrl, fastMaxTokens, thoroughMaxTokens, temperature, readTimeoutSeconds);
    }

    @Override
    public String evaluate(String prompt, List<String> contextChunks) {
        return generate(prompt, contextChunks, true);
    }

    @Override
    public boolean supportsFastMode() {
        return true;
    }

    @Override
    public String evaluateFast(String prompt, List<String> contextChunks) {
        return generate(prompt, contextChunks, false);
    }

    private String generate(String prompt, List<String> contextChunks, boolean thinking) {
        String context = (contextChunks == null || contextChunks.isEmpty())
                ? "(no context provided)"
                : String.join("\n\n---\n\n", contextChunks);
        String user = "CONTEXT:\n" + context + "\n\n" + prompt;
        String mode = thinking ? "thorough" : "fast";

        Map<String, Object> parameters = new LinkedHashMap<>();
        parameters.put("stream", false);
        parameters.put("exclude_input_in_output", true);
        parameters.put("max_tokens", thinking ? thoroughMaxTokens : fastMaxTokens);
        parameters.put("temperature", temperature);

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("text_input", renderChatTemplate(user, thinking));
        payload.put("parameters", parameters);

        long start = System.nanoTime();
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(generateUrl))
                    .timeout(readTimeout)
                    .header("Content-Type", "application/json")
                    .header("Accept", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(om.writeValueAsString(payload), StandardCharsets.UTF_8))
                    .build();

            HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            long latencyMs = (System.nanoTime() - start) / 1_000_000;

            if (response.statusCode() >= 300) {
                throw new IllegalStateException("Triton judge HTTP " + response.statusCode() + ": "
                        + truncate(response.body(), 500));
            }

            JsonNode root = om.readTree(response.body());
            if (root.hasNonNull("error")) {
                throw new IllegalStateException("Triton judge error: " + root.get("error").asText());
            }
            String text = root.path("text_output").asText("");

            if (thinking && !text.contains(THINK_END)) {
                throw new IllegalStateException("Triton judge reasoning cut off at max_tokens=" + thoroughMaxTokens
                        + " (no closing think tag, no verdict). Raise judge.triton.thorough-max-tokens.");
            }
            String cleaned = stripReasoning(text);
            log.info("Triton judge: mode={} latencyMs={} outChars={}", mode, latencyMs, cleaned.length());

            if (cleaned.isBlank()) {
                throw new IllegalStateException("Triton judge returned no final content (mode=" + mode + ")");
            }
            return cleaned;

        } catch (HttpTimeoutException e) {
            throw new IllegalStateException("Triton judge timed out after " + readTimeout.toSeconds()
                    + "s (mode=" + mode + ", url=" + generateUrl + ")", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Triton judge call interrupted", e);
        } catch (IOException e) {
            throw new IllegalStateException("Triton judge call failed (url=" + generateUrl + "): " + e.getMessage(), e);
        }
    }

    /** Nemotron-3 ChatML template; empty think block = reasoning off, open think block = reasoning on. */
    private static String renderChatTemplate(String user, boolean thinking) {
        return IM_START + "system\n" + SYSTEM_PROMPT + IM_END + "\n"
                + IM_START + "user\n" + user + IM_END + "\n"
                + IM_START + "assistant\n"
                + (thinking ? "<think>\n" : "<think></think>");
    }

    static String stripReasoning(String text) {
        String out = text == null ? "" : text;
        int end = out.indexOf(THINK_END);
        if (end >= 0) {
            out = out.substring(end + THINK_END.length());
        }
        return out.replace(IM_END, "").trim();
    }

    private static String truncate(String text, int max) {
        if (text == null) {
            return "";
        }
        return text.length() <= max ? text : text.substring(0, max) + "...";
    }
}
