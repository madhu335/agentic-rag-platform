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
import java.util.regex.Pattern;

/**
 * Judge client for gpt-oss-120b running on the Halo box (OpenAI-compatible API).
 *
 * Differences from OpenAiClient that matter here:
 *  - base URL already contains the API prefix (/api/v1), so we append only /chat/completions
 *  - gpt-oss is a reasoning model: reasoning tokens count against max_tokens, and some servers
 *    leak reasoning / harmony channel markers into "content", so we strip them before
 *    JudgeService extracts the JSON object
 *  - explicit connect/read timeouts, because a 120B model is slower than the 8B on the 5090
 */
@Slf4j
@Component
@ConditionalOnProperty(name = "judge.provider", havingValue = "halo")
public class HaloJudgeClient implements JudgeClient {

    private static final Pattern THINK_BLOCK = Pattern.compile("(?s)<think>.*?</think>");
    private static final String HARMONY_FINAL = "<|channel|>final<|message|>";

    private final HttpClient http;
    private final ObjectMapper om = new ObjectMapper();

    private final String chatUrl;
    private final String model;
    private final String apiKey;
    private final String reasoningEffort;
    private final double temperature;
    private final int maxTokens;
    private final Duration readTimeout;

    public HaloJudgeClient(
            @Value("${judge.halo.base-url}") String baseUrl,
            @Value("${judge.halo.model}") String model,
            @Value("${judge.halo.api-key:}") String apiKey,
            @Value("${judge.halo.reasoning-effort:low}") String reasoningEffort,
            @Value("${judge.halo.temperature:0.1}") double temperature,
            @Value("${judge.halo.max-tokens:1024}") int maxTokens,
            @Value("${judge.halo.connect-timeout-seconds:5}") int connectTimeoutSeconds,
            @Value("${judge.halo.read-timeout-seconds:90}") int readTimeoutSeconds
    ) {
        String cleanBaseUrl = baseUrl.endsWith("/")
                ? baseUrl.substring(0, baseUrl.length() - 1)
                : baseUrl;

        this.chatUrl = cleanBaseUrl + "/chat/completions";
        this.model = model;
        this.apiKey = apiKey;
        this.reasoningEffort = reasoningEffort;
        this.temperature = temperature;
        this.maxTokens = maxTokens;
        this.readTimeout = Duration.ofSeconds(readTimeoutSeconds);
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(connectTimeoutSeconds))
                .build();

        log.info("Halo judge enabled: url={} model={} reasoningEffort={} maxTokens={} readTimeout={}s",
                chatUrl, model, reasoningEffort, maxTokens, readTimeoutSeconds);
    }

    @Override
    public String evaluate(String prompt, List<String> contextChunks) {
        String context = (contextChunks == null || contextChunks.isEmpty())
                ? "(no context provided)"
                : String.join("\n\n---\n\n", contextChunks);

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("model", model);
        payload.put("temperature", temperature);
        payload.put("max_tokens", maxTokens);
        payload.put("stream", false);
        payload.put("messages", List.of(
                Map.of(
                        "role", "system",
                        "content", "Reasoning: " + reasoningEffort + "\n"
                                + "You are a strict evaluator for a retrieval-augmented QA system. "
                                + "Judge the answer only against the provided context. "
                                + "Respond with a single JSON object and nothing else."
                ),
                Map.of(
                        "role", "user",
                        "content", "CONTEXT:\n" + context + "\n\n" + prompt
                )
        ));

        long start = System.nanoTime();

        try {
            String body = om.writeValueAsString(payload);

            HttpRequest.Builder request = HttpRequest.newBuilder()
                    .uri(URI.create(chatUrl))
                    .timeout(readTimeout)
                    .header("Content-Type", "application/json")
                    .header("Accept", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8));

            if (apiKey != null && !apiKey.isBlank()) {
                request.header("Authorization", "Bearer " + apiKey);
            }

            HttpResponse<String> response = http.send(
                    request.build(),
                    HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8)
            );

            long latencyMs = (System.nanoTime() - start) / 1_000_000;

            if (response.statusCode() >= 300) {
                throw new IllegalStateException("Halo judge HTTP " + response.statusCode()
                        + ": " + truncate(response.body(), 500));
            }

            JsonNode root = om.readTree(response.body());
            JsonNode choice = root.path("choices").path(0);
            String content = choice.path("message").path("content").asText("");
            String finishReason = choice.path("finish_reason").asText("");
            JsonNode usage = root.path("usage");

            log.info("Halo judge: latencyMs={} promptTokens={} completionTokens={} finishReason={}",
                    latencyMs,
                    usage.path("prompt_tokens").asInt(-1),
                    usage.path("completion_tokens").asInt(-1),
                    finishReason);

            String cleaned = stripReasoning(content);

            if (cleaned.isBlank()) {
                throw new IllegalStateException("Halo judge returned no final content (finish_reason="
                        + finishReason + "). If finish_reason=length, reasoning used up max_tokens: "
                        + "raise judge.halo.max-tokens or lower judge.halo.reasoning-effort.");
            }

            log.debug("Halo judge raw content={}", cleaned);
            return cleaned;

        } catch (HttpTimeoutException e) {
            throw new IllegalStateException("Halo judge timed out after " + readTimeout.toSeconds()
                    + "s (url=" + chatUrl + ")", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Halo judge call interrupted", e);
        } catch (IOException e) {
            throw new IllegalStateException("Halo judge call failed (url=" + chatUrl + "): "
                    + e.getMessage(), e);
        }
    }

    /**
     * Keeps only the final answer when a server leaks reasoning into "content":
     * harmony channel markers (gpt-oss) or <think> blocks.
     */
    static String stripReasoning(String content) {
        if (content == null) {
            return "";
        }

        String result = content;

        int finalIdx = result.lastIndexOf(HARMONY_FINAL);
        if (finalIdx >= 0) {
            result = result.substring(finalIdx + HARMONY_FINAL.length());
        }

        result = THINK_BLOCK.matcher(result).replaceAll("");
        result = result.replace("<|end|>", "").replace("<|return|>", "");

        return result.trim();
    }

    private static String truncate(String text, int max) {
        if (text == null) {
            return "";
        }
        return text.length() <= max ? text : text.substring(0, max) + "...";
    }
}
