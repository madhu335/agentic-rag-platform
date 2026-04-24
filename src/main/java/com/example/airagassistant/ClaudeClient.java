package com.example.airagassistant;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;


@Component("claudeClient")
public class ClaudeClient implements LlmClient {

    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(15))
            .build();

    private final ObjectMapper om = new ObjectMapper();

    @Value("${claude.base-url:https://api.anthropic.com}")
    private String baseUrl;

    @Value("${claude.api.key}")
    private String apiKey;

    @Value("${claude.model:claude-sonnet-4-6}")
    private String model;

    @Override
    public String answer(String question, List<String> contextChunks) {
        try {
            if (apiKey == null || apiKey.isBlank()) {
                throw new IllegalStateException("Missing ANTHROPIC_API_KEY / claude.api.key");
            }

            String context = String.join("\n\n---\n\n", contextChunks);
            String prompt = "CONTEXT:\n" + context + "\n\nQUESTION:\n" + question;

            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("model", model);
            payload.put("max_tokens", 512);
            payload.put("temperature", 0.2);
            payload.put("messages", List.of(
                    Map.of("role", "user", "content", prompt)
            ));

            String json = om.writeValueAsString(payload);

            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/v1/messages"))
                    .timeout(Duration.ofSeconds(30))
                    .header("x-api-key", apiKey)
                    .header("anthropic-version", "2023-06-01")
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(json, StandardCharsets.UTF_8))
                    .build();

            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));

            if (resp.statusCode() >= 300) {
                throw new RuntimeException("Claude error " + resp.statusCode() + ": " + resp.body());
            }

            JsonNode root = om.readTree(resp.body());
            JsonNode content = root.path("content");

            StringBuilder out = new StringBuilder();
            if (content.isArray()) {
                for (JsonNode block : content) {
                    if ("text".equals(block.path("type").asText())) {
                        out.append(block.path("text").asText());
                    }
                }
            }

            String result = out.toString().trim();
            result = result.replaceAll("(?i)\\n*Sources?:.*$", "").trim();
            return result.isEmpty() ? "I don't know." : result;

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void streamAnswer(String question, List<String> contextChunks, Consumer<String> onToken) {
        String fullAnswer = answer(question, contextChunks);
        onToken.accept(fullAnswer);
    }
}