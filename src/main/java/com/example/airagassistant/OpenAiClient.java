package com.example.airagassistant;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.*;
import java.util.function.Consumer;

@Component("openAiLlmClient")
public class OpenAiClient implements LlmClient{

    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(15))
            .build();

    private final ObjectMapper om = new ObjectMapper();

    @Value("${openai.api.key}")
    private String apiKey;

    @Value("${openai.model:gpt-4.1-mini}")
    private String model;

    @Override
    public String answer(String question, List<String> contextChunks) {
        try {
            if (apiKey == null || apiKey.isBlank()) {
                throw new IllegalStateException("Missing OPENAI_API_KEY (set env var or openai.api.key).");
            }

            String context = String.join("\n\n---\n\n", contextChunks);

            // Chat Completions-style payload (simple + widely supported)
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("model", model);

            List<Map<String, String>> messages = new ArrayList<>();
            messages.add(Map.of(
                    "role", "system",
                    "content", "You are a helpful assistant. Answer ONLY using the provided context. If missing, say you don't know."
            ));
            messages.add(Map.of(
                    "role", "user",
                    "content", "CONTEXT:\n" + context + "\n\nQUESTION:\n" + question
            ));
            payload.put("messages", messages);
            payload.put("temperature", 0.2);

            String json = om.writeValueAsString(payload);

            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create("https://api.openai.com/v1/chat/completions"))
                    .timeout(Duration.ofSeconds(30))
                    .header("Authorization", "Bearer " + apiKey)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(json, StandardCharsets.UTF_8))
                    .build();

            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));

            if (resp.statusCode() >= 300) {
                throw new RuntimeException("OpenAI error " + resp.statusCode() + ": " + resp.body());
            }

            // Parse: choices[0].message.content
            Map<?, ?> root = om.readValue(resp.body(), Map.class);
            List<?> choices = (List<?>) root.get("choices");
            Map<?, ?> first = (Map<?, ?>) choices.get(0);
            Map<?, ?> message = (Map<?, ?>) first.get("message");
            return String.valueOf(message.get("content"));

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void streamAnswer(String question, List<String> contextChunks, Consumer<String> onToken) {
        // Temporary fallback (non-streaming)
        String fullAnswer = answer(question, contextChunks);
        onToken.accept(fullAnswer);
    }
}
