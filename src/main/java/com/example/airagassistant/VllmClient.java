package com.example.airagassistant;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

@Component
@ConditionalOnProperty(name = "llm.provider", havingValue = "vllm")
public class VllmClient implements LlmClient {

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final WebClient webClient;
    private final String baseUrl;
    private final String model;

    public VllmClient(
            WebClient.Builder builder,
            @Value("${llm.vllm.base-url}") String baseUrl,
            @Value("${llm.vllm.model}") String model
    ) {
        this.webClient = builder.baseUrl(baseUrl).build();
        this.baseUrl = baseUrl;
        this.model = model;
    }

    @Override
    public String answer(String question, List<String> contextChunks) {
        try {
            String prompt = buildPrompt(question, contextChunks);

            URL url = new URL(baseUrl + "/v1/chat/completions");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();

            conn.setRequestMethod("POST");
            conn.setDoOutput(true);
            conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
            conn.setRequestProperty("Accept", "application/json");

            Map<String, Object> request = Map.of(
                    "model", model,
                    "messages", List.of(
                            Map.of(
                                    "role", "user",
                                    "content", prompt
                            )
                    )
            );

            String requestBody = objectMapper.writeValueAsString(request);

            try (OutputStream os = conn.getOutputStream()) {
                os.write(requestBody.getBytes(StandardCharsets.UTF_8));
            }

            int status = conn.getResponseCode();
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(
                            status >= 400 ? conn.getErrorStream() : conn.getInputStream(),
                            StandardCharsets.UTF_8
                    )
            );

            StringBuilder responseBody = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                responseBody.append(line);
            }

            String raw = responseBody.toString();

            if (status >= 400) {
                throw new RuntimeException("vLLM error: " + raw);
            }

            JsonNode root = objectMapper.readTree(raw);
            JsonNode content = root.path("choices").path(0).path("message").path("content");

            if (content.isMissingNode() || content.isNull()) {
                return "I don't know.";
            }

            return content.asText().trim();

        } catch (Exception e) {
            throw new RuntimeException("Error calling vLLM", e);
        }
    }

    @Override
    @SuppressWarnings("unchecked")
    public void streamAnswer(String question,
                             List<String> contextChunks,
                             Consumer<String> onToken) {

        String prompt = buildPrompt(question, contextChunks);

        Map<String, Object> request = Map.of(
                "model", model,
                "stream", true,
                "messages", List.of(
                        Map.of(
                                "role", "user",
                                "content", prompt
                        )
                )
        );

        webClient.post()
                .uri("/v1/chat/completions")
                .contentType(MediaType.APPLICATION_JSON)
                .accept(MediaType.TEXT_EVENT_STREAM)
                .bodyValue(request)
                .retrieve()
                .bodyToFlux(String.class)
                .subscribe(chunk -> {
                    if (chunk == null || chunk.isBlank()) {
                        return;
                    }

                    String[] lines = chunk.split("\\r?\\n");
                    for (String line : lines) {
                        if (line == null || line.isBlank()) {
                            continue;
                        }

                        if (!line.startsWith("data:")) {
                            continue;
                        }

                        String json = line.substring(5).trim();

                        if ("[DONE]".equals(json)) {
                            return;
                        }

                        try {
                            Map<String, Object> parsed = objectMapper.readValue(json, Map.class);
                            List<Map<String, Object>> choices =
                                    (List<Map<String, Object>>) parsed.get("choices");

                            if (choices == null || choices.isEmpty()) {
                                continue;
                            }

                            Map<String, Object> delta =
                                    (Map<String, Object>) choices.get(0).get("delta");

                            if (delta == null) {
                                continue;
                            }

                            Object content = delta.get("content");
                            if (content != null) {
                                onToken.accept(content.toString());
                            }
                        } catch (Exception ignored) {
                            // ignore malformed or partial chunks
                        }
                    }
                }, error -> {
                    throw new RuntimeException("Error streaming from vLLM", error);
                });
    }

    private String buildPrompt(String question, List<String> contextChunks) {
        StringBuilder sb = new StringBuilder();

        sb.append("You are a helpful assistant.\n\n");

        if (contextChunks != null && !contextChunks.isEmpty()) {
            sb.append("Context:\n");
            for (int i = 0; i < contextChunks.size(); i++) {
                sb.append(i + 1).append(". ").append(contextChunks.get(i)).append("\n");
            }
            sb.append("\n");
        }

        sb.append("Question:\n").append(question).append("\n\n");
        sb.append("Answer:");

        return sb.toString();
    }
}