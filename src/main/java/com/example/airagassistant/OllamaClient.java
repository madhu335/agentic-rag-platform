package com.example.airagassistant;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

@Component("ollamaLlmClient")
@ConditionalOnProperty(name = "llm.provider", havingValue = "ollama", matchIfMissing = true)
public class OllamaClient implements LlmClient {

    private final RestTemplate restTemplate = new RestTemplate();
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Value("${ollama.base-url}")
    private String baseUrl;

    @Value("${ollama.model}")
    private String model;

    @Override
    public String answer(String question, List<String> contextChunks) {

        String prompt = buildPrompt(question, contextChunks);

        Map<String, Object> body = new HashMap<>();
        body.put("model", model);
        body.put("prompt", prompt);
        body.put("stream", false);

        String url = baseUrl + "/api/generate";

        Map<?, ?> response = restTemplate.postForObject(url, body, Map.class);

        if (response == null || !response.containsKey("response")) {
            return "I don't know.";
        }

        return response.get("response").toString().trim();
    }

    @Override
    public void streamAnswer(String question, List<String> contextChunks, Consumer<String> onToken) {
        try {
            String prompt = buildPrompt(question, contextChunks);

            URL url = new URL(baseUrl + "/api/generate");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();

            conn.setRequestMethod("POST");
            conn.setDoOutput(true);
            conn.setRequestProperty("Content-Type", "application/json");

            String requestBody = """
        {
          "model": "%s",
          "prompt": %s,
          "stream": true
        }
        """.formatted(model, objectMapper.writeValueAsString(prompt));

            try (OutputStream os = conn.getOutputStream()) {
                os.write(requestBody.getBytes());
            }

            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(conn.getInputStream()))) {

                String line;

                while ((line = reader.readLine()) != null) {
                    if (line.isBlank()) continue;

                    JsonNode json = objectMapper.readTree(line);

                    if (json.has("response")) {
                        String token = json.get("response").asText();
                        onToken.accept(token);
                    }

                    if (json.has("done") && json.get("done").asBoolean()) {
                        break;
                    }
                }
            }

        } catch (Exception e) {
            throw new RuntimeException("Error streaming from Ollama", e);
        }
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
