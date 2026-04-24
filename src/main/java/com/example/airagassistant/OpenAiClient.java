package com.example.airagassistant;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.util.*;
import java.util.function.Consumer;

@Slf4j
@Component("openAiClient")
public class OpenAiClient implements LlmClient {

    private final RestTemplate restTemplate = new RestTemplate();
    private final ObjectMapper om = new ObjectMapper();

    @Value("${openai.base-url:http://localhost:8001}")
    private String baseUrl;

    @Value("${openai.api.key:}")
    private String apiKey;

    @Value("${openai.model:meta-llama/Meta-Llama-3.1-8B-Instruct}")
    private String model;

    @Override
    public String answer(String question, List<String> contextChunks) {
        try {
            String context = String.join("\n\n---\n\n", contextChunks);

            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("model", model);
            payload.put("temperature", 0.2);

            List<Map<String, String>> messages = new ArrayList<>();

            Map<String, String> system = new LinkedHashMap<>();
            system.put("role", "system");
            system.put("content", "You are a helpful assistant. Answer ONLY using the provided context. If missing, say you don't know.");
            messages.add(system);

            Map<String, String> user = new LinkedHashMap<>();
            user.put("role", "user");
            user.put("content", "CONTEXT:\n" + context + "\n\nQUESTION:\n" + question);
            messages.add(user);

            payload.put("messages", messages);

            String cleanBaseUrl = baseUrl.endsWith("/")
                    ? baseUrl.substring(0, baseUrl.length() - 1)
                    : baseUrl;

            String url = cleanBaseUrl + "/v1/chat/completions";

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.setAccept(List.of(MediaType.APPLICATION_JSON));

            if (apiKey != null && !apiKey.isBlank()) {
                headers.setBearerAuth(apiKey);
            }

            String json = om.writeValueAsString(payload);

            log.info("OpenAI-compatible URL={}", url);
            log.info("OpenAI-compatible payload length={}", json.length());
            log.debug("OpenAI-compatible payload={}", json);

            HttpEntity<String> request = new HttpEntity<>(json, headers);

            ResponseEntity<String> response = restTemplate.exchange(
                    url,
                    HttpMethod.POST,
                    request,
                    String.class
            );

            if (!response.getStatusCode().is2xxSuccessful()) {
                throw new RuntimeException("OpenAI-compatible error "
                        + response.getStatusCode().value()
                        + ": "
                        + response.getBody());
            }

            JsonNode root = om.readTree(response.getBody());
            return root.path("choices")
                    .get(0)
                    .path("message")
                    .path("content")
                    .asText();

        } catch (Exception e) {
            throw new RuntimeException("OpenAI-compatible client call failed", e);
        }
    }

    @Override
    public void streamAnswer(String question, List<String> contextChunks, Consumer<String> onToken) {
        onToken.accept(answer(question, contextChunks));
    }
}