package com.example.airagassistant;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Component("ollamaLlmClient")
public class OllamaClient implements LlmClient {

    private final RestTemplate restTemplate = new RestTemplate();

    @Value("${ollama.base-url}")
    private String baseUrl;

    @Value("${ollama.model}")
    private String model;

    @Override
    public String answer(String question, List<String> contextChunks) {

        StringBuilder prompt = new StringBuilder();
        prompt.append("Answer the question using ONLY the context below.\n\n");

        for (String ctx : contextChunks) {
            prompt.append("- ").append(ctx).append("\n");
        }

        prompt.append("\nQuestion: ").append(question);
        prompt.append("\nAnswer:");

        Map<String, Object> body = new HashMap<>();
        body.put("model", model);
        body.put("prompt", prompt.toString());
        body.put("stream", false);

        String url = baseUrl + "/api/generate";

        Map<?, ?> response = restTemplate.postForObject(url, body, Map.class);

        if (response == null || !response.containsKey("response")) {
            return "I don't know.";
        }

        return response.get("response").toString().trim();
    }
}
