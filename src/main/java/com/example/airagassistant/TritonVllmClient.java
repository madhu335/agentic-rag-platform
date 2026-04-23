package com.example.airagassistant;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

@Component
@ConditionalOnProperty(name = "llm.provider", havingValue = "triton-vllm")
public class TritonVllmClient implements LlmClient {

    private final WebClient webClient;
    private final String modelName;

    public TritonVllmClient(
            WebClient.Builder builder,
            @Value("${llm.triton.base-url}") String baseUrl,
            @Value("${llm.triton.model-name}") String modelName
    ) {
        this.webClient = builder.baseUrl(baseUrl).build();
        this.modelName = modelName;
    }

    @Override
    @SuppressWarnings("unchecked")
    public String answer(String question, List<String> contextChunks) {
        String prompt = buildPrompt(question, contextChunks);

        Map<String, Object> request = Map.of(
                "inputs", List.of(
                        Map.of(
                                "name", "text_input",
                                "shape", List.of(1),
                                "datatype", "BYTES",
                                "data", List.of(prompt)
                        )
                )
        );

        Map<String, Object> response = webClient.post()
                .uri("/v2/models/{model}/infer", modelName)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(request)
                .retrieve()
                .bodyToMono(Map.class)
                .timeout(Duration.ofSeconds(120))
                .block();

        if (response == null || !response.containsKey("outputs")) {
            return "I don't know.";
        }

        List<Map<String, Object>> outputs = (List<Map<String, Object>>) response.get("outputs");

        Map<String, Object> textOutput = outputs.stream()
                .filter(o -> "text_output".equals(o.get("name")))
                .findFirst()
                .orElse(null);

        if (textOutput == null) {
            return "I don't know.";
        }

        Object rawData = textOutput.get("data");
        if (!(rawData instanceof List<?> list) || list.isEmpty()) {
            return "I don't know.";
        }

        Object first = list.get(0);
        return first == null ? "I don't know." : first.toString().trim();
    }

    @Override
    public void streamAnswer(String question, List<String> contextChunks, Consumer<String> onToken) {
        throw new UnsupportedOperationException("Streaming not wired yet for Triton vLLM");
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