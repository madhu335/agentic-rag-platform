package com.example.airagassistant.rag;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Component
public class TritonEmbeddingClient {

    private final WebClient webClient;
    private final String modelName;

    public TritonEmbeddingClient(
            WebClient.Builder builder,
            @Value("${embedding.triton.base-url}") String baseUrl,
            @Value("${embedding.triton.model-name}") String modelName
    ) {
        this.webClient = builder.baseUrl(baseUrl).build();
        this.modelName = modelName;
    }

    public List<Double> embed(String text) {
        List<List<Double>> result = embedBatch(List.of(text == null ? "" : text));
        if (result.isEmpty()) {
            throw new IllegalStateException("Empty embedding response from Triton");
        }
        return result.get(0);
    }

    @SuppressWarnings("unchecked")
    public List<List<Double>> embedBatch(List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return List.of();
        }

        List<List<String>> data = texts.stream()
                .map(t -> List.of(t == null ? "" : t))
                .toList();

        Map<String, Object> body = Map.of(
                "inputs", List.of(
                        Map.of(
                                "name", "TEXT",
                                "shape", List.of(texts.size(), 1),
                                "datatype", "BYTES",
                                "data", data
                        )
                )
        );

        Map<String, Object> response = webClient.post()
                .uri("/v2/models/{model}/infer", modelName)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(Map.class)
                .timeout(Duration.ofSeconds(60))
                .block();

        if (response == null || !response.containsKey("outputs")) {
            throw new IllegalStateException("Empty Triton response");
        }

        List<Map<String, Object>> outputs = (List<Map<String, Object>>) response.get("outputs");

        Map<String, Object> embeddingOutput = outputs.stream()
                .filter(o -> "EMBEDDING".equals(o.get("name")))
                .findFirst()
                .orElseThrow(() -> new IllegalStateException("EMBEDDING output missing"));

        Object rawData = embeddingOutput.get("data");
        if (!(rawData instanceof List<?> rawList) || rawList.isEmpty()) {
            throw new IllegalStateException("Embedding data missing or invalid");
        }

        Object rawShape = embeddingOutput.get("shape");
        if (!(rawShape instanceof List<?> shapeList) || shapeList.size() < 2) {
            throw new IllegalStateException("Embedding shape missing or invalid: " + rawShape);
        }

        int batchSize = ((Number) shapeList.get(0)).intValue();
        int dim = ((Number) shapeList.get(1)).intValue();

        if (rawList.size() != batchSize * dim) {
            throw new IllegalStateException(
                    "Embedding size mismatch: flat=" + rawList.size() +
                            " expected=" + (batchSize * dim)
            );
        }

        List<List<Double>> result = new ArrayList<>(batchSize);

        for (int i = 0; i < batchSize; i++) {
            List<Double> vector = new ArrayList<>(dim);
            for (int j = 0; j < dim; j++) {
                int index = i * dim + j;
                vector.add(((Number) rawList.get(index)).doubleValue());
            }
            result.add(vector);
        }

        return result;
    }
}