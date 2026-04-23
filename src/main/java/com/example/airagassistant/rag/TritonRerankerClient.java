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
public class TritonRerankerClient {

    private final WebClient webClient;
    private final String modelName;

    public TritonRerankerClient(
            WebClient.Builder builder,
            @Value("${reranker.triton.base-url}") String baseUrl,
            @Value("${reranker.triton.model-name}") String modelName
    ) {
        this.webClient = builder.baseUrl(baseUrl).build();
        this.modelName = modelName;
    }

    @SuppressWarnings("unchecked")
    public List<Double> score(String query, List<String> documents) {
        if (documents == null || documents.isEmpty()) {
            return List.of();
        }

        List<List<String>> queries = documents.stream()
                .map(d -> List.of(query == null ? "" : query))
                .toList();

        List<List<String>> docs = documents.stream()
                .map(d -> List.of(d == null ? "" : d))
                .toList();

        Map<String, Object> body = Map.of(
                "inputs", List.of(
                        Map.of(
                                "name", "QUERY",
                                "shape", List.of(documents.size(), 1),
                                "datatype", "BYTES",
                                "data", queries
                        ),
                        Map.of(
                                "name", "DOCUMENT",
                                "shape", List.of(documents.size(), 1),
                                "datatype", "BYTES",
                                "data", docs
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
            throw new IllegalStateException("Empty Triton reranker response");
        }

        List<Map<String, Object>> outputs = (List<Map<String, Object>>) response.get("outputs");

        Map<String, Object> scoreOutput = outputs.stream()
                .filter(o -> "SCORE".equals(o.get("name")))
                .findFirst()
                .orElseThrow(() -> new IllegalStateException("SCORE output missing"));

        Object rawData = scoreOutput.get("data");
        if (!(rawData instanceof List<?> rawList)) {
            throw new IllegalStateException("Invalid SCORE data");
        }

        List<Double> scores = new ArrayList<>();

        // shape [N,1] may deserialize as List<List<Number>>
        if (!rawList.isEmpty() && rawList.get(0) instanceof List<?> firstRow) {
            for (Object rowObj : rawList) {
                List<?> row = (List<?>) rowObj;
                if (row.isEmpty()) {
                    scores.add(0.0);
                } else {
                    scores.add(((Number) row.get(0)).doubleValue());
                }
            }
            return scores;
        }

        // fallback: flat list
        for (Object value : rawList) {
            scores.add(((Number) value).doubleValue());
        }

        return scores;
    }
}