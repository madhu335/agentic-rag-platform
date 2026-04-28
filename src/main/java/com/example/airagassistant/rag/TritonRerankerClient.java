package com.example.airagassistant.rag;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Slf4j
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

    /**
     * Scores documents against the query. Chunks into sub-batches of MAX_BATCH
     * because the Triton {@code cross_reranker} model is configured with
     * {@code max_batch_size: 32} and rejects larger requests with HTTP 400.
     * Order of returned scores matches the input document order.
     */
    public List<Double> score(String query, List<String> documents) {
        if (documents == null || documents.isEmpty()) {
            return List.of();
        }

        final int MAX_BATCH = 32;
        if (documents.size() <= MAX_BATCH) {
            return scoreBatch(query, documents);
        }

        int numBatches = (documents.size() + MAX_BATCH - 1) / MAX_BATCH;
        log.debug("Reranker chunking {} documents into {} sub-batches of <= {}",
                documents.size(), numBatches, MAX_BATCH);

        List<Double> all = new ArrayList<>(documents.size());
        for (int i = 0; i < documents.size(); i += MAX_BATCH) {
            int end = Math.min(i + MAX_BATCH, documents.size());
            all.addAll(scoreBatch(query, documents.subList(i, end)));
        }
        return all;
    }

    @SuppressWarnings("unchecked")
    private List<Double> scoreBatch(String query, List<String> documents) {
        // KFServing v2 spec: BYTES inputs with shape [N,1] are passed as a
        // FLAT list of N strings. Triton reshapes internally based on the
        // shape header. Sending a nested List<List<String>> trips strict
        // parsers — the article path hit 400 Bad Request because article
        // body windows contain newlines and unicode that some Triton builds
        // refuse to parse out of the nested form.
        String safeQuery = query == null ? "" : query;
        List<String> queries = documents.stream()
                .map(d -> safeQuery)
                .toList();

        List<String> docs = documents.stream()
                .map(d -> d == null ? "" : d)
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
                .onStatus(
                        status -> status.is4xxClientError() || status.is5xxServerError(),
                        resp -> resp.bodyToMono(String.class)
                                .defaultIfEmpty("")
                                .map(errBody -> new IllegalStateException(
                                        "Triton reranker " + resp.statusCode().value()
                                                + " — " + errBody))
                )
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