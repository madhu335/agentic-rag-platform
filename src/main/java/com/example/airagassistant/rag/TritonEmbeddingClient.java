package com.example.airagassistant.rag;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Triton embedding client.
 *
 * embedBatch() splits the input into sub-batches of at most maxBatchSize texts.
 * Triton rejects (HTTP 400) any request whose batch dimension exceeds the model's
 * max_batch_size in config.pbtxt. Bulk ingestion embeds a whole page at once
 * (50 vehicles ≈ 250+ chunk texts), so without sub-batching every page failed.
 *
 * Configure with embedding.triton.max-batch-size (default 32). It must be
 * less than or equal to max_batch_size in the text_embedding model config.
 */
@Component
public class TritonEmbeddingClient {

    private final WebClient webClient;
    private final String modelName;
    private final int maxBatchSize;

    public TritonEmbeddingClient(
            WebClient.Builder builder,
            @Value("${embedding.triton.base-url}") String baseUrl,
            @Value("${embedding.triton.model-name}") String modelName,
            @Value("${embedding.triton.max-batch-size:32}") int maxBatchSize
    ) {
        this.webClient = builder
                .baseUrl(baseUrl)
                // a 32 x 768 float response is ~0.5 MB of JSON; raise the 256 KB default buffer
                .codecs(c -> c.defaultCodecs().maxInMemorySize(16 * 1024 * 1024))
                .build();
        this.modelName = modelName;
        this.maxBatchSize = Math.max(1, maxBatchSize);
    }

    public List<Double> embed(String text) {
        List<List<Double>> result = embedBatch(List.of(text == null ? "" : text));
        if (result.isEmpty()) {
            throw new IllegalStateException("Empty embedding response from Triton");
        }
        return result.get(0);
    }

    /**
     * Embeds any number of texts, preserving input order.
     * Sends ceil(N / maxBatchSize) sequential Triton requests.
     */
    public List<List<Double>> embedBatch(List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return List.of();
        }

        List<List<Double>> all = new ArrayList<>(texts.size());
        for (int start = 0; start < texts.size(); start += maxBatchSize) {
            int end = Math.min(start + maxBatchSize, texts.size());
            List<List<Double>> part = inferOnce(texts.subList(start, end));
            if (part.size() != end - start) {
                throw new IllegalStateException(
                        "Triton returned " + part.size() + " embeddings for a sub-batch of " + (end - start));
            }
            all.addAll(part);
        }
        return all;
    }

    @SuppressWarnings("unchecked")
    private List<List<Double>> inferOnce(List<String> texts) {
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
