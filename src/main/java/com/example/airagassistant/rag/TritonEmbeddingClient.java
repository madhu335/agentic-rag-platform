package com.example.airagassistant.rag;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Triton embedding client.
 *
 * Models:
 *   text_embedding  BAAI/bge-base-en-v1.5, 768 dims, symmetric (input TEXT only)
 *   nemotron_embed  Nemotron 3.5 Embed 1B, 2048 dims, asymmetric (TEXT + INPUT_TYPE query|passage)
 *
 * Set embedding.triton.send-input-type=true for nemotron_embed. It must stay false for
 * text_embedding, which rejects the unknown INPUT_TYPE input.
 *
 * embedBatch() splits the input into sub-batches of at most maxBatchSize texts.
 * Triton rejects (HTTP 400) any request whose batch dimension exceeds the model's
 * max_batch_size in config.pbtxt. Bulk ingestion embeds a whole page at once
 * (50 vehicles = 250+ chunk texts), so without sub-batching every page failed.
 *
 * Configure with embedding.triton.max-batch-size (default 32). It must be
 * less than or equal to max_batch_size in the model config.
 */
@Slf4j
@Component
public class TritonEmbeddingClient {

    public enum InputType {
        QUERY("query"),
        PASSAGE("passage");

        private final String wire;

        InputType(String wire) {
            this.wire = wire;
        }
    }

    private final WebClient webClient;
    private final String modelName;
    private final int maxBatchSize;
    private final boolean sendInputType;

    public TritonEmbeddingClient(
            WebClient.Builder builder,
            @Value("${embedding.triton.base-url}") String baseUrl,
            @Value("${embedding.triton.model-name}") String modelName,
            @Value("${embedding.triton.max-batch-size:32}") int maxBatchSize,
            @Value("${embedding.triton.send-input-type:false}") boolean sendInputType
    ) {
        this.webClient = builder
                .baseUrl(baseUrl)
                // a 32 x 2048 float response is ~1.4 MB of JSON; raise the 256 KB default buffer
                .codecs(c -> c.defaultCodecs().maxInMemorySize(16 * 1024 * 1024))
                .build();
        this.modelName = modelName;
        this.maxBatchSize = Math.max(1, maxBatchSize);
        this.sendInputType = sendInputType;
        log.info("TritonEmbeddingClient -> {} model={} maxBatchSize={} sendInputType={}",
                baseUrl, modelName, this.maxBatchSize, sendInputType);
    }

    /** Backward-compatible: a single text is treated as a search query. */
    public List<Double> embed(String text) {
        return embed(text, InputType.QUERY);
    }

    public List<Double> embed(String text, InputType type) {
        List<List<Double>> result = embedBatch(List.of(text == null ? "" : text), type);
        if (result.isEmpty()) {
            throw new IllegalStateException("Empty embedding response from Triton");
        }
        return result.get(0);
    }

    /** Backward-compatible: a batch is treated as stored passages. */
    public List<List<Double>> embedBatch(List<String> texts) {
        return embedBatch(texts, InputType.PASSAGE);
    }

    /**
     * Embeds any number of texts, preserving input order.
     * Sends ceil(N / maxBatchSize) sequential Triton requests.
     */
    public List<List<Double>> embedBatch(List<String> texts, InputType type) {
        if (texts == null || texts.isEmpty()) {
            return List.of();
        }

        List<List<Double>> all = new ArrayList<>(texts.size());
        for (int start = 0; start < texts.size(); start += maxBatchSize) {
            int end = Math.min(start + maxBatchSize, texts.size());
            List<List<Double>> part = inferOnce(texts.subList(start, end), type);
            if (part.size() != end - start) {
                throw new IllegalStateException(
                        "Triton returned " + part.size() + " embeddings for a sub-batch of " + (end - start));
            }
            all.addAll(part);
        }
        return all;
    }

    @SuppressWarnings("unchecked")
    private List<List<Double>> inferOnce(List<String> texts, InputType type) {
        List<List<String>> data = texts.stream()
                .map(t -> List.of(t == null ? "" : t))
                .toList();

        List<Map<String, Object>> inputs = new ArrayList<>(2);
        inputs.add(tensor("TEXT", data));
        if (sendInputType) {
            List<List<String>> types = Collections.nCopies(texts.size(), List.of(type.wire));
            inputs.add(tensor("INPUT_TYPE", types));
        }

        Map<String, Object> body = Map.of("inputs", inputs);

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

    private static Map<String, Object> tensor(String name, List<List<String>> rows) {
        Map<String, Object> t = new LinkedHashMap<>();
        t.put("name", name);
        t.put("shape", List.of(rows.size(), 1));
        t.put("datatype", "BYTES");
        t.put("data", rows);
        return t;
    }
}
