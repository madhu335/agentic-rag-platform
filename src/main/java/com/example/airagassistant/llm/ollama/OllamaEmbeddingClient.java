package com.example.airagassistant.llm.ollama;

import com.example.airagassistant.rag.EmbeddingClient;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.util.ArrayList;
import java.util.List;

/**
 * Ollama embedding client with batch support.
 *
 * Two endpoints:
 *   - embed(text)       → /api/embeddings (legacy, single text, backward compat)
 *   - embedBatch(texts) → /api/embed      (batch, sends all texts in one call)
 *
 * embedBatch splits large batches into sub-batches of BATCH_SIZE to avoid
 * Ollama context overflow on very long inputs. Each sub-batch is one HTTP
 * round-trip. Within each round-trip, Ollama parallelizes embedding across
 * GOMAXPROCS workers.
 *
 * Batch sizing:
 *   nomic-embed-text context window is 8192 tokens. With ~600 word chunks
 *   (~800 tokens each), a batch of 8-10 chunks is safe. We default to 32
 *   which works for shorter chunks; Ollama auto-truncates if needed.
 *   If you see truncation warnings in Ollama logs, lower BATCH_SIZE.
 */
@Slf4j
@Component
@ConditionalOnProperty(name = "embedding.provider", havingValue = "ollama", matchIfMissing = true)
public class OllamaEmbeddingClient implements EmbeddingClient {

    private static final int BATCH_SIZE = 32;

    private final RestClient restClient;

    @Value("${ollama.embedding.model:nomic-embed-text}")
    private String embeddingModel;

    public OllamaEmbeddingClient(
            @Value("${ollama.base-url:http://localhost:11434}") String baseUrl,
            RestClient.Builder builder
    ) {
        this.restClient = builder.baseUrl(baseUrl).build();
    }

    // ─── Single embed (backward compatible, uses legacy endpoint) ─────────

    @Override
    public List<Double> embed(String text) {
        var req = new OllamaEmbeddingsRequest(embeddingModel, text);

        var res = restClient.post()
                .uri("/api/embeddings")
                .body(req)
                .retrieve()
                .body(OllamaEmbeddingsResponse.class);

        if (res == null || res.embedding() == null || res.embedding().isEmpty()) {
            throw new IllegalStateException("Empty embedding from Ollama");
        }
        return res.embedding();
    }

    // ─── Batch embed (uses /api/embed with input array) ───────────────────

    @Override
    public List<List<Double>> embedBatch(List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return List.of();
        }

        // Single text — use the single endpoint for simplicity
        if (texts.size() == 1) {
            return List.of(embed(texts.get(0)));
        }

        List<List<Double>> allEmbeddings = new ArrayList<>(texts.size());

        // Split into sub-batches to avoid context overflow
        for (int start = 0; start < texts.size(); start += BATCH_SIZE) {
            int end = Math.min(start + BATCH_SIZE, texts.size());
            List<String> subBatch = texts.subList(start, end);

            log.debug("Embedding batch {}-{} of {} texts", start, end - 1, texts.size());

            var req = new OllamaEmbedRequest(embeddingModel, subBatch);

            var res = restClient.post()
                    .uri("/api/embed")
                    .body(req)
                    .retrieve()
                    .body(OllamaEmbedResponse.class);

            if (res == null || res.embeddings() == null) {
                throw new IllegalStateException(
                        "Batch embedding returned null for texts " + start + "-" + (end - 1));
            }

            if (res.embeddings().size() != subBatch.size()) {
                throw new IllegalStateException(
                        "Expected " + subBatch.size() + " embeddings but got " + res.embeddings().size());
            }

            allEmbeddings.addAll(res.embeddings());
        }

        log.info("Batch embedded {} texts in {} round-trips",
                texts.size(), (int) Math.ceil((double) texts.size() / BATCH_SIZE));

        return allEmbeddings;
    }
}
