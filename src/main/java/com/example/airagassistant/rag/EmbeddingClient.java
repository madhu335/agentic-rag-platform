package com.example.airagassistant.rag;

import java.util.List;

public interface EmbeddingClient {

    List<Double> embed(String text);

    /**
     * Batch-embed multiple texts in a single call.
     *
     * Default implementation calls embed() serially for backward compatibility.
     * OllamaEmbeddingClient overrides this to use Ollama's /api/embed batch
     * endpoint, which processes all texts in one HTTP round-trip.
     *
     * Performance impact:
     *   - 20 chunks × serial embed():  20 HTTP round-trips × ~20ms = ~400ms
     *   - 20 chunks × embedBatch():    1 HTTP round-trip  × ~80ms  = ~80ms
     *   - 100 chunks × serial embed(): 100 round-trips    × ~20ms  = ~2000ms
     *   - 100 chunks × embedBatch():   2 round-trips (batch of 50) × ~200ms = ~400ms
     */
    default List<List<Double>> embedBatch(List<String> texts) {
        return texts.stream().map(this::embed).toList();
    }
}
