package com.example.airagassistant.llm.ollama;

import java.util.List;

/**
 * Response from Ollama's /api/embed endpoint (batch mode).
 *
 * Returns "embeddings" (plural) — a list of vectors, one per input text,
 * in the same order as the input array.
 *
 * Note: the legacy /api/embeddings endpoint returns "embedding" (singular).
 * This DTO is for the newer /api/embed endpoint only.
 */
public record OllamaEmbedResponse(
        List<List<Double>> embeddings
) {}
