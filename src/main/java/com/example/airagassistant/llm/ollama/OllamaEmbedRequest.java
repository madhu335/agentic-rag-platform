package com.example.airagassistant.llm.ollama;

import java.util.List;

/**
 * Request body for Ollama's /api/embed endpoint (batch mode).
 *
 * Unlike /api/embeddings (legacy, single text), /api/embed accepts
 * an "input" field that can be a single string or an array of strings.
 * When input is an array, Ollama embeds all texts in one forward pass
 * using internal parallelism (GOMAXPROCS workers).
 *
 * See: https://docs.ollama.com/capabilities/embeddings
 */
public record OllamaEmbedRequest(
        String model,
        List<String> input
) {}
