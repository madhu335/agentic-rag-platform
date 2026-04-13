package com.example.airagassistant.llm.ollama;

public record OllamaEmbeddingsRequest(
        String model,
        String prompt
) {}
