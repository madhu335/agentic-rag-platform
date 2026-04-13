package com.example.airagassistant.llm.ollama;

import java.util.List;

public record OllamaEmbeddingsResponse(
        List<Double> embedding
) {}
