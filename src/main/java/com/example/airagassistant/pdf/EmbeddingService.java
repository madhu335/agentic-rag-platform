package com.example.airagassistant.pdf;

import com.example.airagassistant.llm.ollama.OllamaEmbeddingClient;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class EmbeddingService {

    private final OllamaEmbeddingClient ollamaEmbeddingClient;

    public List<Double> embed(String text) {
        return ollamaEmbeddingClient.embed(text);
    }
}