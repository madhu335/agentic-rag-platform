package com.example.airagassistant.pdf;

import com.example.airagassistant.llm.ollama.OllamaEmbeddingClient;
import com.example.airagassistant.rag.EmbeddingClient;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class EmbeddingService {

    private final EmbeddingClient embeddingClient;


    public List<Double> embed(String text) {
        return embeddingClient.embed(text);
    }
}