package com.example.airagassistant.llm.ollama;

import com.example.airagassistant.rag.EmbeddingClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.util.List;

@Component("ollamaEmbeddingClient")
public class OllamaEmbeddingClient implements EmbeddingClient {

    private final RestClient restClient;

    @Value("${ollama.embedding.model:nomic-embed-text}")
    private String embeddingModel;

    public OllamaEmbeddingClient(
            @Value("${ollama.base-url:http://localhost:11434}") String baseUrl,
            RestClient.Builder builder
    ) {
        this.restClient = builder.baseUrl(baseUrl).build();
    }

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
}
