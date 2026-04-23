package com.example.airagassistant.rag;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@Primary
@ConditionalOnProperty(name = "embedding.provider", havingValue = "triton")
public class TritonEmbeddingAdapter implements EmbeddingClient {

    private final TritonEmbeddingClient tritonEmbeddingClient;

    public TritonEmbeddingAdapter(TritonEmbeddingClient tritonEmbeddingClient) {
        this.tritonEmbeddingClient = tritonEmbeddingClient;
    }

    @Override
    public List<Double> embed(String text) {
        return tritonEmbeddingClient.embed(text);
    }

    @Override
    public List<List<Double>> embedBatch(List<String> texts) {
        return tritonEmbeddingClient.embedBatch(texts);
    }
}