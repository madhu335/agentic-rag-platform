package com.example.airagassistant.rag;

import java.util.List;

public interface EmbeddingClient {
    List<Double> embed(String text);
}
