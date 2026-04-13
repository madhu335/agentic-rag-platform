package com.example.airagassistant.rag;

import java.util.List;

public record VectorRecord(
        long documentId,
        int chunkIndex,
        String id,
        String text,
        List<Double> vector
) {}