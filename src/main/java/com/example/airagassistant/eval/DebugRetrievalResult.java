package com.example.airagassistant.eval;

import java.util.List;

public record DebugRetrievalResult(
        String question,
        List<String> vector,
        List<String> bm25,
        List<String> hybrid,
        List<String> hybridRerank
) {}