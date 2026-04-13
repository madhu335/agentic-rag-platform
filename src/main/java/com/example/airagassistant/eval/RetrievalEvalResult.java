package com.example.airagassistant.eval;

import java.util.List;

public record RetrievalEvalResult(
        String caseId,
        String question,
        List<String> returnedChunkIds,
        List<String> expectedChunkIds,
        double precisionAtK,
        double recallAtK,
        boolean hitAtK
) {}