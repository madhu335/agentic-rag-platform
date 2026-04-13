package com.example.airagassistant.eval;

import java.util.List;

public record EvalCase(
        String id,
        String question,
        List<String> expectedAnswerContains,
        List<String> expectedChunkIds
) {}