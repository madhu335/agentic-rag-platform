package com.example.airagassistant.eval;

import java.util.List;

public record AnswerEvalResult(
        String caseId,
        String question,
        String answer,
        List<String> expectedAnswerContains,
        List<String> missingExpectedPhrases,
        boolean pass
) {}