package com.example.airagassistant.judge;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class JudgeServiceJsonExtractTest {

    @Test
    void extractsPlainJson() {
        String raw = """
                {"grounded":true,"correct":true,"complete":true,"score":0.8,"reason":"ok"}
                """;

        String json = extractJsonObject(raw);

        assertEquals(raw.trim(), json);
    }

    @Test
    void extractsJsonInsideMarkdownFence() {
        String raw = """
                ```json
                {"grounded":true,"correct":true,"complete":true,"score":0.8,"reason":"ok"}
                ```
                """;

        String json = extractJsonObject(raw);

        assertEquals("""
                {"grounded":true,"correct":true,"complete":true,"score":0.8,"reason":"ok"}
                """.trim(), json);
    }

    @Test
    void extractsJsonWhenTextAppearsBeforeIt() {
        String raw = """
                Here is the evaluation:
                {"grounded":false,"correct":false,"complete":false,"score":0.2,"reason":"not supported"}
                """;

        String json = extractJsonObject(raw);

        assertEquals("""
                {"grounded":false,"correct":false,"complete":false,"score":0.2,"reason":"not supported"}
                """.trim(), json);
    }

    // Copy this from JudgeService for now, or make it package-private static in JudgeService.
    private static String extractJsonObject(String text) {
        if (text == null || text.isBlank()) {
            return "{}";
        }

        String trimmed = text.trim();

        if (trimmed.startsWith("```")) {
            trimmed = trimmed
                    .replaceFirst("^```json\\s*", "")
                    .replaceFirst("^```\\s*", "")
                    .replaceFirst("\\s*```$", "")
                    .trim();
        }

        int firstBrace = trimmed.indexOf('{');
        int lastBrace = trimmed.lastIndexOf('}');

        if (firstBrace >= 0 && lastBrace > firstBrace) {
            return trimmed.substring(firstBrace, lastBrace + 1);
        }

        return trimmed;
    }
}