package com.example.airagassistant.judge;

import com.example.airagassistant.trace.TraceHelper;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class JudgeServiceTest {

    @Test
    void parsesFencedJsonResponse() {
        JudgeClient judgeClient = mock(JudgeClient.class);
        TraceHelper traceHelper = mock(TraceHelper.class);
        ObjectMapper objectMapper = new ObjectMapper();

        when(traceHelper.run(any(), any(), any())).thenAnswer(invocation -> {
            var supplier = invocation.getArgument(2, java.util.function.Supplier.class);
            return supplier.get();
        });

        when(judgeClient.evaluate(any(), any())).thenReturn("""
                ```json
                {"grounded":true,"correct":true,"complete":true,"score":0.8,"reason":"ok"}
                ```
                """);

        JudgeService service = new JudgeService(judgeClient, objectMapper, traceHelper);

        JudgeResult result = service.evaluate("q", "a", List.of("ctx1"));

        assertTrue(result.grounded());
        assertTrue(result.correct());
        assertTrue(result.complete());
        assertEquals(0.8, result.score(), 0.0001);
        assertEquals("ok", result.reason());
    }

    @Test
    void fallsBackWhenResponseIsNotJson() {
        JudgeClient judgeClient = mock(JudgeClient.class);
        TraceHelper traceHelper = mock(TraceHelper.class);
        ObjectMapper objectMapper = new ObjectMapper();

        when(traceHelper.run(any(), any(), any())).thenAnswer(invocation -> {
            var supplier = invocation.getArgument(2, java.util.function.Supplier.class);
            return supplier.get();
        });

        when(judgeClient.evaluate(any(), any()))
                .thenReturn("I think this answer looks good")
                .thenReturn("Still not valid json");

        JudgeService service = new JudgeService(judgeClient, objectMapper, traceHelper);

        JudgeResult result = service.evaluate("q", "a", List.of("ctx1"));

        assertFalse(result.grounded());
        assertFalse(result.correct());
        assertFalse(result.complete());
        assertEquals(0.0, result.score(), 0.0001);
        assertEquals("judge_unavailable", result.reason());
    }
}