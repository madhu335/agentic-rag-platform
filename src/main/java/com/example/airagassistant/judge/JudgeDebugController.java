package com.example.airagassistant.judge;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * Dev-only endpoint to exercise the judge in isolation (no retrieval, no answer generation).
 * Enabled with judge.debug.enabled=true.
 */
@RestController
@RequestMapping("/api/judge")
@ConditionalOnProperty(name = "judge.debug.enabled", havingValue = "true")
public class JudgeDebugController {

    private final JudgeService judgeService;
    private final String provider;

    public JudgeDebugController(
            JudgeService judgeService,
            @Value("${judge.provider:default}") String provider
    ) {
        this.judgeService = judgeService;
        this.provider = provider;
    }

    public record JudgeTestRequest(String question, String answer, List<String> context) {}

    public record JudgeTestResponse(String provider, long latencyMs, JudgeResult result) {}

    @PostMapping("/test")
    public JudgeTestResponse test(@RequestBody JudgeTestRequest request) {
        long start = System.nanoTime();

        JudgeResult result = judgeService.evaluate(
                request.question(),
                request.answer(),
                request.context() == null ? List.of() : request.context()
        );

        long latencyMs = (System.nanoTime() - start) / 1_000_000;
        return new JudgeTestResponse(provider, latencyMs, result);
    }
}
