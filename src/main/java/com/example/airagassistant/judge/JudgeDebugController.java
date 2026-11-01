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
 *
 * mode (optional): "auto" (default) = JudgeService.evaluate (cascade on triton, the async-audit path);
 *                  "thorough"       = JudgeService.evaluateThorough (the orchestrator's synchronous path).
 * The response also reports what the orchestrator's risk check would decide for this question and answer.
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

    public record JudgeTestRequest(String question, String answer, List<String> context, String mode) {}

    public record JudgeTestResponse(String provider, String mode, long latencyMs, String questionRisk,
                                    String answerChecks, JudgeResult result) {}

    @PostMapping("/test")
    public JudgeTestResponse test(@RequestBody JudgeTestRequest request) {
        List<String> context = request.context() == null ? List.of() : request.context();
        String mode = "thorough".equalsIgnoreCase(request.mode()) ? "thorough" : "auto";

        long start = System.nanoTime();
        JudgeResult result = "thorough".equals(mode)
                ? judgeService.evaluateThorough(request.question(), request.answer(), context)
                : judgeService.evaluate(request.question(), request.answer(), context);
        long latencyMs = (System.nanoTime() - start) / 1_000_000;

        String questionRisk = QuestionRisk.reason(request.question());
        String answerChecks = AnswerChecks.check(request.question(), request.answer(), context).reason();
        return new JudgeTestResponse(provider, mode, latencyMs, questionRisk == null ? "none" : questionRisk,
                answerChecks, result);
    }
}
