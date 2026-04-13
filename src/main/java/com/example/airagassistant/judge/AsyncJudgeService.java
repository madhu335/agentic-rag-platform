package com.example.airagassistant.judge;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
@Slf4j
public class AsyncJudgeService {

    private final JudgeService judgeService;

    @Async
    public void evaluateAsync(String question, String answer, List<String> chunks, String routeUsed) {
        try {
            JudgeResult result = judgeService.evaluate(question, answer, chunks);

            log.info("ASYNC JUDGE → route={} score={} grounded={} correct={} complete={}",
                    routeUsed,
                    result.score(),
                    result.grounded(),
                    result.correct(),
                    result.complete()
            );

        } catch (Exception e) {
            log.warn("Async judge failed: {}", e.getMessage());
        }
    }
}