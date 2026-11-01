package com.example.airagassistant.judge;

import jakarta.annotation.PreDestroy;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Background judge for answers that were returned without a synchronous judge (low-risk answers).
 * The user never waits; the verdict is logged and traced for quality monitoring.
 *
 * Uses its own small, bounded thread pool instead of Spring's @Async: @EnableAsync is not enabled in this app
 * (it is commented out in AiRagAssistantApplication, so the old @Async annotation here ran synchronously), and
 * enabling it globally would change any other @Async method. Bounded on purpose: judge calls share the GPU with
 * answer generation, so when the queue is full new audit jobs are DROPPED (counted and logged), never queued
 * without limit and never run on the request thread.
 */
@Slf4j
@Service
public class AsyncJudgeService {

    private final JudgeService judgeService;
    private final ThreadPoolExecutor executor;
    private final AtomicLong submitted = new AtomicLong();
    private final AtomicLong completed = new AtomicLong();
    private final AtomicLong dropped = new AtomicLong();
    private final AtomicLong failed = new AtomicLong();

    public AsyncJudgeService(
            JudgeService judgeService,
            @Value("${judge.async.threads:2}") int threads,
            @Value("${judge.async.queue-capacity:200}") int queueCapacity
    ) {
        this.judgeService = judgeService;
        int poolSize = Math.max(1, threads);
        AtomicInteger counter = new AtomicInteger();
        ThreadFactory factory = r -> {
            Thread t = new Thread(r, "async-judge-" + counter.incrementAndGet());
            t.setDaemon(true);
            return t;
        };
        this.executor = new ThreadPoolExecutor(poolSize, poolSize, 60, TimeUnit.SECONDS,
                new ArrayBlockingQueue<>(Math.max(1, queueCapacity)), factory,
                (job, pool) -> {
                    long n = dropped.incrementAndGet();
                    log.warn("ASYNC JUDGE queue full, audit dropped (dropped total={})", n);
                });
        log.info("AsyncJudgeService: threads={} queueCapacity={}", poolSize, queueCapacity);
    }

    /** Queues a background judge of the answer. Never blocks and never throws. */
    public void evaluateAsync(String question, String answer, List<String> chunks, String routeUsed) {
        submitted.incrementAndGet();
        executor.execute(() -> {
            long start = System.nanoTime();
            try {
                JudgeResult result = judgeService.evaluate(question, answer, chunks);
                completed.incrementAndGet();
                log.info("ASYNC JUDGE -> route={} verdict={} score={} stage={} latencyMs={} reason={}",
                        routeUsed, result.verdict(), result.score(), result.stage(),
                        (System.nanoTime() - start) / 1_000_000, result.reason());
                if (!result.accepted() && !result.isUnavailable()) {
                    log.warn("ASYNC JUDGE flagged a returned answer: route={} verdict={} reason={} question={}",
                            routeUsed, result.verdict(), result.reason(), question);
                }
            } catch (Exception e) {
                failed.incrementAndGet();
                log.warn("Async judge failed: {}", e.getMessage());
            }
        });
    }

    /** submitted / completed / dropped / failed / queued, for logs or an admin endpoint. */
    public String stats() {
        return "submitted=" + submitted.get() + " completed=" + completed.get() + " dropped=" + dropped.get()
                + " failed=" + failed.get() + " queued=" + executor.getQueue().size();
    }

    @PreDestroy
    void shutdown() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        log.info("AsyncJudgeService stopped: {}", stats());
    }
}
