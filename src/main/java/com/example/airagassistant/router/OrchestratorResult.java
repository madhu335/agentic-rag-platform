package com.example.airagassistant.router;

import com.example.airagassistant.judge.JudgeResult;

import java.util.List;

public record OrchestratorResult(
        String routeUsed,
        String answer,
        List<String> retrievedChunkIds,
        List<String> citedChunkIds,
        List<Chunk> chunks,
        int usedChunks,
        Double bestScore,
        String reason,
        String outcome,
        JudgeResult judge
) {
    public record Chunk(String id, String text) {}
}