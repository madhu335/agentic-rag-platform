package com.example.airagassistant.router;

import com.example.airagassistant.rag.VehicleCardDto;
import com.example.airagassistant.judge.JudgeResult;

import java.util.List;

public record OrchestratorResult(
        String routeUsed,
        String answer,
        List<String> retrievedChunkIds,
        List<String> citedChunkIds,
        List<VehicleCardDto> cards,
        List<Chunk> chunks,
        int usedChunks,
        Double retrievalScore,
        String reason,
        String outcome,
        JudgeResult judge
) {
    public record Chunk(String id, String text) {}
}