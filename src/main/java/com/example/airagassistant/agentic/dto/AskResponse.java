package com.example.airagassistant.agentic.dto;

import com.example.airagassistant.judge.JudgeResult;
import com.example.airagassistant.rag.VehicleCardDto;

import java.util.List;

public record AskResponse(
        String answer,
        List<String> citedChunkIds,
        List<String> retrievedChunkIds,
        int usedChunks,
        Double retrievalScore,
        JudgeResult judge,
        List<VehicleCardDto> cards,
        List<ChunkDto> chunks
) {}