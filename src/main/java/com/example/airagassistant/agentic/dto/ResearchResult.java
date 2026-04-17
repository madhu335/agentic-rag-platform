package com.example.airagassistant.agentic.dto;

import com.example.airagassistant.judge.JudgeResult;

import java.util.List;

public record ResearchResult(
        String answer,
        List<String> citations,
        Double retrievalScore,
        JudgeResult judge,
        List<String> chunks
) {}