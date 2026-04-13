package com.example.airagassistant.agentic.tools;

import com.example.airagassistant.agentic.dto.ResearchResult;
import com.example.airagassistant.router.OrchestratorResult;
import com.example.airagassistant.router.OrchestratorService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class ResearchTool {

    private final OrchestratorService orchestratorService;

    public ResearchResult research(String docId, String prompt, int topK) {
        OrchestratorResult result = orchestratorService.handle(prompt, docId, topK);

        return new ResearchResult(
                result.answer(),
                result.citedChunkIds(),
                result.bestScore(),
                result.judge(),
                result.chunks().stream().map(OrchestratorResult.Chunk::text).toList()
        );
    }
}