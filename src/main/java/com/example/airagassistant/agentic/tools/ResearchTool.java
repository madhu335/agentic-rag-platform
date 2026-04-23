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
        String docType = inferDocType(docId);

        OrchestratorResult result = orchestratorService.handle(
                prompt,
                docType,
                docId,
                topK
        );

        return new ResearchResult(
                result.answer(),
                result.citedChunkIds(),
                result.retrievalScore(),
                result.judge(),
                result.chunks().stream().map(OrchestratorResult.Chunk::text).toList()
        );
    }

    private String inferDocType(String docId) {
        if (docId == null || docId.isBlank()) {
            return "article";
        }

        String id = docId.trim().toLowerCase();

        if (id.equals("vehicles") || id.equals("fleet") || id.equals("all-vehicles")) {
            return "vehicle";
        }

        if (id.equals("articles") || id.equals("all-articles")) {
            return "article";
        }

        if (id.startsWith("motortrend-") || id.contains("review")) {
            return "article";
        }

        return "vehicle";
    }
}