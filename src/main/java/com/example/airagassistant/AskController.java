package com.example.airagassistant;

import com.example.airagassistant.judge.JudgeResult;
import com.example.airagassistant.router.OrchestratorResult;
import com.example.airagassistant.router.OrchestratorService;
import com.example.airagassistant.trace.TraceHelper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@CrossOrigin(origins = "http://localhost:5173")
@RestController
@RequiredArgsConstructor
@Slf4j
public class AskController {

    private final OrchestratorService orchestratorService;
    private final TraceHelper traceHelper;

    public record AskRequest(String docId, String question, Integer topK) {}

    public record AskResponse(
            String answer,
            List<String> retrievedChunkIds,
            List<String> citedChunkIds,
            List<ChunkDto> chunks,
            int usedChunks,
            Double bestScore,
            JudgeResult judge
    ) {}

    public record ChunkDto(
            String id,
            String text
    ) {}

    @PostMapping("/ask")
    public AskResponse ask(@RequestBody AskRequest req) {
        validateRequest(req);

        int topK = (req.topK() == null || req.topK() <= 0) ? 5 : req.topK();

        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.span.kind", "chain");
        attrs.put("gen_ai.prompt.0.content", req.question());
        attrs.put("langsmith.metadata.doc_id", req.docId());
        attrs.put("langsmith.metadata.top_k", topK);

        return traceHelper.run("ask-request", attrs, () -> {
            OrchestratorResult result = orchestratorService.handle(
                    req.question(),
                    req.docId(),
                    topK
            );

            log.info(
                    "Route={} Score={} JudgeScore={} Outcome={}",
                    result.routeUsed(),
                    result.bestScore(),
                    result.judge() != null ? result.judge().score() : null,
                    result.outcome()
            );

            traceHelper.addAttributes(buildResultAttributes(result));

            List<ChunkDto> chunks = result.chunks().stream()
                    .map(c -> new ChunkDto(c.id(), c.text()))
                    .toList();

            return new AskResponse(
                    result.answer(),
                    result.retrievedChunkIds(),
                    result.citedChunkIds(),
                    chunks,
                    result.usedChunks(),
                    result.bestScore(),
                    result.judge()
            );
        });
    }
    private Map<String, Object> buildResultAttributes(OrchestratorResult result) {
        Map<String, Object> attrs = new LinkedHashMap<>();

        attrs.put("langsmith.metadata.route_used", result.routeUsed());
        attrs.put("langsmith.metadata.reason", result.reason());
        attrs.put("langsmith.metadata.outcome", result.outcome());

        if (result.bestScore() != null) {
            attrs.put("langsmith.metadata.best_score", result.bestScore());
        }

        if (result.judge() != null) {
            attrs.put("langsmith.metadata.judge.score", result.judge().score());
            attrs.put("langsmith.metadata.judge.grounded", result.judge().grounded());
            attrs.put("langsmith.metadata.judge.correct", result.judge().correct());
            attrs.put("langsmith.metadata.judge.complete", result.judge().complete());
        }

        return attrs;
    }
    @GetMapping("/orchestrate")
    public OrchestratorResult orchestrate(
            @RequestParam String docId,
            @RequestParam String question,
            @RequestParam(defaultValue = "5") int k
    ) {
        return orchestratorService.handle(question, docId, k);
    }

    private void validateRequest(AskRequest req) {
        if (req == null || req.question() == null || req.question().isBlank()) {
            throw new BadRequestException("question is required");
        }

        if (req.docId() == null || req.docId().isBlank()) {
            throw new BadRequestException("docId is required");
        }
    }

    @ResponseStatus(HttpStatus.BAD_REQUEST)
    static class BadRequestException extends RuntimeException {
        public BadRequestException(String message) {
            super(message);
        }
    }
}