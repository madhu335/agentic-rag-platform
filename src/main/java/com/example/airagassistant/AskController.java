package com.example.airagassistant;

import com.example.airagassistant.agentic.dto.AskResponse;
import com.example.airagassistant.agentic.dto.ChunkDto;
import com.example.airagassistant.rag.RagAnswerService;
import com.example.airagassistant.rag.RagAnswerService.StreamEvent;
import com.example.airagassistant.router.OrchestratorResult;
import com.example.airagassistant.router.OrchestratorService;
import com.example.airagassistant.trace.TraceHelper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@CrossOrigin(origins = "http://localhost:5173")
@RestController
@RequiredArgsConstructor
@Slf4j
public class AskController {

    private final OrchestratorService orchestratorService;
    private final TraceHelper traceHelper;
    private final RagAnswerService ragAnswerService;

    private final ExecutorService sseExecutor = Executors.newCachedThreadPool();

    public record AskRequest(String docType, String docId, String question, Integer topK) {}

    @PostMapping("/ask")
    public AskResponse ask(@RequestBody AskRequest req) {
        validateRequest(req);

        int topK = (req.topK() == null || req.topK() <= 0) ? 5 : req.topK();

        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.span.kind", "chain");
        attrs.put("gen_ai.prompt.0.content", req.question());
        attrs.put("langsmith.metadata.doc_type", req.docType());
        attrs.put("langsmith.metadata.doc_id", req.docId());
        attrs.put("langsmith.metadata.top_k", topK);

        return traceHelper.run("ask-request", attrs, () -> {
            OrchestratorResult result = orchestratorService.handle(
                    req.question(),
                    req.docType(),
                    req.docId(),
                    topK
            );

            log.info(
                    "Route={} DocType={} DocId={} Score={} JudgeScore={} Outcome={}",
                    result.routeUsed(),
                    req.docType(),
                    req.docId(),
                    result.retrievalScore(),
                    result.judge() != null ? result.judge().score() : null,
                    result.outcome()
            );

            traceHelper.addAttributes(buildResultAttributes(result));

            List<ChunkDto> chunks = result.chunks().stream()
                    .map(c -> new ChunkDto(c.id(), c.text()))
                    .toList();

            return new AskResponse(
                    result.answer(),
                    result.citedChunkIds(),
                    result.retrievedChunkIds(),
                    result.usedChunks(),
                    result.retrievalScore(),
                    result.judge(),
                    result.cards(),
                    chunks
            );
        });
    }

    private Map<String, Object> buildResultAttributes(OrchestratorResult result) {
        Map<String, Object> attrs = new LinkedHashMap<>();

        attrs.put("langsmith.metadata.route_used", result.routeUsed());
        attrs.put("langsmith.metadata.reason", result.reason());
        attrs.put("langsmith.metadata.outcome", result.outcome());

        if (result.retrievalScore() != null) {
            attrs.put("langsmith.metadata.best_score", result.retrievalScore());
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
            @RequestParam String docType,
            @RequestParam String docId,
            @RequestParam String question,
            @RequestParam(defaultValue = "5") int k
    ) {
        return orchestratorService.handle(question, docType, docId, k);
    }

    private void validateRequest(AskRequest req) {
        if (req == null) {
            throw new BadRequestException("request is required");
        }

        if (req.docType() == null || req.docType().isBlank()) {
            throw new BadRequestException("docType is required");
        }

        if (req.docId() == null || req.docId().isBlank()) {
            throw new BadRequestException("docId is required");
        }

        if (req.question() == null || req.question().isBlank()) {
            throw new BadRequestException("question is required");
        }
    }

    @ResponseStatus(HttpStatus.BAD_REQUEST)
    static class BadRequestException extends RuntimeException {
        public BadRequestException(String message) {
            super(message);
        }
    }

    @PostMapping(value = "/ask/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter askStream(@RequestBody AskRequest req) {
        validateRequest(req);
        int topK = (req.topK() == null || req.topK() <= 0) ? 5 : req.topK();

        SseEmitter emitter = new SseEmitter(0L);

        sseExecutor.submit(() -> {
            try {
                emitter.send(SseEmitter.event()
                        .name("start")
                        .data(mapOfNullable(
                                "docType", req.docType(),
                                "docId", req.docId(),
                                "question", req.question(),
                                "topK", topK
                        ), MediaType.APPLICATION_JSON));

                emitter.send(SseEmitter.event()
                        .name("status")
                        .data(mapOfNullable(
                                "stage", "retrieving",
                                "message", "Retrieving relevant chunks"
                        ), MediaType.APPLICATION_JSON));

                OrchestratorResult result = orchestratorService.handle(
                        req.question(),
                        req.docType(),
                        req.docId(),
                        topK
                );

                emitter.send(SseEmitter.event()
                        .name("sources")
                        .data(mapOfNullable(
                                "cards", result.cards() != null ? result.cards() : List.of()
                        ), MediaType.APPLICATION_JSON));

                emitter.send(SseEmitter.event()
                        .name("status")
                        .data(mapOfNullable(
                                "stage", "answering",
                                "message", "Generating answer"
                        ), MediaType.APPLICATION_JSON));

                if (result.answer() != null && !result.answer().isBlank()) {
                    emitter.send(SseEmitter.event()
                            .name("token")
                            .data(mapOfNullable("value", result.answer()), MediaType.APPLICATION_JSON));
                }

                emitter.send(SseEmitter.event()
                        .name("done")
                        .data(mapOfNullable(
                                "answer", result.answer() != null ? result.answer() : "",
                                "citedChunkIds", result.citedChunkIds() != null ? result.citedChunkIds() : List.of(),
                                "retrievedChunkIds", result.retrievedChunkIds() != null ? result.retrievedChunkIds() : List.of(),
                                "usedChunks", result.usedChunks(),
                                "retrievalScore", result.retrievalScore(),
                                "cards", result.cards() != null ? result.cards() : List.of()
                        ), MediaType.APPLICATION_JSON));

                emitter.complete();
            } catch (Exception e) {
                try {
                    emitter.send(SseEmitter.event()
                            .name("error")
                            .data(mapOfNullable(
                                    "message", e.getMessage() != null ? e.getMessage() : "Streaming failed"
                            ), MediaType.APPLICATION_JSON));
                } catch (IOException ignored) {
                }
                emitter.completeWithError(e);
            }
        });

        return emitter;
    }

    private void sendEvent(SseEmitter emitter, StreamEvent event) {
        try {
            emitter.send(SseEmitter.event()
                    .name(event.type())
                    .data(event.payload(), MediaType.APPLICATION_JSON));
        } catch (IOException e) {
            throw new RuntimeException("Failed to send SSE event", e);
        }
    }

    private Map<String, Object> mapOfNullable(Object... keyValues) {
        if (keyValues.length % 2 != 0) {
            throw new IllegalArgumentException("keyValues must contain an even number of entries");
        }

        Map<String, Object> map = new LinkedHashMap<>();
        for (int i = 0; i < keyValues.length; i += 2) {
            Object key = keyValues[i];
            if (!(key instanceof String s)) {
                throw new IllegalArgumentException("Map key must be a String at index " + i);
            }
            map.put(s, keyValues[i + 1]);
        }
        return map;
    }
}