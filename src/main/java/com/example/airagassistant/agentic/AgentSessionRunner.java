package com.example.airagassistant.agentic;

import com.example.airagassistant.agentic.state.AgentSessionState;
import com.example.airagassistant.agentic.state.AgentStateStore;
import com.example.airagassistant.agentic.state.StepHistoryEntry;
import com.example.airagassistant.rag.RagAnswerService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.Supplier;

@Component
@RequiredArgsConstructor
public class AgentSessionRunner {

    private final AgentStateStore stateStore;

    public <T> T runWithSession(
            String userRequest,
            String docId,
            Supplier<T> action
    ) {
        String sessionId = UUID.randomUUID().toString();

        AgentSessionState state = AgentSessionState.empty(sessionId, userRequest, docId);
        stateStore.save(state);

        try {
            state = addHistory(state, "research", Map.of(), "STARTED", null, null, "Starting RAG");
            stateStore.save(state);

            T result = action.get();

            state = addHistory(state, "research", Map.of(), "SUCCESS", "Completed successfully", null, "RAG success");
            stateStore.save(state);

            return result;

        } catch (Exception e) {
            state = addHistory(
                    state,
                    "research",
                    Map.of(),
                    "FAILED",
                    null,
                    e.getMessage(),
                    "RAG failed"
            );
            stateStore.save(state);
            throw e;
        }
    }

    private AgentSessionState addHistory(
            AgentSessionState state,
            String step,
            Map<String, Object> args,
            String status,
            String resultSummary,
            String error,
            String fallbackSummary
    ) {
        List<StepHistoryEntry> newHistory = new ArrayList<>(state.history());

        newHistory.add(new StepHistoryEntry(
                step,
                args == null ? Map.of() : args,
                status,
                resultSummary != null ? resultSummary : fallbackSummary,
                error,
                Instant.now()
        ));

        return state.withHistory(newHistory);
    }

    public RagAnswerService.RagResult runRagWithSession(
            String userRequest,
            String docId,
            Supplier<RagAnswerService.RagResult> action
    ) {
        String sessionId = UUID.randomUUID().toString();

        AgentSessionState state =
                AgentSessionState.empty(sessionId, userRequest, docId);

        stateStore.save(state);

        try {
            state = addHistory(state, "research", Map.of(), "STARTED", "Starting RAG", null, null);
            stateStore.save(state);

            RagAnswerService.RagResult result = action.get();

            // 🔥 save actual RAG output into session
            state = state.withResearch(new AgentSessionState.ResearchSnapshot(
                    result.answer(),
                    result.citedChunkIds(),
                    result.bestScore(),
                    null,
                    result.retrievedChunkIds()
            ));

            state = addHistory(state, "research", Map.of(), "SUCCESS", "RAG success", null, null);
            stateStore.save(state);

            return result;

        } catch (Exception e) {
            state = addHistory(state, "research", Map.of(), "FAILED", null, e.getMessage(), "RAG failed");
            stateStore.save(state);
            throw e;
        }
    }
}