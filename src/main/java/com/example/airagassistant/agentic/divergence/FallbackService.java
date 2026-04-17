package com.example.airagassistant.agentic.divergence;

import com.example.airagassistant.agentic.dto.ResearchResult;
import com.example.airagassistant.agentic.state.AgentSessionState;
import com.example.airagassistant.agentic.state.AgentStateStore;
import com.example.airagassistant.agentic.tools.ResearchTool;
import com.example.airagassistant.judge.JudgeResult;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class FallbackService {

    private final ResearchTool researchTool;
    private final AgentStateStore stateStore;
    private final QueryRefinementService queryRefinementService;

    public void retryWithBetterQuery(String sessionId, String originalQuery) {
        AgentSessionState state = stateStore.get(sessionId);

        String previousAnswer = null;
        List<String> retrievedChunks = null;
        JudgeResult judge = null;

        if (state.research() != null) {
            previousAnswer = state.research().summary();
            retrievedChunks = state.research().chunks();
            judge = state.research().judge();
        }

        String improvedQuery = queryRefinementService.refineQuery(
                originalQuery,
                previousAnswer,
                retrievedChunks,
                judge
        );

        ResearchResult retryResult = researchTool.research(state.docId(), improvedQuery, 10);

        AgentSessionState updated = state.withResearch(new AgentSessionState.ResearchSnapshot(
                retryResult.answer(),
                retryResult.citations(),
                retryResult.retrievalScore(),
                retryResult.judge(),
                retryResult.chunks()
        ));

        stateStore.save(updated);
    }
}