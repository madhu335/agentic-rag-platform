package com.example.airagassistant.agentic.divergence;

import com.example.airagassistant.agentic.dto.ResearchResult;
import com.example.airagassistant.agentic.state.AgentSessionState;
import com.example.airagassistant.agentic.state.AgentStateStore;
import com.example.airagassistant.agentic.tools.ResearchTool;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class FallbackService {

    private final ResearchTool researchTool;
    private final AgentStateStore stateStore;

    public void retryWithBetterQuery(String sessionId, String originalQuery) {
        AgentSessionState state = stateStore.get(sessionId);
        String improvedQuery = originalQuery + " with more details and examples";

        ResearchResult retryResult = researchTool.research(state.docId(), improvedQuery, 10);

        AgentSessionState updated = state.withResearch(new AgentSessionState.ResearchSnapshot(
                retryResult.answer(),
                retryResult.citations(),
                retryResult.confidenceScore(),
                retryResult.judge(),
                retryResult.chunks()
        ));

        stateStore.save(updated);
    }
}