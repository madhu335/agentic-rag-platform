package com.example.airagassistant.agentic.divergence;

import com.example.airagassistant.agentic.dto.ResearchResult;
import com.example.airagassistant.policy.EvaluationDecision;
import com.example.airagassistant.policy.ResultEvaluationPolicy;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class DivergenceCheckService {

    private final ResultEvaluationPolicy resultEvaluationPolicy;

    public boolean isDiverged(ResearchResult result) {
        EvaluationDecision decision = resultEvaluationPolicy.evaluateResearchResult(result);
        return decision.shouldRetry();
    }
}