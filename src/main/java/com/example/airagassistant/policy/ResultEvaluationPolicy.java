package com.example.airagassistant.policy;

import com.example.airagassistant.agentic.dto.ResearchResult;
import com.example.airagassistant.router.OrchestratorResult;

public interface ResultEvaluationPolicy {

    EvaluationDecision evaluateOrchestratorResult(OrchestratorResult result);

    EvaluationDecision evaluateResearchResult(ResearchResult result);
}