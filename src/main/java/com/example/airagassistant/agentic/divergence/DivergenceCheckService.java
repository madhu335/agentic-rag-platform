package com.example.airagassistant.agentic.divergence;

import com.example.airagassistant.agentic.dto.ResearchResult;
import org.springframework.stereotype.Service;

@Service
public class DivergenceCheckService {

    private static final double MIN_CONFIDENCE = 0.08;   // lowered a lot
    private static final double MIN_JUDGE_SCORE = 0.70;

    public boolean isDiverged(ResearchResult result) {
        if (result == null) {
            return true;
        }

        boolean judgeFailed =
                result.judge() == null
                        || !result.judge().grounded()
                        || result.judge().score() < MIN_JUDGE_SCORE;

        boolean veryLowConfidence =
                result.confidenceScore() == null
                        || result.confidenceScore() < MIN_CONFIDENCE;

        return judgeFailed || veryLowConfidence;
    }
}