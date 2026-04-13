package com.example.airagassistant.eval;

import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.example.airagassistant.rag.RetrievalMode;

@Service
public class EvaluationService {

    private final GoldenSetLoader goldenSetLoader;
    private final RetrievalEvaluator retrievalEvaluator;
    private final AnswerEvaluator answerEvaluator;

    public EvaluationService(GoldenSetLoader goldenSetLoader,
                             RetrievalEvaluator retrievalEvaluator,
                             AnswerEvaluator answerEvaluator) {
        this.goldenSetLoader = goldenSetLoader;
        this.retrievalEvaluator = retrievalEvaluator;
        this.answerEvaluator = answerEvaluator;
    }

    public List<RetrievalEvalResult> runRetrievalEval(String docId, int k) {
        return retrievalEvaluator.evaluate(goldenSetLoader.load(), docId, k);
    }

    public List<AnswerEvalResult> runAnswerEval(String docId, int k) {
        return answerEvaluator.evaluate(goldenSetLoader.load(), docId, k);
    }

    public EvalSummary summarize(String docId, int k) {
        List<RetrievalEvalResult> retrievalResults = runRetrievalEval(docId, k);
        List<AnswerEvalResult> answerResults = runAnswerEval(docId, k);

        int total = Math.max(retrievalResults.size(), answerResults.size());

        double avgPrecision = retrievalResults.stream()
                .mapToDouble(RetrievalEvalResult::precisionAtK)
                .average()
                .orElse(0.0);

        double avgRecall = retrievalResults.stream()
                .mapToDouble(RetrievalEvalResult::recallAtK)
                .average()
                .orElse(0.0);

        double hitRate = retrievalResults.stream()
                .filter(RetrievalEvalResult::hitAtK)
                .count() / (double) Math.max(1, retrievalResults.size());

        double answerPassRate = answerResults.stream()
                .filter(AnswerEvalResult::pass)
                .count() / (double) Math.max(1, answerResults.size());

        return new EvalSummary(total, avgPrecision, avgRecall, hitRate, answerPassRate);
    }

    public Map<RetrievalMode, EvalSummary> compareModes(String docId, int k) {
        Map<RetrievalMode, EvalSummary> results = new HashMap<>();

        for (RetrievalMode mode : RetrievalMode.values()) {
            List<RetrievalEvalResult> retrievalResults =
                    retrievalEvaluator.evaluate(goldenSetLoader.load(), docId, k, mode);

            double avgPrecision = retrievalResults.stream()
                    .mapToDouble(RetrievalEvalResult::precisionAtK)
                    .average()
                    .orElse(0.0);

            double avgRecall = retrievalResults.stream()
                    .mapToDouble(RetrievalEvalResult::recallAtK)
                    .average()
                    .orElse(0.0);

            double hitRate = retrievalResults.stream()
                    .filter(RetrievalEvalResult::hitAtK)
                    .count() / (double) Math.max(1, retrievalResults.size());

            results.put(mode, new EvalSummary(
                    retrievalResults.size(),
                    avgPrecision,
                    avgRecall,
                    hitRate,
                    0.0 // answer eval later
            ));
        }

        return results;
    }
}