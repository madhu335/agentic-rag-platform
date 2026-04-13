package com.example.airagassistant.eval;

import com.example.airagassistant.rag.RagAnswerService;
import com.example.airagassistant.rag.RagAnswerService.RagResult;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class AnswerEvaluator {

    private final RagAnswerService ragAnswerService;

    public AnswerEvaluator(RagAnswerService ragAnswerService) {
        this.ragAnswerService = ragAnswerService;
    }

    public List<AnswerEvalResult> evaluate(List<EvalCase> cases, String docId, int topK) {
        List<AnswerEvalResult> results = new ArrayList<>();

        for (EvalCase evalCase : cases) {

            RagResult result = ragAnswerService.answer(docId, evalCase.question(), topK);
            String answer = result.answer();

            List<String> missing = new ArrayList<>();
            for (String phrase : safe(evalCase.expectedAnswerContains())) {
                if (answer == null || !answer.toLowerCase().contains(phrase.toLowerCase())) {
                    missing.add(phrase);
                }
            }

            boolean pass = missing.isEmpty();

            results.add(new AnswerEvalResult(
                    evalCase.id(),
                    evalCase.question(),
                    answer,
                    safe(evalCase.expectedAnswerContains()),
                    missing,
                    pass
            ));
        }

        return results;
    }

    private List<String> safe(List<String> values) {
        return values == null ? List.of() : values;
    }
}