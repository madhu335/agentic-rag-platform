package com.example.airagassistant.eval;

import com.example.airagassistant.rag.RagRetriever;
import com.example.airagassistant.rag.RetrievalMode;
import com.example.airagassistant.rag.SearchHit;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

@Service
public class RetrievalEvaluator {

    private final RagRetriever ragRetriever;

    public RetrievalEvaluator(RagRetriever ragRetriever) {
        this.ragRetriever = ragRetriever;
    }

    public List<RetrievalEvalResult> evaluate(List<EvalCase> cases, String docId, int k) {
        List<RetrievalEvalResult> results = new ArrayList<>();

        for (EvalCase evalCase : cases) {
            List<SearchHit> hits = ragRetriever.retrieve(docId, evalCase.question(), k);

            List<String> returnedChunkIds = hits.stream()
                    .map(hit -> hit.record().id())
                    .collect(Collectors.toList());

            Set<String> expected = new HashSet<>(safe(evalCase.expectedChunkIds()));

            long relevantReturned = returnedChunkIds.stream()
                    .filter(expected::contains)
                    .count();

            double precisionAtK = returnedChunkIds.isEmpty() ? 0.0 : (double) relevantReturned / returnedChunkIds.size();
            double recallAtK = expected.isEmpty() ? 0.0 : (double) relevantReturned / expected.size();
            boolean hitAtK = relevantReturned > 0;

            results.add(new RetrievalEvalResult(
                    evalCase.id(),
                    evalCase.question(),
                    returnedChunkIds,
                    evalCase.expectedChunkIds(),
                    precisionAtK,
                    recallAtK,
                    hitAtK
            ));
        }

        return results;
    }

    private List<String> safe(List<String> values) {
        return values == null ? List.of() : values;
    }

    public List<RetrievalEvalResult> evaluate(
            List<EvalCase> cases,
            String docId,
            int k,
            RetrievalMode mode
    ) {
        List<RetrievalEvalResult> results = new ArrayList<>();

        for (EvalCase evalCase : cases) {
            List<SearchHit> hits = ragRetriever.retrieve(docId, evalCase.question(), k, mode);

            List<String> returnedChunkIds = hits.stream()
                    .map(hit -> hit.record().id())
                    .toList();

            Set<String> expected = new HashSet<>(safe(evalCase.expectedChunkIds()));

            long relevantReturned = returnedChunkIds.stream()
                    .filter(expected::contains)
                    .count();

            double precisionAtK = returnedChunkIds.isEmpty() ? 0.0 : (double) relevantReturned / returnedChunkIds.size();
            double recallAtK = expected.isEmpty() ? 0.0 : (double) relevantReturned / expected.size();
            boolean hitAtK = relevantReturned > 0;

            results.add(new RetrievalEvalResult(
                    evalCase.id(),
                    evalCase.question(),
                    returnedChunkIds,
                    evalCase.expectedChunkIds(),
                    precisionAtK,
                    recallAtK,
                    hitAtK
            ));
        }

        return results;
    }
}