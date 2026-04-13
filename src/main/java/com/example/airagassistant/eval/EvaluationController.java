package com.example.airagassistant.eval;

import com.example.airagassistant.rag.*;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/eval")
@RequiredArgsConstructor
public class EvaluationController {

    private final EvaluationService evaluationService;
    private final PgVectorStore vectorStore;
    private final ReRankService reRankService;
    private final EmbeddingClient embeddingClient;
    private final VehicleEvaluationService vehicleEvaluationService;


    @GetMapping("/retrieval")
    public List<RetrievalEvalResult> retrievalEval(
            @RequestParam String docId,
            @RequestParam(defaultValue = "5") int k
    ) {
        return evaluationService.runRetrievalEval(docId, k);
    }

    @GetMapping("/answer")
    public List<AnswerEvalResult> answerEval(
            @RequestParam String docId,
            @RequestParam(defaultValue = "5") int k
    ) {
        return evaluationService.runAnswerEval(docId, k);
    }

    @GetMapping("/report")
    public EvalSummary report(
            @RequestParam String docId,
            @RequestParam(defaultValue = "5") int k
    ) {
        return evaluationService.summarize(docId, k);
    }

    @GetMapping("/compare")
    public Map<RetrievalMode, EvalSummary> compare(
            @RequestParam String docId,
            @RequestParam(defaultValue = "5") int k
    ) {
        return evaluationService.compareModes(docId, k);
    }

    @GetMapping("/debug")
    public DebugRetrievalResult debug(
            @RequestParam String docId,
            @RequestParam String question,
            @RequestParam(defaultValue = "5") int k
    ) {
        return new DebugRetrievalResult(
                question,
                vectorIds(docId, question, k),
                bm25Ids(docId, question, k),
                hybridIds(docId, question, k),
                hybridRerankIds(docId, question, k)
        );
    }
    @GetMapping("/vehicles/recall/report")
    public VehicleEvaluationService.EvalReport runVehicleEval() {
        return vehicleEvaluationService.runGoldenSet();
    }

    private List<SearchHit> vectorSearch(String docId, String question, int k) {
        List<Double> qVec = embeddingClient.embed(question);
        return vectorStore.searchWithScores(docId, qVec, k);
    }

    private List<SearchHit> bm25Search(String docId, String question, int k) {
        return vectorStore.keywordSearch(docId, question, k);
    }

    private List<SearchHit> hybridSearch(String docId, String question, int k) {
        List<Double> qVec = embeddingClient.embed(question);
        return vectorStore.hybridSearch(docId, qVec, question, k);
    }

    private List<SearchHit> hybridRerankSearch(String docId, String question, int k) {
        List<SearchHit> hits = hybridSearch(docId, question, k);
        return reRankService.rerank(question, hits);
    }
    private List<String> vectorIds(String docId, String question, int k) {
        return vectorSearch(docId, question, k).stream()
                .map(hit -> hit.record().id())
                .toList();
    }

    private List<String> bm25Ids(String docId, String question, int k) {
        return bm25Search(docId, question, k).stream()
                .map(hit -> hit.record().id())
                .toList();
    }

    private List<String> hybridIds(String docId, String question, int k) {
        return hybridSearch(docId, question, k).stream()
                .map(hit -> hit.record().id())
                .toList();
    }

    private List<String> hybridRerankIds(String docId, String question, int k) {
        return hybridRerankSearch(docId, question, k).stream()
                .map(hit -> hit.record().id())
                .toList();
    }
}