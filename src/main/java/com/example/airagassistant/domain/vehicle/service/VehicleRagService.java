package com.example.airagassistant.domain.vehicle.service;

import com.example.airagassistant.LlmClient;
import com.example.airagassistant.rag.EmbeddingClient;
import com.example.airagassistant.rag.PgVectorStore;
import com.example.airagassistant.rag.RagAnswerService;
import com.example.airagassistant.rag.RetrievalMode;
import com.example.airagassistant.rag.SearchHit;
import com.example.airagassistant.rag.TritonRerankerClient;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@Slf4j
@Service
@RequiredArgsConstructor
public class VehicleRagService {

    public static final String VEHICLE_DOC_PREFIX = "";
    private static final int DEFAULT_TOP_K = 5;

    private final RagAnswerService ragAnswerService;
    private final EmbeddingClient embeddingClient;
    private final PgVectorStore vectorStore;
    private final LlmClient llmClient;
    private final VehicleAttributeRepository attributeRepository;
    private final TritonRerankerClient rerankerClient;

    /** Max chunks one vehicle may occupy in a fleet result before others get a slot. */
    @Value("${vehicle.search.max-chunks-per-vehicle:3}")
    private int maxChunksPerVehicle = 3;

    /**
     * Collapse near-duplicate variants (same make + model, different trim/wheels/year)
     * so one model can't fill a fleet result. See {@link VariantCollapser}.
     * Set false to A/B against the uncollapsed ranking.
     */
    @Value("${vehicle.search.collapse-variants:true}")
    private boolean collapseVariants = true;

    /**
     * Minimum best-match scores before fleet search returns anything. Both must be weak to reject.
     * These depend on the embedding model's score scale: bge-base put real matches at 0.5-0.8
     * (old hard-coded 0.50); Nemotron 3.5 Embed puts them at ~0.27-0.48 and nonsense at ~0.26.
     * Nonsense close to the floor is left to the relevance gate (isRelevantAfterRetrieval).
     */
    @Value("${vehicle.search.min-vector-score:0.25}")
    private double minVectorScore = 0.25;

    @Value("${vehicle.search.min-keyword-score:0.2}")
    private double minKeywordScore = 0.2;

    /**
     * Relevance gate after retrieval: "reranker" (default), "llm" (previous LLM prompt) or "off".
     * The reranker gate scores the top hits with the cross-encoder (cross_reranker on Triton) and
     * passes if the best (query, chunk) score >= min-rerank-score. A cross-encoder reads query and
     * chunk together, so its score is a direct relevance signal; cosine similarity is not.
     * The LLM gate answered relevant=true for every eval query, nonsense included.
     */
    @Value("${vehicle.search.relevance-gate:reranker}")
    private String relevanceGate = "reranker";

    @Value("${vehicle.search.min-rerank-score:0.0}")
    private double minRerankScore = 0.0;

    @Value("${vehicle.search.rerank-gate-top-n:5}")
    private int rerankGateTopN = 5;

    /** Variants are dropped after retrieval, so fetch a wider pool when collapsing. */
    private int candidatePool(int topK) {
        return collapseVariants ? Math.max(topK * 6, 40) : Math.max(topK * 3, 12);
    }

    // ─── Single-vehicle query ──────────────────────────────────────────────

    public RagAnswerService.RagResult askVehicle(String vehicleId, String question) {
        return askVehicle(vehicleId, question, DEFAULT_TOP_K, RetrievalMode.HYBRID);
    }

    public RagAnswerService.RagResult askVehicle(String vehicleId, String question,
                                                 int topK, RetrievalMode mode) {
        if (vehicleId == null || vehicleId.isBlank()) {
            throw new IllegalArgumentException("vehicleId cannot be blank");
        }
        if (question == null || question.isBlank()) {
            throw new IllegalArgumentException("question cannot be blank");
        }

        log.debug("Vehicle RAG ask — vehicleId='{}' question='{}'", vehicleId, question);

        return ragAnswerService.answerWithMode("vehicle", vehicleId, question, topK, mode);
    }

    /**
     * Relevance gate. Its only job is to keep nonsense out (gibberish or off-topic queries such
     * as the "xyzabc vehicle" edge case). Ranking quality is handled downstream by RRF, the
     * numeric boost and the judge, so this gate errs towards true to protect recall.
     *
     * The previous prompt mixed "true only if it clearly matches" with "true if it could
     * plausibly help, even partially". A weak model ignored the strict rule; a stronger,
     * more literal model (Qwen3-30B-A3B) applied it and rejected valid fleet questions such as
     * "which vehicle has the best resale value".
     */
    private boolean isRelevantAfterRetrieval(String question, List<SearchHit> hits) {
        if (hits == null || hits.isEmpty()) {
            return false;
        }
        return switch (relevanceGate == null ? "reranker" : relevanceGate.trim().toLowerCase()) {
            case "off" -> true;
            case "llm" -> isRelevantByLlm(question, hits);
            default -> isRelevantByReranker(question, hits);
        };
    }

    /**
     * Cross-encoder gate: best rerank score over the top N hits must reach min-rerank-score.
     * Always logs the best score so the threshold can be calibrated from eval runs.
     * Fails open (returns true) if the reranker is unavailable, to protect recall.
     */
    private boolean isRelevantByReranker(String question, List<SearchHit> hits) {
        List<SearchHit> top = hits.stream().limit(Math.max(1, rerankGateTopN)).toList();
        try {
            List<String> docs = top.stream().map(h -> h.record().text()).toList();
            List<Double> scores = rerankerClient.score(question, docs);
            int bestIdx = 0;
            for (int i = 1; i < scores.size(); i++) {
                if (scores.get(i) > scores.get(bestIdx)) bestIdx = i;
            }
            double best = scores.isEmpty() ? 0.0 : scores.get(bestIdx);
            boolean pass = best >= minRerankScore;
            log.info("Rerank gate: best={} threshold={} pass={} topChunk={} q={}",
                    String.format("%.4f", best), minRerankScore, pass,
                    scores.isEmpty() ? "-" : top.get(bestIdx).record().id(), question);
            return pass;
        } catch (Exception e) {
            log.warn("Rerank gate failed; allowing results. query='{}'", question, e);
            return true;
        }
    }

    /** Previous gate: asks the LLM whether the query is gibberish/off-topic. Kept as an option. */
    private boolean isRelevantByLlm(String question, List<SearchHit> hits) {

        String context = hits.stream()
                .limit(5)
                .map(h -> h.record().id() + "\n" + h.record().text())
                .reduce("", (a, b) -> a + "\n\n---\n\n" + b);

        String prompt = """
            You are a gate for a vehicle search system. Decide only whether the user query is a
            real question that this vehicle context could help with.

            Return {"relevant": false} ONLY if:
            - the query is gibberish or random characters, or
            - the query is about something unrelated to vehicles (for example recipes or weather).

            Otherwise return {"relevant": true}, even if the context only partly answers the query
            or covers only some of the vehicles it asks about.

            Return ONLY the JSON, with no explanation.

            User query:
            %s

            Retrieved context:
            %s
            """.formatted(question, context);

        try {
            String response = llmClient.answer(prompt, List.of());

            log.info("Vehicle retrieval relevance judge response: {}", response);

            String normalized = response.toLowerCase()
                    .replace(" ", "")
                    .replace("\n", "")
                    .replace("\r", "");

            return normalized.contains("\"relevant\":true");

        } catch (Exception e) {
            log.warn("Vehicle retrieval relevance judge failed; allowing results. query='{}'", question, e);
            return true; // fail-open to protect recall
        }
    }

    // ─── Cross-vehicle (fleet) query ───────────────────────────────────────

    public List<SearchHit> searchAllVehicles(String question, int topK, String docType) {
        if (question == null || question.isBlank()) {
            throw new IllegalArgumentException("question cannot be blank");
        }
        if (docType == null || docType.isBlank() || "vehicle".equals(docType)) {
            Optional<List<SearchHit>> structured = structuredSearch(question, topK);
            if (structured.isPresent()) {
                return structured.get();
            }
        }

        int candidateK = candidatePool(topK);
        List<Double> queryVector = embeddingClient.embed(question);
        List<SearchHit> vectorHits = vectorStore.searchAllWithScores(queryVector, candidateK, docType);
        List<SearchHit> keywordHits = vectorStore.keywordSearchAll(question, candidateK, docType);
        double bestVector = vectorHits.isEmpty() ? 0 : vectorHits.get(0).score();
        double bestKeyword = keywordHits.isEmpty() ? 0 : keywordHits.get(0).score();

        boolean weakVector = bestVector < minVectorScore;
        boolean weakKeyword = bestKeyword < minKeywordScore;

        if (weakVector && weakKeyword) {
            log.warn("Rejecting low-quality retrieval: vector={} keyword={} q={}",
                    bestVector, bestKeyword, question);
            return List.of();
        }
        List<SearchHit> fused = fuseRRF(vectorHits, keywordHits, candidateK);
        // One model family (46 Rivian R1T variants) must not crowd others out of top-K
        if (collapseVariants) {
            fused = VariantCollapser.collapse(fused, topK);
        }
        // One vehicle with many matching chunks must not crowd others out of top-K
        fused = diversifyByVehicle(fused, maxChunksPerVehicle, topK);

        if (!isRelevantAfterRetrieval(question, fused)) {
            log.warn("Vehicle chunks rejected by relevance judge — query='{}'", question);
            return List.of();
        }

        return fused.stream().limit(topK).toList();
    }

    /**
     * Searches across ALL ingested vehicles.
     *
     * Strategy:
     *  1. If the question is a threshold or superlative over a known attribute
     *     ("over 500 horsepower", "most torque"), answer it with SQL over
     *     vehicle_attributes. See {@link #structuredSearch}.
     *  2. Otherwise run vector + keyword search, RRF-fuse, diversify by vehicle,
     *     and pass the result through the relevance gate.
     */
    public List<SearchHit> searchAllVehicles(String question, int topK) {
        if (question == null || question.isBlank()) {
            throw new IllegalArgumentException("question cannot be blank");
        }
        Optional<List<SearchHit>> structured = structuredSearch(question, topK);
        if (structured.isPresent()) {
            return structured.get();
        }

        int candidateK = candidatePool(topK);
        List<Double> queryVector = embeddingClient.embed(question);
        List<SearchHit> vectorHits = vectorStore.searchAllWithScores(queryVector, candidateK);
        List<SearchHit> keywordHits = vectorStore.keywordSearchAll(question, candidateK);
        double bestVector = vectorHits.isEmpty() ? 0 : vectorHits.get(0).score();
        double bestKeyword = keywordHits.isEmpty() ? 0 : keywordHits.get(0).score();

        boolean weakVector = bestVector < minVectorScore;
        boolean weakKeyword = bestKeyword < minKeywordScore;

        if (weakVector && weakKeyword) {
            log.warn("Rejecting low-quality retrieval: vector={} keyword={} q={}",
                    bestVector, bestKeyword, question);
            return List.of();
        }
        List<SearchHit> fused = fuseRRF(vectorHits, keywordHits, candidateK);
        // One model family (46 Rivian R1T variants) must not crowd others out of top-K
        if (collapseVariants) {
            fused = VariantCollapser.collapse(fused, topK);
        }
        // One vehicle with many matching chunks must not crowd others out of top-K
        fused = diversifyByVehicle(fused, maxChunksPerVehicle, topK);

        if (!isRelevantAfterRetrieval(question, fused)) {
            log.warn("Vehicle chunks rejected by relevance judge — query='{}'", question);
            return List.of();
        }

        return fused.stream().limit(topK).toList();
    }

    // ─── Structured (attribute) search ─────────────────────────────────────

    /**
     * Answers threshold and superlative questions with SQL instead of text similarity.
     *
     * Text retrieval matches tokens, not values: at fleet scale "over 500 horsepower"
     * returned the Mercedes S500 and Lexus LS 500 because "500" appears in the trim.
     * When {@link StructuredVehicleQuery} recognises the question, vehicles are filtered
     * and ranked on vehicle_attributes and one relevant chunk per vehicle is returned.
     *
     * Returns empty (caller falls back to hybrid search) when the question is not
     * structured, or when no vehicle has the attribute at all. That keeps a parser false
     * positive or a data gap from ever turning a good text answer into nothing.
     * The relevance gate is skipped here: a matched attribute query is a valid fleet question.
     */
    Optional<List<SearchHit>> structuredSearch(String question, int topK) {
        Optional<StructuredVehicleQuery> parsed = StructuredVehicleQuery.parse(question);
        if (parsed.isEmpty()) {
            return Optional.empty();
        }
        try {
            // Over-fetch so collapsing variants still leaves topK distinct models
            int fetch = collapseVariants ? topK * 5 : topK;
            List<SearchHit> hits = attributeRepository.search(parsed.get(), fetch);
            if (collapseVariants) {
                hits = VariantCollapser.collapse(hits, topK);
            }
            hits = hits.stream().limit(topK).toList();
            if (hits.isEmpty()) {
                log.info("Structured query matched no vehicles, falling back to hybrid — q='{}'", question);
                return Optional.empty();
            }
            return Optional.of(hits);
        } catch (Exception e) {
            log.warn("Structured query failed, falling back to hybrid — q='{}'", question, e);
            return Optional.empty();
        }
    }

    // ─── Per-vehicle diversification ──────────────────────────────────────

    /**
     * Cap-then-backfill diversification (a GROUP BY vehicle with a per-group limit before LIMIT).
     *
     * 1. Walk hits in their current order (RRF + numeric boost), keeping at most
     *    {@code maxPerVehicle} chunks per vehicle.
     * 2. If that yields fewer than {@code topK}, refill from the skipped chunks in order,
     *    so the result is never smaller than before.
     *
     * Without this, a vehicle with many similar chunks (e.g. 8 of 13 slots for the BMW M3)
     * pushes relevant vehicles out of top-K: golden-set entry S4-005 ("AWD vehicles").
     * A cap above 1 keeps single-vehicle questions asked fleet-wide answerable.
     */
    static List<SearchHit> diversifyByVehicle(List<SearchHit> hits, int maxPerVehicle, int topK) {
        if (hits == null || hits.isEmpty() || maxPerVehicle <= 0) {
            return hits;
        }

        Map<String, Integer> perVehicle = new HashMap<>();
        List<SearchHit> kept = new ArrayList<>();
        List<SearchHit> skipped = new ArrayList<>();

        for (SearchHit hit : hits) {
            String vehicle = vehicleIdOf(hit.record().id());
            int count = perVehicle.getOrDefault(vehicle, 0);
            if (count < maxPerVehicle) {
                kept.add(hit);
                perVehicle.put(vehicle, count + 1);
            } else {
                skipped.add(hit);
            }
        }

        for (SearchHit hit : skipped) {
            if (kept.size() >= topK) {
                break;
            }
            kept.add(hit);
        }

        return kept;
    }

    /** "bmw-m3-2025-competition:21" -> "bmw-m3-2025-competition" */
    static String vehicleIdOf(String chunkId) {
        if (chunkId == null) {
            return "";
        }
        int idx = chunkId.lastIndexOf(':');
        return idx >= 0 ? chunkId.substring(0, idx) : chunkId;
    }

    // ─── RRF fusion ───────────────────────────────────────────────────────

    private List<SearchHit> fuseRRF(List<SearchHit> vectorHits,
                                    List<SearchHit> keywordHits,
                                    int topK) {
        java.util.Map<String, Double> rrfScores = new java.util.LinkedHashMap<>();
        java.util.Map<String, SearchHit> source = new java.util.LinkedHashMap<>();
        int K = 10;

        for (int i = 0; i < vectorHits.size(); i++) {
            String id = vectorHits.get(i).record().id();
            source.putIfAbsent(id, vectorHits.get(i));
            rrfScores.merge(id, 1.0 / (K + i + 1), Double::sum);
        }
        for (int i = 0; i < keywordHits.size(); i++) {
            String id = keywordHits.get(i).record().id();
            source.putIfAbsent(id, keywordHits.get(i));
            rrfScores.merge(id, 1.0 / (K + i + 1), Double::sum);
        }

        return rrfScores.entrySet().stream()
                .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                .limit(topK)
                .map(e -> new SearchHit(source.get(e.getKey()).record(), e.getValue()))
                .toList();
    }
}
