package com.example.airagassistant.domain.vehicle.service;

import com.example.airagassistant.rag.EmbeddingClient;
import com.example.airagassistant.rag.PgVectorStore;
import com.example.airagassistant.rag.RagAnswerService;
import com.example.airagassistant.rag.RetrievalMode;
import com.example.airagassistant.rag.SearchHit;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Slf4j
@Service
@RequiredArgsConstructor
public class VehicleRagService {

    public static final String VEHICLE_DOC_PREFIX = "";
    private static final int DEFAULT_TOP_K = 5;

    private final RagAnswerService ragAnswerService;
    private final EmbeddingClient embeddingClient;
    private final PgVectorStore vectorStore;

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

    // ─── Cross-vehicle (fleet) query ───────────────────────────────────────

    /**
     * Searches across ALL ingested vehicles.
     *
     * Strategy:
     *  1. Run vector + keyword search, RRF-fuse results.
     *  2. If the question contains a numeric threshold ("over 500 horsepower",
     *     "less than $40,000"), apply a post-filter that re-sorts: chunks
     *     satisfying the condition come first, others fall to the end.
     *     This corrects the embedding model's weakness with numeric comparisons.
     */
    public List<SearchHit> searchAllVehicles(String question, int topK) {
        if (question == null || question.isBlank()) {
            throw new IllegalArgumentException("question cannot be blank");
        }

        // Fetch a wider pool so numeric post-filter has enough candidates
        int candidateK = Math.max(topK * 3, 12);
        List<Double> queryVector = embeddingClient.embed(question);
        List<SearchHit> vectorHits  = vectorStore.searchAllWithScores(queryVector, candidateK);
        List<SearchHit> keywordHits = vectorStore.keywordSearchAll(question, candidateK);
        List<SearchHit> fused       = fuseRRF(vectorHits, keywordHits, candidateK);

        // Apply numeric boost/penalty if the question has a threshold expression
        NumericFilter filter = NumericFilter.parse(question);
        if (filter != null) {
            log.debug("Fleet search numeric filter — field={} op={} value={}",
                    filter.field(), filter.operator(), filter.value());
            fused = applyNumericBoost(fused, filter);
        }

        return fused.stream().limit(topK).toList();
    }

    // ─── Numeric post-filter ───────────────────────────────────────────────

    private List<SearchHit> applyNumericBoost(List<SearchHit> hits, NumericFilter filter) {
        return hits.stream()
                .sorted((a, b) -> {
                    boolean aMatch = filter.matches(a.record().text());
                    boolean bMatch = filter.matches(b.record().text());

                    if (aMatch && bMatch) {
                        // Both satisfy the condition — sort by actual field value so
                        // e.g. 835hp ranks above 670hp above 503hp for "over 500hp"
                        double aVal = filter.extractValue(a.record().text());
                        double bVal = filter.extractValue(b.record().text());
                        boolean ascending = filter.operator().equals("LT") || filter.operator().equals("LTE");
                        return ascending
                                ? Double.compare(aVal, bVal)   // cheapest first for "under $X"
                                : Double.compare(bVal, aVal);  // highest first for "over X hp"
                    }

                    if (!aMatch && !bMatch) return Double.compare(b.score(), a.score()); // keep RRF order
                    return aMatch ? -1 : 1; // matching tier always above non-matching
                })
                .toList();
    }

    // ─── RRF fusion ───────────────────────────────────────────────────────

    private List<SearchHit> fuseRRF(List<SearchHit> vectorHits,
                                    List<SearchHit> keywordHits,
                                    int topK) {
        java.util.Map<String, Double>   rrfScores = new java.util.LinkedHashMap<>();
        java.util.Map<String, SearchHit> source   = new java.util.LinkedHashMap<>();
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

    // ─── NumericFilter ────────────────────────────────────────────────────

    /**
     * Parses threshold expressions from natural language:
     *   "over 500 horsepower"  → Horsepower > 500
     *   "at least 300 hp"      → Horsepower >= 300
     *   "under $40,000"        → MSRP < 40000
     *   "less than 30 mpg"     → Fuel Economy < 30
     *
     * Returns null if no recognisable threshold is found.
     */
    record NumericFilter(String field, String operator, double value) {

        boolean matches(String chunkText) {
            if (chunkText == null) return false;
            for (String line : chunkText.split("\n")) {
                if (!line.toLowerCase().contains(field.toLowerCase())) continue;
                Matcher m = Pattern.compile("(\\d+(?:\\.\\d+)?)").matcher(line);
                if (!m.find()) continue;
                double lineVal = Double.parseDouble(m.group(1));
                return switch (operator) {
                    case "GT"  -> lineVal >  value;
                    case "GTE" -> lineVal >= value;
                    case "LT"  -> lineVal <  value;
                    case "LTE" -> lineVal <= value;
                    default    -> false;
                };
            }
            return false;
        }

        /** Returns the numeric value of this field from the chunk text, or -1 if not found. */
        double extractValue(String chunkText) {
            if (chunkText == null) return -1;
            for (String line : chunkText.split("\n")) {
                if (!line.toLowerCase().contains(field.toLowerCase())) continue;
                Matcher m = Pattern.compile("(\\d+(?:\\.\\d+)?)").matcher(line);
                if (m.find()) return Double.parseDouble(m.group(1));
            }
            return -1;
        }

        static NumericFilter parse(String question) {
            String q = question.toLowerCase();

            String op;
            if (q.matches(".*(\\bover\\b|more than|above|greater than).*"))       op = "GT";
            else if (q.matches(".*(at least|minimum|or more).*"))                  op = "GTE";
            else if (q.matches(".*(\\bunder\\b|less than|below|cheaper than).*")) op = "LT";
            else if (q.matches(".*(at most|maximum|or less).*"))                   op = "LTE";
            else return null;

            Matcher numMatcher = Pattern.compile("(\\d{1,3}(?:,\\d{3})*)").matcher(q);
            if (!numMatcher.find()) return null;
            double value = Double.parseDouble(numMatcher.group(1).replace(",", ""));

            String field;
            if (q.matches(".*(horsepower|\\bhp\\b|\\bpower\\b).*"))           field = "Horsepower";
            else if (q.matches(".*(torque|lb-ft).*"))                          field = "Torque";
            else if (q.matches(".*(mpg|fuel economy|mileage|efficient).*"))    field = "Fuel Economy";
            else if (q.matches(".*(price|msrp|cost|\\$|dollar|budget).*"))     field = "MSRP";
            else return null;

            return new NumericFilter(field, op, value);
        }
    }
}