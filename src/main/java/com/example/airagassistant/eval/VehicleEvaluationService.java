package com.example.airagassistant.eval;

import com.example.airagassistant.domain.vehicle.service.VehicleRagService;
import com.example.airagassistant.rag.EmbeddingClient;
import com.example.airagassistant.rag.PgVectorStore;
import com.example.airagassistant.rag.SearchHit;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class VehicleEvaluationService {

    private final VehicleRagService vehicleRagService;
    private final PgVectorStore     vectorStore;
    private final EmbeddingClient   embeddingClient;
    private final JdbcTemplate      jdbcTemplate;
    private final ObjectMapper      objectMapper;

    private static final double RECALL_TARGET = 0.85;
    private static final int    DEFAULT_TOP_K = 5;

    // ─── Run ──────────────────────────────────────────────────────────────────

    public EvalReport runGoldenSet() {
        List<GoldenEntry> entries = loadGoldenSet();
        log.info("Running vehicle evaluation — {} golden set entries", entries.size());

        List<EntryResult> results = entries.stream()
                .map(this::evaluate)
                .toList();

        return buildReport(results);
    }

    // ─── Evaluate one entry ───────────────────────────────────────────────────

    private EntryResult evaluate(GoldenEntry entry) {
        try {
            List<SearchHit> hits = vehicleRagService.searchAllVehicles(entry.query(), DEFAULT_TOP_K);

            List<String> returnedVehicleIds = hits.stream()
                    .map(h -> extractVehicleId(h.record().id()))
                    .distinct()
                    .toList();

            List<String> returnedChunkIds = hits.stream()
                    .map(h -> h.record().id())
                    .toList();

            Set<String> expected   = new HashSet<>(entry.expectedVehicleIds());
            Set<String> returned   = new HashSet<>(returnedVehicleIds);
            Set<String> acceptable = new HashSet<>(entry.acceptableVehicleIds());

            long correctCount    = returned.stream().filter(expected::contains).count();
            long acceptableCount = returned.stream()
                    .filter(v -> acceptable.contains(v) && !expected.contains(v)).count();

            double recall    = expected.isEmpty() ? 1.0 : (double) correctCount / expected.size();
            double precision = returned.isEmpty() ? 1.0 : (double) correctCount / returned.size();
            boolean passed   = correctCount >= entry.minimumRecallCount();

            // Edge case: nonsense query must return nothing meaningful
            if (entry.expectedVehicleIds().isEmpty() && entry.minimumRecallCount() == 0) {
                passed = returnedVehicleIds.isEmpty()
                        || hits.stream().allMatch(h -> h.score() < 0.02);
            }

            // Diagnose failures
            FailureAnalysis analysis = null;
            if (!passed) {
                Set<String> missedVehicleIds = new HashSet<>(expected);
                missedVehicleIds.removeAll(returned);
                analysis = diagnose(entry, missedVehicleIds, returnedChunkIds, hits);
            }

            return new EntryResult(entry, returnedVehicleIds, returnedChunkIds,
                    recall, precision, correctCount, acceptableCount, passed, analysis);

        } catch (Exception e) {
            log.error("Evaluation failed for entry {}: {}", entry.entryId(), e.getMessage());
            FailureAnalysis analysis = new FailureAnalysis(
                    "EVALUATION_EXCEPTION",
                    "Exception during evaluation: " + e.getMessage(),
                    List.of(),
                    List.of(),
                    List.of("Fix the exception before re-running")
            );
            return new EntryResult(entry, List.of(), List.of(),
                    0.0, 0.0, 0, 0, false, analysis);
        }
    }

    // ─── Failure analysis ─────────────────────────────────────────────────────

    /**
     * Inspects missed vehicles and returns a human-readable diagnosis explaining
     * exactly WHY retrieval failed — missing chunks, wrong vocabulary, low scores, etc.
     *
     * Failure reasons:
     *  MISSING_CHUNKS        — expected vehicle has no chunks at all in document_chunks
     *  MISSING_CHUNK_TYPE    — vehicle ingested but specific chunk type not present
     *                          (e.g. no ownership_cost chunk — ingested via simple /ingest)
     *  LOW_SCORE             — chunk exists but scores too low to make top-K
     *  VOCABULARY_MISMATCH   — chunk exists and scores ok but wrong vocabulary
     *  OUTRANKED             — chunk exists but less relevant vehicles scored higher
     *  EDGE_CASE_LEAK        — nonsense query returned results it should not have
     */
    private FailureAnalysis diagnose(GoldenEntry entry,
                                     Set<String> missedVehicleIds,
                                     List<String> returnedChunkIds,
                                     List<SearchHit> hits) {
        List<String> reasons     = new ArrayList<>();
        List<String> chunkGaps   = new ArrayList<>();
        List<String> suggestions = new ArrayList<>();

        // Edge case: nonsense query leaked results
        if (entry.expectedVehicleIds().isEmpty() && !returnedChunkIds.isEmpty()) {
            return new FailureAnalysis(
                    "EDGE_CASE_LEAK",
                    "Nonsense query '" + entry.query() + "' returned " + returnedChunkIds.size()
                            + " results instead of empty. All queries match something via vector similarity.",
                    chunkGaps,
                    returnedChunkIds,
                    List.of(
                            "Add a minimum score threshold for fleet search results",
                            "Consider a query classifier that detects nonsense/out-of-domain queries",
                            "Return empty if bestScore < 0.05"
                    )
            );
        }

        for (String vehicleId : missedVehicleIds) {
            // Check 1: Does the vehicle have ANY chunks in the DB?
            List<String> existingChunks = getExistingChunkIds(vehicleId);

            if (existingChunks.isEmpty()) {
                reasons.add(vehicleId + ": NOT INGESTED — no chunks found in document_chunks");
                chunkGaps.add(vehicleId + ": 0 chunks");
                suggestions.add("Ingest " + vehicleId + " via POST /vehicles/ingest or /vehicles/ingest/rich");
                continue;
            }

            // Check 2: Does it have the chunk type matching the query category?
            String expectedChunkType = categoryToChunkType(entry.category());
            boolean hasChunkType = existingChunks.stream()
                    .anyMatch(id -> chunkTypeMatches(id, expectedChunkType, vehicleId));

            if (!hasChunkType) {
                reasons.add(vehicleId + ": MISSING CHUNK TYPE '" + expectedChunkType
                        + "' — vehicle only has chunks: " + summarizeChunks(existingChunks, vehicleId));
                chunkGaps.add(vehicleId + " missing: " + expectedChunkType + " chunk");
                suggestions.add("Re-ingest " + vehicleId + " via /vehicles/ingest/rich "
                        + "with a fully populated RichVehicleRecord including "
                        + expectedChunkType + " data");
                continue;
            }

            // Check 3: Did the chunk exist but score too low to make top-K?
            double bestScoreForVehicle = hits.stream()
                    .filter(h -> extractVehicleId(h.record().id()).equals(vehicleId))
                    .mapToDouble(SearchHit::score)
                    .max()
                    .orElse(0.0);

            if (bestScoreForVehicle == 0.0) {
                // Chunk type exists but didn't appear in results at all
                double directScore = scoreChunkDirectly(vehicleId, entry.query());
                if (directScore < 0.04) {
                    reasons.add(vehicleId + ": VOCABULARY MISMATCH — chunk exists but scores "
                            + String.format("%.4f", directScore)
                            + " for this query. Chunk vocabulary doesn't match query terms.");
                    suggestions.add("Rewrite " + expectedChunkType + " chunk for " + vehicleId
                            + " to include terms: " + extractKeyTerms(entry.query()));
                } else {
                    reasons.add(vehicleId + ": OUTRANKED — chunk scored "
                            + String.format("%.4f", directScore)
                            + " but other vehicles scored higher and pushed it out of top-" + DEFAULT_TOP_K);
                    suggestions.add("Increase topK to " + (DEFAULT_TOP_K + 3)
                            + " for '" + entry.category() + "' category queries");
                }
            } else {
                reasons.add(vehicleId + ": LOW SCORE — best score was "
                        + String.format("%.4f", bestScoreForVehicle)
                        + " — below retrieval threshold");
                suggestions.add("Add more narrative context to " + expectedChunkType
                        + " chunk for " + vehicleId);
            }
        }

        String primaryReason = classifyPrimaryReason(reasons);

        return new FailureAnalysis(
                primaryReason,
                String.join(" | ", reasons),
                chunkGaps,
                returnedChunkIds,
                suggestions
        );
    }

    // ─── Diagnosis helpers ────────────────────────────────────────────────────

    private List<String> getExistingChunkIds(String vehicleId) {
        try {
            return jdbcTemplate.queryForList(
                    "SELECT doc_id || ':' || chunk_index FROM document_chunks WHERE doc_id = ? ORDER BY chunk_index",
                    String.class, vehicleId
            );
        } catch (Exception e) {
            return List.of();
        }
    }

    private boolean chunkTypeMatches(String chunkId, String expectedType, String vehicleId) {
        // chunk_index ranges by type (from VehicleChunkBuilder):
        // :1=identity, :2=performance, :3=ownership_cost, :4=rankings
        // :5=safety, :6=features_trims, :7=reviews, :10-14=maintenance, :20+=recalls
        // For simple ingestion there is only :1
        if (expectedType == null) return true;
        String indexStr = chunkId.replace(vehicleId + ":", "");
        try {
            int idx = Integer.parseInt(indexStr);
            return switch (expectedType) {
                case "performance"    -> idx == 2;
                case "ownership_cost" -> idx == 3;
                case "rankings"       -> idx == 4;
                case "safety"         -> idx == 5;
                case "features_trims" -> idx == 6;
                case "reviews"        -> idx == 7;
                case "maintenance"    -> idx >= 10 && idx <= 19;
                case "recall"         -> idx >= 20;
                case "identity"       -> idx == 1;
                default               -> true;
            };
        } catch (NumberFormatException e) {
            return false;
        }
    }

    private String summarizeChunks(List<String> chunkIds, String vehicleId) {
        return chunkIds.stream()
                .map(id -> {
                    String idx = id.replace(vehicleId + ":", "");
                    try {
                        int i = Integer.parseInt(idx);
                        return i + "(" + indexToTypeName(i) + ")";
                    } catch (NumberFormatException e) { return idx; }
                })
                .collect(Collectors.joining(", "));
    }

    private String indexToTypeName(int idx) {
        if (idx == 1)           return "identity";
        if (idx == 2)           return "performance";
        if (idx == 3)           return "ownership_cost";
        if (idx == 4)           return "rankings";
        if (idx == 5)           return "safety";
        if (idx == 6)           return "features";
        if (idx == 7)           return "reviews";
        if (idx >= 10 && idx <= 19) return "maintenance";
        if (idx >= 20)          return "recall";
        return "unknown";
    }

    private double scoreChunkDirectly(String vehicleId, String query) {
        try {
            List<Double> qVec = embeddingClient.embed(query);
            List<SearchHit> hits = vectorStore.searchWithScores(vehicleId, qVec, 1);
            return hits.isEmpty() ? 0.0 : hits.get(0).score();
        } catch (Exception e) {
            return 0.0;
        }
    }

    private String categoryToChunkType(String category) {
        return switch (category) {
            case "PERFORMANCE"    -> "performance";
            case "OWNERSHIP_COST" -> "ownership_cost";
            case "MAINTENANCE"    -> "maintenance";
            case "SAFETY"         -> "safety";
            case "RANKINGS"       -> "rankings";
            case "FEATURES"       -> "features_trims";
            case "CROSS_CHUNK"    -> null;  // spans multiple — don't check one type
            case "FLEET_FILTER"   -> null;
            default               -> null;
        };
    }

    private String extractKeyTerms(String query) {
        return Arrays.stream(query.toLowerCase().split("\\s+"))
                .filter(t -> t.length() > 3)
                .filter(t -> !Set.of("which", "what", "does", "have", "with",
                        "that", "this", "from", "will", "more", "most").contains(t))
                .collect(Collectors.joining(", "));
    }

    private String classifyPrimaryReason(List<String> reasons) {
        if (reasons.isEmpty()) return "UNKNOWN";
        String combined = String.join(" ", reasons).toUpperCase();
        if (combined.contains("NOT INGESTED"))        return "MISSING_CHUNKS";
        if (combined.contains("MISSING CHUNK TYPE"))  return "MISSING_CHUNK_TYPE";
        if (combined.contains("VOCABULARY MISMATCH")) return "VOCABULARY_MISMATCH";
        if (combined.contains("OUTRANKED"))           return "OUTRANKED";
        if (combined.contains("LOW SCORE"))           return "LOW_SCORE";
        if (combined.contains("EDGE_CASE_LEAK"))      return "EDGE_CASE_LEAK";
        return "UNKNOWN";
    }

    // ─── Report builder ───────────────────────────────────────────────────────

    private EvalReport buildReport(List<EntryResult> results) {
        double overallRecall    = avg(results, EntryResult::recall);
        double overallPrecision = avg(results, EntryResult::precision);
        long   totalPassed      = results.stream().filter(EntryResult::passed).count();
        boolean meetsTarget     = overallRecall >= RECALL_TARGET;

        Map<String, CategoryStats> byCategory   = groupStats(results, r -> r.entry().category());
        Map<String, CategoryStats> byDifficulty = groupStats(results, r -> r.entry().difficulty());
        Map<String, CategoryStats> bySource     = groupStats(results, r -> r.entry().source());

        List<FailedEntry> failed = results.stream()
                .filter(r -> !r.passed())
                .map(r -> new FailedEntry(
                        r.entry().entryId(),
                        r.entry().query(),
                        r.entry().expectedVehicleIds(),
                        r.returnedVehicleIds(),
                        r.recall(),
                        r.analysis()
                ))
                .toList();

        log.info("══════════════════════════════════════════════");
        log.info("Vehicle RAG Evaluation Report");
        log.info("  Overall recall:    {:.1f}%  (target {:.0f}%)",
                overallRecall * 100, RECALL_TARGET * 100);
        log.info("  Overall precision: {:.1f}%", overallPrecision * 100);
        log.info("  Passed: {}/{}", totalPassed, results.size());
        log.info("  Meets target: {}", meetsTarget ? "YES" : "NO");
        failed.forEach(f -> {
            log.warn("  FAILED [{}] {} → {}",
                    f.entryId(), f.query(),
                    f.analysis() != null ? f.analysis().primaryReason() : "unknown");
            if (f.analysis() != null) {
                f.analysis().suggestions().forEach(s -> log.warn("    → {}", s));
            }
        });
        log.info("══════════════════════════════════════════════");

        return new EvalReport(overallRecall, overallPrecision,
                (int) totalPassed, results.size(),
                meetsTarget, byCategory, byDifficulty, bySource, failed);
    }

    private Map<String, CategoryStats> groupStats(
            List<EntryResult> results,
            java.util.function.Function<EntryResult, String> keyFn) {
        return results.stream().collect(Collectors.groupingBy(keyFn,
                Collectors.collectingAndThen(Collectors.toList(), g -> new CategoryStats(
                        avg(g, EntryResult::recall),
                        avg(g, EntryResult::precision),
                        (int) g.stream().filter(EntryResult::passed).count(),
                        g.size()
                ))));
    }

    private double avg(List<EntryResult> list,
                       java.util.function.ToDoubleFunction<EntryResult> fn) {
        return list.stream().mapToDouble(fn).average().orElse(0.0);
    }

    // ─── Golden set loader ────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private List<GoldenEntry> loadGoldenSet() {
        try (InputStream is = getClass()
                .getResourceAsStream("/eval/vehicle_golden_set.json")) {
            if (is == null) throw new IllegalStateException(
                    "vehicle_golden_set.json not found in /resources/eval/");
            List<Map<String, Object>> raw = objectMapper.readValue(is, List.class);
            return raw.stream().map(this::toEntry).toList();
        } catch (Exception e) {
            throw new IllegalStateException("Failed to load golden set: " + e.getMessage(), e);
        }
    }

    @SuppressWarnings("unchecked")
    private GoldenEntry toEntry(Map<String, Object> m) {
        return new GoldenEntry(
                str(m, "entryId"), str(m, "query"),
                listOf(m, "expectedVehicleIds"), listOf(m, "acceptableVehicleIds"),
                str(m, "category"), str(m, "difficulty"), str(m, "source"),
                ((Number) m.getOrDefault("minimumRecallCount", 1)).intValue(),
                listOf(m, "expectedChunkIds"), (String) m.get("notes")
        );
    }

    private String str(Map<String, Object> m, String key) {
        return (String) m.getOrDefault(key, "");
    }

    @SuppressWarnings("unchecked")
    private List<String> listOf(Map<String, Object> m, String key) {
        Object v = m.get(key);
        return v == null ? List.of() : (List<String>) v;
    }

    private String extractVehicleId(String chunkId) {
        int idx = chunkId.lastIndexOf(':');
        return idx > 0 ? chunkId.substring(0, idx) : chunkId;
    }

    // ─── Records ──────────────────────────────────────────────────────────────

    public record GoldenEntry(
            String entryId, String query,
            List<String> expectedVehicleIds, List<String> acceptableVehicleIds,
            String category, String difficulty, String source,
            int minimumRecallCount, List<String> expectedChunkIds, String notes
    ) {}

    public record EntryResult(
            GoldenEntry entry, List<String> returnedVehicleIds,
            List<String> returnedChunkIds, double recall, double precision,
            long correctCount, long acceptableCount,
            boolean passed, FailureAnalysis analysis
    ) {}

    public record FailureAnalysis(
            String primaryReason,
            String detail,
            List<String> chunkGaps,
            List<String> returnedChunkIds,
            List<String> suggestions
    ) {}

    public record CategoryStats(
            double recall, double precision, int passed, int total
    ) {}

    public record FailedEntry(
            String entryId, String query,
            List<String> expected, List<String> returned,
            double recall, FailureAnalysis analysis
    ) {}

    public record EvalReport(
            double overallRecall, double overallPrecision,
            int totalPassed, int totalEntries, boolean meetsTarget,
            Map<String, CategoryStats> byCategory,
            Map<String, CategoryStats> byDifficulty,
            Map<String, CategoryStats> bySource,
            List<FailedEntry> failedEntries
    ) {}
}