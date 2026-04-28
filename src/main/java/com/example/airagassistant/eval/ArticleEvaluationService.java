package com.example.airagassistant.eval;

import com.example.airagassistant.domain.article.service.ArticleRagService;
import com.example.airagassistant.domain.article.service.ArticleRagService.ArticleSearchHit;
import com.example.airagassistant.rag.PgVectorStore;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Article-side counterpart to {@link VehicleEvaluationService}.
 *
 * Runs the article golden set (resources/eval/article_golden_set.json)
 * through {@link ArticleRagService#searchAllArticles(String, int)} and
 * reports recall + precision per category, difficulty, and source, plus
 * per-failure diagnosis (MISSING_CHUNKS / MISSING_ARTICLE / LOW_SCORE /
 * OUTRANKED / VOCABULARY_MISMATCH / EDGE_CASE_LEAK).
 *
 * Expose via EvaluationController:
 *   GET /api/eval/articles/recall/report
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ArticleEvaluationService {

    private final ArticleRagService articleRagService;
    private final PgVectorStore     vectorStore;
    private final JdbcTemplate      jdbcTemplate;
    private final ObjectMapper      objectMapper;

    private static final double RECALL_TARGET = 0.85;
    private static final int    DEFAULT_TOP_K = 5;

    // ─── Run ──────────────────────────────────────────────────────────────────

    public EvalReport runGoldenSet() {
        List<GoldenEntry> entries = loadGoldenSet();
        log.info("Running article evaluation — {} golden set entries", entries.size());

        List<EntryResult> results = entries.stream()
                .map(this::evaluate)
                .toList();

        return buildReport(results);
    }

    // ─── Evaluate one entry ───────────────────────────────────────────────────

    private EntryResult evaluate(GoldenEntry entry) {
        try {
            List<ArticleSearchHit> hits = articleRagService.searchAllArticles(
                    entry.query(), DEFAULT_TOP_K);

            List<String> returnedArticleIds = hits.stream()
                    .map(ArticleSearchHit::articleId)
                    .distinct()
                    .toList();

            List<String> returnedChunkIds = hits.stream()
                    .map(ArticleSearchHit::chunkId)
                    .toList();

            Set<String> expected   = new HashSet<>(entry.expectedArticleIds());
            Set<String> returned   = new HashSet<>(returnedArticleIds);
            Set<String> acceptable = new HashSet<>(entry.acceptableArticleIds());

            long correctCount    = returned.stream().filter(expected::contains).count();
            long acceptableCount = returned.stream()
                    .filter(a -> acceptable.contains(a) && !expected.contains(a)).count();

            double recall    = expected.isEmpty() ? 1.0 : (double) correctCount / expected.size();
            double precision = returned.isEmpty() ? 1.0 : (double) correctCount / returned.size();
            boolean passed   = correctCount >= entry.minimumRecallCount();

            // Edge case: nonsense query must return nothing meaningful.
            // Article search reports scores on a 0..1 scale; be conservative.
            if (entry.expectedArticleIds().isEmpty() && entry.minimumRecallCount() == 0) {
                passed = returnedArticleIds.isEmpty()
                        || hits.stream().allMatch(h -> h.score() < 0.05);
            }

            // Diagnose failures
            FailureAnalysis analysis = null;
            if (!passed) {
                Set<String> missedArticleIds = new HashSet<>(expected);
                missedArticleIds.removeAll(returned);
                analysis = diagnose(entry, missedArticleIds, returnedChunkIds, hits);
            }

            return new EntryResult(entry, returnedArticleIds, returnedChunkIds,
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
     * Article-specific failure reasons:
     *
     *  MISSING_ARTICLE       — articleId has no chunks in document_chunks at all
     *                          (probably not ingested)
     *  MISSING_CHUNK_TYPE    — article ingested but the specific chunk type
     *                          expected by this category isn't present
     *                          (e.g. no pros_cons chunk for a SEMANTIC query)
     *  LOW_SCORE             — chunk exists but scored below retrieval threshold
     *  VOCABULARY_MISMATCH   — chunk exists, scored ok, but wrong vocabulary
     *                          for this query phrasing
     *  OUTRANKED             — chunk exists but other articles scored higher
     *                          and pushed it out of top-K
     *  EDGE_CASE_LEAK        — nonsense query returned results it should not have
     */
    private FailureAnalysis diagnose(GoldenEntry entry,
                                     Set<String> missedArticleIds,
                                     List<String> returnedChunkIds,
                                     List<ArticleSearchHit> hits) {
        List<String> reasons     = new ArrayList<>();
        List<String> chunkGaps   = new ArrayList<>();
        List<String> suggestions = new ArrayList<>();

        // Edge case: nonsense query leaked results
        if (entry.expectedArticleIds().isEmpty() && !returnedChunkIds.isEmpty()) {
            return new FailureAnalysis(
                    "EDGE_CASE_LEAK",
                    "Nonsense query '" + entry.query() + "' returned "
                            + returnedChunkIds.size()
                            + " results instead of empty. All queries match something via vector similarity.",
                    chunkGaps,
                    returnedChunkIds,
                    List.of(
                            "Add a weak-retrieval floor to searchAllArticles "
                                    + "(reject if bestVector < 0.50 && bestKeyword < 0.2)",
                            "Consider a retrieval-time relevance judge like VehicleRagService",
                            "Return empty if bestScore < 0.05"
                    )
            );
        }

        for (String articleId : missedArticleIds) {
            List<String> existingChunks = getExistingChunkIds(articleId);

            if (existingChunks.isEmpty()) {
                reasons.add(articleId + ": NOT INGESTED — no chunks found in document_chunks");
                chunkGaps.add(articleId + ": 0 chunks");
                suggestions.add("Ingest " + articleId + " via POST /articles/ingest");
                continue;
            }

            String expectedChunkType = categoryToChunkType(entry.category());
            if (expectedChunkType != null) {
                boolean hasChunkType = existingChunks.stream()
                        .anyMatch(id -> chunkTypeMatches(id, expectedChunkType));
                if (!hasChunkType) {
                    reasons.add(articleId + ": MISSING CHUNK TYPE '" + expectedChunkType
                            + "' — article has chunks at indexes: "
                            + summarizeChunks(existingChunks, articleId));
                    chunkGaps.add(articleId + " missing: " + expectedChunkType);
                    suggestions.add("Re-ingest " + articleId
                            + " ensuring the CmsArticle payload includes the "
                            + expectedChunkType + " fields (rating / pros / cons / sections / body)");
                    continue;
                }
            }

            double bestScoreForArticle = hits.stream()
                    .filter(h -> h.articleId().equals(articleId))
                    .mapToDouble(ArticleSearchHit::score)
                    .max()
                    .orElse(0.0);

            if (bestScoreForArticle == 0.0) {
                reasons.add(articleId + ": LOW SCORE — no chunks scored above retrieval threshold "
                        + "for query '" + entry.query() + "'. Likely VOCABULARY MISMATCH: "
                        + "chunk prose doesn't use key terms from the query ["
                        + extractKeyTerms(entry.query()) + "]");
                suggestions.add("Rewrite " + articleId
                        + " chunks to include the query vocabulary, "
                        + "or enrich the vehicle anchor so this article "
                        + "surfaces for vehicle-centric queries");
            } else {
                reasons.add(articleId + ": OUTRANKED — best chunk scored "
                        + String.format("%.4f", bestScoreForArticle)
                        + " but other articles scored higher and pushed it out of top-"
                        + DEFAULT_TOP_K);
                suggestions.add("Increase topK to " + (DEFAULT_TOP_K + 3)
                        + " for '" + entry.category() + "' category queries, "
                        + "or add a reranker pass after searchAllArticles");
            }
        }

        String primaryReason = classifyPrimaryReason(reasons);
        return new FailureAnalysis(
                primaryReason,
                String.join(" | ", reasons),
                chunkGaps,
                returnedChunkIds,
                suggestions.stream().distinct().toList()
        );
    }

    // ─── DB helpers ───────────────────────────────────────────────────────────

    /**
     * document_chunks stores article chunks with id = "{articleId}:{chunkIndex}"
     * and doc_type = "article". We filter on doc_type to avoid matching
     * same-prefixed vehicle chunks.
     */
    private List<String> getExistingChunkIds(String articleId) {
        try {
            return jdbcTemplate.queryForList(
                    "SELECT id FROM document_chunks "
                            + "WHERE doc_type = 'article' AND id LIKE ?",
                    String.class,
                    articleId + ":%"
            );
        } catch (Exception e) {
            log.warn("Could not query chunk existence for article {}: {}",
                    articleId, e.getMessage());
            return List.of();
        }
    }

    /**
     * Chunk ID prefix mapping mirrors ArticleChunkBuilder:
     *   :1  identity_verdict
     *   :2  ratings
     *   :3  pros_cons
     *   :4  vehicle_references
     *   :10+ section_*
     *   :50+ body_window
     */
    private boolean chunkTypeMatches(String chunkId, String expectedType) {
        int colon = chunkId.lastIndexOf(':');
        if (colon < 0) return false;
        int idx;
        try {
            idx = Integer.parseInt(chunkId.substring(colon + 1));
        } catch (NumberFormatException nfe) {
            return false;
        }
        return switch (expectedType) {
            case "identity_verdict"    -> idx == 1;
            case "ratings"             -> idx == 2;
            case "pros_cons"           -> idx == 3;
            case "vehicle_references"  -> idx == 4;
            case "section"             -> idx >= 10 && idx < 50;
            case "body_window"         -> idx >= 50;
            default                    -> false;
        };
    }

    private String summarizeChunks(List<String> chunkIds, String articleId) {
        return chunkIds.stream()
                .map(id -> id.substring(articleId.length() + 1))
                .sorted()
                .limit(10)
                .collect(Collectors.joining(", "));
    }

    /**
     * Maps golden-set category to the expected chunk type the retrieval
     * should have surfaced. null = don't pin a specific chunk type (the
     * answer could legitimately come from several).
     */
    private String categoryToChunkType(String category) {
        return switch (category) {
            case "DIRECT_LOOKUP"    -> "identity_verdict";
            case "COMPARISON"       -> "identity_verdict";
            case "RANKING"          -> "identity_verdict";
            case "VEHICLE_MENTION"  -> "vehicle_references";
            case "SEMANTIC"         -> null;  // could be pros_cons OR section OR body
            case "CROSS_ARTICLE"    -> null;
            default                 -> null;
        };
    }

    private String extractKeyTerms(String query) {
        return Arrays.stream(query.toLowerCase().split("\\s+"))
                .filter(t -> t.length() > 3)
                .filter(t -> !Set.of("which", "what", "does", "have", "with",
                        "that", "this", "from", "will", "more", "most",
                        "about", "articles", "article").contains(t))
                .collect(Collectors.joining(", "));
    }

    private String classifyPrimaryReason(List<String> reasons) {
        if (reasons.isEmpty()) return "UNKNOWN";
        String combined = String.join(" ", reasons).toUpperCase();
        if (combined.contains("NOT INGESTED"))        return "MISSING_ARTICLE";
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
                        r.entry().expectedArticleIds(),
                        r.returnedArticleIds(),
                        r.recall(),
                        r.analysis()
                ))
                .toList();

        log.info("══════════════════════════════════════════════");
        log.info("Article RAG Evaluation Report");
        log.info(String.format("  Overall recall:    %.1f%%  (target %.0f%%)",
                overallRecall * 100, RECALL_TARGET * 100));
        log.info(String.format("  Overall precision: %.1f%%", overallPrecision * 100));
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
                .getResourceAsStream("/eval/article_golden_set.json")) {
            if (is == null) throw new IllegalStateException(
                    "article_golden_set.json not found in /resources/eval/");
            List<Map<String, Object>> raw = objectMapper.readValue(is, List.class);
            return raw.stream().map(this::toEntry).toList();
        } catch (Exception e) {
            throw new IllegalStateException("Failed to load article golden set: "
                    + e.getMessage(), e);
        }
    }

    private GoldenEntry toEntry(Map<String, Object> m) {
        return new GoldenEntry(
                str(m, "entryId"),
                str(m, "query"),
                listOf(m, "expectedArticleIds"),
                listOf(m, "acceptableArticleIds"),
                str(m, "category"),
                str(m, "difficulty"),
                str(m, "source"),
                ((Number) m.getOrDefault("minimumRecallCount", 1)).intValue(),
                listOf(m, "expectedChunkIds"),
                (String) m.get("notes")
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

    // ─── Records ──────────────────────────────────────────────────────────────

    public record GoldenEntry(
            String entryId, String query,
            List<String> expectedArticleIds, List<String> acceptableArticleIds,
            String category, String difficulty, String source,
            int minimumRecallCount, List<String> expectedChunkIds, String notes
    ) {}

    public record EntryResult(
            GoldenEntry entry, List<String> returnedArticleIds,
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
