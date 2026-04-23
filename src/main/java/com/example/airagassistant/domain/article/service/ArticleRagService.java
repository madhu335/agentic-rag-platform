package com.example.airagassistant.domain.article.service;

import com.example.airagassistant.rag.EmbeddingClient;
import com.example.airagassistant.rag.PgVectorStore;
import com.example.airagassistant.rag.RagAnswerService;
import com.example.airagassistant.rag.RetrievalMode;
import com.example.airagassistant.rag.SearchHit;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Article-specific RAG service.
 *
 * Three retrieval modes:
 *
 * 1. Single-article ask: articleId scoped
 *    → "What did MotorTrend say about the M3's performance?"
 *    → Searches only within that article's chunks
 *
 * 2. Vehicle-scoped search: find all articles about a vehicle
 *    → "Show me all M3 reviews"
 *    → Searches across all articles whose anchor mentions that vehicle
 *
 * 3. Cross-article search: semantic across all articles
 *    → "Which car did MotorTrend rate best for value in 2025?"
 *    → Searches all article chunks globally with rating-aware filtering
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ArticleRagService {

    private static final int DEFAULT_TOP_K = 5;

    private final RagAnswerService ragAnswerService;
    private final EmbeddingClient  embeddingClient;
    private final PgVectorStore    vectorStore;

    // ─── 1. Single-article ask ────────────────────────────────────────────────

    public RagAnswerService.RagResult askArticle(String articleId,
                                                 String question,
                                                 int topK,
                                                 RetrievalMode mode) {
        if (articleId == null || articleId.isBlank())
            throw new IllegalArgumentException("articleId cannot be blank");
        if (question == null || question.isBlank())
            throw new IllegalArgumentException("question cannot be blank");

        log.debug("Article ask — articleId='{}' question='{}'", articleId, question);
        return ragAnswerService.answerWithMode("article", articleId, question, topK, mode);
    }

    // ─── 2. Vehicle-scoped search ─────────────────────────────────────────────

    /**
     * Finds all articles that mention a specific vehicle.
     *
     * Works because the vehicle identity anchor is embedded in every chunk.
     * "BMW M3" appears in all chunks of any article that reviewed an M3.
     *
     * Returns grouped results: articleId → top matching chunks
     */
    public Map<String, List<SearchHit>> searchArticlesByVehicle(String vehicleQuery,
                                                                int topK) {
        if (vehicleQuery == null || vehicleQuery.isBlank())
            throw new IllegalArgumentException("vehicleQuery cannot be blank");

        log.debug("Vehicle-scoped article search — vehicle='{}'", vehicleQuery);

        List<Double> queryVector = embeddingClient.embed(vehicleQuery);
        List<SearchHit> vectorHits  = vectorStore.searchAllWithScores(queryVector, topK * 3, "article");
        List<SearchHit> keywordHits = vectorStore.keywordSearchAll(vehicleQuery, topK * 3, "article");
        List<SearchHit> fused       = fuseRRF(vectorHits, keywordHits, topK * 2);

        // Two filters combined in one stream:
        // 1. Article-level chunks only (:1-:4) — no body window noise
        // 2. Minimum score >= 0.12 — drops semantically weak matches
        //    (e.g. M3 review chunk scoring 0.09 for a Mercedes query)
        List<SearchHit> articleLevel = fused.stream()
                .filter(h -> isArticleLevelChunk(h.record().id()))
                .filter(h -> h.score() >= 0.12)
                .limit(topK)
                .toList();

        // Group by articleId so caller sees which articles matched
        return articleLevel.stream()
                .collect(Collectors.groupingBy(
                        h -> extractArticleId(h.record().id())
                ));
    }

    // ─── 3. Cross-article semantic search ────────────────────────────────────

    /**
     * Searches across ALL article chunks globally.
     *
     * Supports:
     *   - Opinion queries: "best sports sedan for daily driving"
     *   - Rating queries: "cars rated over 9 out of 10"
     *   - Comparison queries: "M3 vs 911 interior"
     *   - Section-specific: "most comfortable ride"
     */
    public List<ArticleSearchHit> searchAllArticles(String question, int topK) {
        if (question == null || question.isBlank())
            throw new IllegalArgumentException("question cannot be blank");

        log.debug("Cross-article search — question='{}'", question);

        int candidateK = Math.max(topK * 3, 15);
        List<Double> queryVector = embeddingClient.embed(question);
        List<SearchHit> vectorHits  = vectorStore.searchAllWithScores(queryVector, topK * 3, "article");
        List<SearchHit> keywordHits = vectorStore.keywordSearchAll(question, topK * 3, "article");
        List<SearchHit> fused       = fuseRRF(vectorHits, keywordHits, candidateK);

        // Apply rating filter if question contains a score threshold
        RatingFilter ratingFilter = RatingFilter.parse(question);
        if (ratingFilter != null) {
            log.debug("Rating filter detected — category={} op={} value={}",
                    ratingFilter.category(), ratingFilter.operator(), ratingFilter.value());
            fused = applyRatingFilter(fused, ratingFilter);
        }

        // Prefer article-level chunks (identity/ratings/pros_cons) over body windows
        // for cross-article queries — gives cleaner answers
        List<SearchHit> ranked = rankByChunkPreference(fused, question);

        return ranked.stream()
                .limit(topK)
                .map(h -> toArticleSearchHit(h, ranked.indexOf(h) + 1))
                .toList();
    }

    // ─── Rating filter ────────────────────────────────────────────────────────

    /**
     * Same pattern as VehicleRagService.NumericFilter but for article ratings.
     *
     * Handles:
     *   "cars rated over 9 out of 10"    → overall >= 9
     *   "best value rating"              → value category
     *   "top performance scores"         → performance category
     */
    private List<SearchHit> applyRatingFilter(List<SearchHit> hits, RatingFilter filter) {
        return hits.stream()
                .sorted((a, b) -> {
                    boolean aMatch = filter.matches(a.record().text());
                    boolean bMatch = filter.matches(b.record().text());
                    if (aMatch && bMatch) {
                        double aVal = filter.extractValue(a.record().text());
                        double bVal = filter.extractValue(b.record().text());
                        return Double.compare(bVal, aVal); // highest rating first
                    }
                    if (!aMatch && !bMatch) return Double.compare(b.score(), a.score());
                    return aMatch ? -1 : 1;
                })
                .toList();
    }

    /**
     * Prefer structured chunks (identity:1, ratings:2, pros_cons:3) over body windows (50+)
     * for cross-article queries — gives cleaner, more authoritative answers.
     *
     * Body window chunks are better for deep-dive single-article questions.
     */
    private List<SearchHit> rankByChunkPreference(List<SearchHit> hits, String question) {
        boolean isDeepDive = question.toLowerCase().matches(
                ".*(explain|describe|detail|tell me more|how does|what exactly).*");

        if (isDeepDive) return hits; // body windows are fine for deep-dive

        return hits.stream()
                .sorted((a, b) -> {
                    int aPriority = chunkPriority(a.record().id());
                    int bPriority = chunkPriority(b.record().id());
                    if (aPriority != bPriority) return Integer.compare(aPriority, bPriority);
                    return Double.compare(b.score(), a.score());
                })
                .toList();
    }

    private int chunkPriority(String chunkId) {
        int idx = extractChunkIndex(chunkId);
        if (idx == 1) return 1;  // identity + verdict — highest priority
        if (idx == 2) return 2;  // ratings
        if (idx == 3) return 3;  // pros/cons
        if (idx == 4) return 4;  // vehicle references
        if (idx >= 10 && idx < 50) return 5;  // sections
        return 6;                              // body windows — lowest priority
    }

    // ─── RRF fusion ───────────────────────────────────────────────────────────

    private List<SearchHit> fuseRRF(List<SearchHit> vectorHits,
                                    List<SearchHit> keywordHits,
                                    int topK) {
        java.util.Map<String, Double>    rrfScores = new java.util.LinkedHashMap<>();
        java.util.Map<String, SearchHit> source    = new java.util.LinkedHashMap<>();
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

    // ─── Helpers ──────────────────────────────────────────────────────────────

    private boolean isArticleLevelChunk(String chunkId) {
        int idx = extractChunkIndex(chunkId);
        return idx >= 1 && idx <= 4;
    }

    private String extractArticleId(String chunkId) {
        int idx = chunkId.lastIndexOf(':');
        return idx > 0 ? chunkId.substring(0, idx) : chunkId;
    }

    private int extractChunkIndex(String chunkId) {
        int idx = chunkId.lastIndexOf(':');
        if (idx < 0) return -1;
        try { return Integer.parseInt(chunkId.substring(idx + 1)); }
        catch (NumberFormatException e) { return -1; }
    }

    private ArticleSearchHit toArticleSearchHit(SearchHit hit, int rank) {
        return new ArticleSearchHit(
                hit.record().id(),
                extractArticleId(hit.record().id()),
                extractChunkIndex(hit.record().id()),
                rank,
                hit.score(),
                excerpt(hit.record().text(), 250)
        );
    }

    private String excerpt(String text, int maxLen) {
        if (text == null) return "";
        return text.length() <= maxLen ? text : text.substring(0, maxLen) + "…";
    }

    // ─── Response records ─────────────────────────────────────────────────────

    public record ArticleSearchHit(
            String chunkId,
            String articleId,
            int    chunkIndex,
            int    rank,
            double score,
            String excerpt
    ) {}

    // ─── RatingFilter ─────────────────────────────────────────────────────────

    record RatingFilter(String category, String operator, double value) {

        boolean matches(String text) {
            if (text == null) return false;
            String label = categoryLabel();
            for (String line : text.split("\\.")) {
                if (!line.toLowerCase().contains(label)) continue;
                java.util.regex.Matcher m =
                        java.util.regex.Pattern.compile("(\\d+(?:\\.\\d+)?)").matcher(line);
                if (!m.find()) continue;
                double v = Double.parseDouble(m.group(1));
                if (v > 10) continue; // skip year numbers etc
                return switch (operator) {
                    case "GT"  -> v >  value;
                    case "GTE" -> v >= value;
                    case "LT"  -> v <  value;
                    default    -> false;
                };
            }
            return false;
        }

        double extractValue(String text) {
            if (text == null) return 0;
            String label = categoryLabel();
            for (String line : text.split("\\.")) {
                if (!line.toLowerCase().contains(label)) continue;
                java.util.regex.Matcher m =
                        java.util.regex.Pattern.compile("(\\d+(?:\\.\\d+)?)").matcher(line);
                while (m.find()) {
                    double v = Double.parseDouble(m.group(1));
                    if (v <= 10) return v;
                }
            }
            return 0;
        }

        private String categoryLabel() {
            return switch (category) {
                case "overall"      -> "overall score";
                case "performance"  -> "performance rated";
                case "comfort"      -> "comfort and ride rated";
                case "interior"     -> "interior quality rated";
                case "value"        -> "value for money rated";
                case "technology"   -> "technology and infotainment rated";
                default             -> "overall score";
            };
        }

        static RatingFilter parse(String question) {
            String q = question.toLowerCase();

            String op;
            if (q.matches(".*(\\bover\\b|more than|above|greater than|at least).*")) op = "GTE";
            else if (q.matches(".*(\\bunder\\b|less than|below).*"))                  op = "LT";
            else if (q.matches(".*(top rated|highest rated|best rated|top score).*")) {
                return new RatingFilter("overall", "GTE", 8.5);
            }
            else return null;

            java.util.regex.Matcher numMatcher =
                    java.util.regex.Pattern.compile("(\\d+(?:\\.\\d+)?)").matcher(q);
            if (!numMatcher.find()) return null;
            double value = Double.parseDouble(numMatcher.group(1));
            if (value > 10) return null; // probably a year, not a rating

            String category = "overall";
            if (q.contains("performance"))  category = "performance";
            else if (q.contains("comfort")) category = "comfort";
            else if (q.contains("interior")) category = "interior";
            else if (q.contains("value"))   category = "value";
            else if (q.contains("tech"))    category = "technology";

            return new RatingFilter(category, op, value);
        }
    }
}