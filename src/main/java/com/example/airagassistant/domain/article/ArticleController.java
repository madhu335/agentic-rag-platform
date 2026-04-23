package com.example.airagassistant.domain.article;

import com.example.airagassistant.agentic.AgentSessionRunner;
import com.example.airagassistant.domain.article.service.ArticleRagService;
import com.example.airagassistant.rag.RagAnswerService;
import com.example.airagassistant.rag.RetrievalMode;
import com.example.airagassistant.rag.SearchHit;
import com.example.airagassistant.rag.ingestion.article.ArticleIngestionService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * Article RAG REST API.
 * <p>
 * POST /articles/ingest                — ingest a CMS article
 * POST /articles/{id}/ask             — ask a question about one article
 * POST /articles/search               — cross-article semantic search
 * POST /articles/search/vehicle       — find all articles about a vehicle
 */
@RestController
@RequestMapping("/articles")
@RequiredArgsConstructor
public class ArticleController {

    private final ArticleIngestionService ingestionService;
    private final ArticleRagService articleRagService;
    private final AgentSessionRunner sessionRunner;
    // ─── DTOs ─────────────────────────────────────────────────────────────────

    public record ArticleAskRequest(
            String question,
            Integer topK,
            String mode
    ) {
    }

    public record ArticleAskResponse(
            String articleId,
            String answer,
            List<String> retrievedChunkIds,
            List<String> citedChunkIds,
            int usedChunks,
            Double bestScore
    ) {
    }

    public record ArticleSearchRequest(
            String question,
            Integer topK
    ) {
    }

    public record VehicleArticleSearchRequest(
            String vehicleQuery,
            Integer topK
    ) {
    }

    public record VehicleArticlesResponse(
            String vehicleQuery,
            int articleCount,
            Map<String, List<ArticleRagService.ArticleSearchHit>> articlesByVehicle
    ) {
    }

    // ─── Ingest ───────────────────────────────────────────────────────────────

    /**
     * POST /articles/ingest
     * Ingests a CmsArticle as semantic chunks.
     * <p>
     * Returns storedChunkIds so you can verify which chunks were created.
     */
    @PostMapping("/ingest")
    public ArticleIngestionService.IngestResult ingest(@RequestBody CmsArticle article) {
        return ingestionService.ingestArticle(article);
    }

    // ─── Single-article ask ───────────────────────────────────────────────────

    /**
     * POST /articles/{articleId}/ask
     * <p>
     * Example:
     * POST /articles/motortrend-bmw-m3-2025-review/ask
     * { "question": "What did MotorTrend say about the M3's performance?" }
     */
    @PostMapping("/{articleId}/ask")
    public ArticleAskResponse askArticle(
            @PathVariable String articleId,
            @RequestBody ArticleAskRequest req
    ) {
        int topK = (req.topK() == null || req.topK() <= 0) ? 5 : req.topK();
        RetrievalMode mode = parseMode(req.mode());

        RagAnswerService.RagResult result = sessionRunner.runRagWithSession(
                req.question(),
                articleId,
                () -> articleRagService.askArticle(
                        articleId,
                        req.question(),
                        topK,
                        mode
                )
        );

        return new ArticleAskResponse(
                articleId,
                result.answer(),
                result.retrievedChunkIds(),
                result.citedChunkIds(),
                result.usedChunks(),
                result.retrievalScore()
        );
    }

    // ─── Cross-article search ─────────────────────────────────────────────────

    /**
     * POST /articles/search
     * Semantic search across all ingested articles.
     * <p>
     * Examples:
     * { "question": "which car did MotorTrend rate best for value in 2025?" }
     * { "question": "cars rated over 9 out of 10" }
     * { "question": "best sports sedan for daily driving" }
     * { "question": "M3 vs 911 interior comparison" }
     */
    @PostMapping("/search")
    public List<ArticleRagService.ArticleSearchHit> searchArticles(
            @RequestBody ArticleSearchRequest req
    ) {
        int topK = (req.topK() == null || req.topK() <= 0) ? 5 : req.topK();
        return articleRagService.searchAllArticles(req.question(), topK);
    }

    // ─── Vehicle-scoped article search ────────────────────────────────────────

    /**
     * POST /articles/search/vehicle
     * Finds all articles that reviewed or mentioned a specific vehicle.
     * <p>
     * Example:
     * { "vehicleQuery": "BMW M3 2025" }
     * Returns: all MotorTrend articles that feature the M3
     */
    @PostMapping("/search/vehicle")
    public VehicleArticlesResponse searchByVehicle(
            @RequestBody VehicleArticleSearchRequest req
    ) {
        int topK = (req.topK() == null || req.topK() <= 0) ? 10 : req.topK();

        Map<String, List<SearchHit>> results =
                articleRagService.searchArticlesByVehicle(req.vehicleQuery(), topK);

        // Convert SearchHit to ArticleSearchHit for cleaner response
        Map<String, List<ArticleRagService.ArticleSearchHit>> converted =
                new java.util.LinkedHashMap<>();

        int[] rank = {1};
        results.forEach((articleId, hits) -> {
            List<ArticleRagService.ArticleSearchHit> articleHits = hits.stream()
                    .map(h -> new ArticleRagService.ArticleSearchHit(
                            h.record().id(),
                            articleId,
                            extractChunkIndex(h.record().id()),
                            rank[0]++,
                            h.score(),
                            excerpt(h.record().text(), 200)
                    ))
                    .toList();
            converted.put(articleId, articleHits);
        });

        return new VehicleArticlesResponse(req.vehicleQuery(), converted.size(), converted);
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    private RetrievalMode parseMode(String mode) {
        if (mode == null || mode.isBlank()) return RetrievalMode.HYBRID;
        try {
            return RetrievalMode.valueOf(mode.toUpperCase());
        } catch (IllegalArgumentException e) {
            return RetrievalMode.HYBRID;
        }
    }

    private int extractChunkIndex(String chunkId) {
        int idx = chunkId.lastIndexOf(':');
        if (idx < 0) return -1;
        try {
            return Integer.parseInt(chunkId.substring(idx + 1));
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    private String excerpt(String text, int maxLen) {
        if (text == null) return "";
        return text.length() <= maxLen ? text : text.substring(0, maxLen) + "…";
    }
}