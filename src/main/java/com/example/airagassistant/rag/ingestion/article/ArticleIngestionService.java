package com.example.airagassistant.rag.ingestion.article;

import com.example.airagassistant.domain.article.CmsArticle;
import com.example.airagassistant.rag.EmbeddingClient;
import com.example.airagassistant.rag.PgVectorStore;
import com.example.airagassistant.rag.VectorRecord;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Ingests CmsArticle into pgvector as semantic chunks.
 *
 * doc_id convention: articleId (e.g. "motortrend-bmw-m3-2025-review")
 *
 * chunk_index assignment (see ArticleChunkBuilder):
 *   1   = identity + verdict
 *   2   = ratings narrative
 *   3   = pros and cons
 *   4   = vehicle references
 *   10+ = article sections
 *   50+ = body text sliding windows
 *
 * Gap at 5-9: reserved for future chunk types
 *   (e.g. 5=comments_summary, 6=comparison_table, 7=spec_sheet)
 * Gap at 20-49: reserved for future section types
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ArticleIngestionService {

    private final EmbeddingClient      embeddingClient;
    private final PgVectorStore        vectorStore;
    private final ArticleChunkBuilder  chunkBuilder;

    public IngestResult ingestArticle(CmsArticle article) {
        validateArticle(article);

        List<ArticleChunkBuilder.ArticleChunk> chunks = chunkBuilder.buildChunks(article);
        log.info("Ingesting article '{}' — {} chunks ({} body windows)",
                article.articleId(), chunks.size(),
                chunks.stream().filter(c -> c.chunkType().startsWith("body_window")).count());

        long         documentId = stableDocumentId(article.articleId());
        List<VectorRecord> records  = new ArrayList<>();
        List<String>       chunkIds = new ArrayList<>();
        List<String>       errors   = new ArrayList<>();

        for (ArticleChunkBuilder.ArticleChunk chunk : chunks) {
            String chunkId = article.articleId() + ":" + chunk.chunkIndex();
            try {
                List<Double> embedding = embed(chunk.text(), chunkId);
                records.add(new VectorRecord(
                        documentId,
                        chunk.chunkIndex(),
                        chunkId,
                        chunk.text(),
                        embedding
                ));
                chunkIds.add(chunkId);
                log.debug("  Embedded [{}] type={} words={}",
                        chunkId, chunk.chunkType(),
                        chunk.text().split("\\s+").length);
            } catch (Exception e) {
                String msg = "Embedding failed for " + chunkId + ": " + e.getMessage();
                log.error(msg);
                errors.add(msg);
            }
        }

        if (records.isEmpty()) {
            throw new IllegalStateException(
                    "All chunks failed embedding for articleId=" + article.articleId());
        }

        vectorStore.upsert(records);

        log.info("Article '{}' ingested — {}/{} chunks stored, {} errors",
                article.articleId(), records.size(), chunks.size(), errors.size());

        return new IngestResult(article.articleId(), chunkIds, errors);
    }

    // ─── Result ───────────────────────────────────────────────────────────────

    public record IngestResult(
            String       articleId,
            List<String> storedChunkIds,
            List<String> errors
    ) {
        public boolean hasErrors()  { return !errors.isEmpty(); }
        public int     chunkCount() { return storedChunkIds.size(); }
    }

    // ─── Validation ───────────────────────────────────────────────────────────

    private void validateArticle(CmsArticle article) {
        if (article == null)
            throw new IllegalArgumentException("CmsArticle cannot be null");
        if (article.articleId() == null || article.articleId().isBlank())
            throw new IllegalArgumentException("articleId is required");
        if (article.articleId().contains(":"))
            throw new IllegalArgumentException(
                    "articleId must not contain ':' — reserved for chunkId format. Got: "
                            + article.articleId());
        if (article.title() == null || article.title().isBlank())
            throw new IllegalArgumentException(
                    "title is required for articleId=" + article.articleId());
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    private List<Double> embed(String text, String chunkId) {
        List<Double> vec = embeddingClient.embed(text);
        if (vec == null || vec.isEmpty())
            throw new IllegalStateException("Embedding returned empty for chunkId=" + chunkId);
        return vec;
    }

    private long stableDocumentId(String articleId) {
        return Math.abs((long) articleId.hashCode());
    }
}