package com.example.airagassistant.rag.retrieval;

import com.example.airagassistant.rag.EmbeddingClient;
import com.example.airagassistant.rag.SearchHit;
import com.example.airagassistant.rag.VectorRecord;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Two-tier vehicle retrieval for fleet-wide queries at scale.
 *
 * The problem:
 *   At 90K vehicles × 100 chunks = 9M rows, scanning the full document_chunks
 *   table for cross-vehicle queries (e.g. "which car has best fuel economy?")
 *   takes 500ms-2s. Unacceptable for interactive use.
 *
 * The solution:
 *   Tier 1: Search vehicle_summaries (90K rows, one per vehicle) to find
 *           top-N candidate vehicles. Uses HNSW on halfvec. ~5ms.
 *   Tier 2: Search chunks_vehicle WHERE doc_id IN (top-N) for detailed
 *           chunks. 10 vehicles × 100 chunks = 1000 rows. Flat scan. <1ms.
 *
 * Total: ~6ms for fleet search instead of 500ms-2s.
 *
 * When to use:
 *   - VehicleRagService.searchAllVehicles() → use this instead of PgVectorStore.searchAllWithScores()
 *   - ArticleSubAgent vehicle-enriched search → can use Tier 1 to find candidates
 *   - Any query that doesn't scope to a single doc_id
 *
 * When NOT to use:
 *   - Single-vehicle queries (doc_id = X) → use PgVectorStore.searchWithScores() directly
 *   - Article search → use PgVectorStore with doc_type = 'article'
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class VehicleSummaryService {

    private final JdbcTemplate jdbcTemplate;
    private final EmbeddingClient embeddingClient;

    // ─── Summary population (called during ingestion) ─────────────────────

    /**
     * Create or update the vehicle summary for two-tier retrieval.
     *
     * Called after ingesting a rich vehicle. Concatenates the identity +
     * performance + reviews chunks into a single summary text, embeds it,
     * and upserts into vehicle_summaries.
     *
     * The summary embedding represents the vehicle's "fingerprint" — dense
     * enough for Tier 1 to distinguish between 90K vehicles, but small
     * enough to index efficiently (one row per vehicle).
     */
    public void upsertSummary(String vehicleId, String make, String model,
                              int year, String trim, String summaryText,
                              int chunkCount) {
        log.info("Upserting vehicle summary for '{}' — {} chars, {} chunks",
                vehicleId, summaryText.length(), chunkCount);

        List<Double> embedding = embeddingClient.embed(summaryText);
        String vectorLiteral = toPgVector(embedding);

        String sql = """
            INSERT INTO vehicle_summaries
                (vehicle_id, make, model, year, trim, summary_text, embedding, chunk_count)
            VALUES (?, ?, ?, ?, ?, ?, ?::vector, ?)
            ON CONFLICT (vehicle_id)
            DO UPDATE SET
                make         = EXCLUDED.make,
                model        = EXCLUDED.model,
                year         = EXCLUDED.year,
                trim         = EXCLUDED.trim,
                summary_text = EXCLUDED.summary_text,
                embedding    = EXCLUDED.embedding,
                chunk_count  = EXCLUDED.chunk_count,
                updated_at   = NOW()
            """;

        jdbcTemplate.update(sql, vehicleId, make, model, year, trim,
                summaryText, vectorLiteral, chunkCount);

        log.info("Vehicle summary upserted for '{}'", vehicleId);
    }

    /**
     * Build summary text from the key chunks of a vehicle.
     *
     * Concatenates identity (chunk 1) + performance (chunk 2) + reviews (chunk 7)
     * to create a rich summary that captures the vehicle's essential character.
     * This summary is what Tier 1 searches against.
     */
    public String buildSummaryText(String vehicleId) {
        String sql = """
            SELECT content FROM document_chunks
            WHERE doc_id = ? AND chunk_index IN (1, 2, 7)
            ORDER BY chunk_index
            """;

        List<String> chunks = jdbcTemplate.queryForList(sql, String.class, vehicleId);

        if (chunks.isEmpty()) {
            log.warn("No chunks found for vehicle '{}' — cannot build summary", vehicleId);
            return "";
        }

        return String.join(" ", chunks);
    }

    // ─── Two-tier retrieval (called during queries) ───────────────────────

    /**
     * Tier 1: Find top-N candidate vehicles by searching summaries.
     *
     * Uses halfvec HNSW index on vehicle_summaries (90K rows).
     * Returns vehicle IDs ranked by relevance.
     */
    public List<VehicleSummaryHit> searchSummaries(String question, int topK) {
        List<Double> queryVector = embeddingClient.embed(question);
        String vectorLiteral = toPgVector(queryVector);

        String sql = """
            SELECT
                vehicle_id,
                make,
                model,
                year,
                trim,
                summary_text,
                chunk_count,
                embedding_half <=> ?::halfvec AS distance
            FROM vehicle_summaries
            ORDER BY embedding_half <=> ?::halfvec
            LIMIT ?
            """;

        List<VehicleSummaryHit> hits = jdbcTemplate.query(sql, ps -> {
            ps.setString(1, vectorLiteral);
            ps.setString(2, vectorLiteral);
            ps.setInt(3, Math.max(1, topK));
        }, (rs, rowNum) -> new VehicleSummaryHit(
                rs.getString("vehicle_id"),
                rs.getString("make"),
                rs.getString("model"),
                rs.getInt("year"),
                rs.getString("trim"),
                rs.getInt("chunk_count"),
                1.0 - rs.getDouble("distance")  // cosine similarity
        ));

        log.info("Tier 1 — found {} candidate vehicles for '{}'", hits.size(), question);
        return hits;
    }

    /**
     * Tier 2: Deep retrieval within candidate vehicles.
     *
     * Searches chunks_vehicle for detailed chunks, scoped to the
     * candidate vehicle IDs from Tier 1. At 10 candidates × 100 chunks
     * = 1000 rows, this is essentially a flat scan.
     */
    public List<SearchHit> searchDetailChunks(List<String> candidateVehicleIds,
                                              String question, int topK) {
        if (candidateVehicleIds == null || candidateVehicleIds.isEmpty()) {
            return List.of();
        }

        List<Double> queryVector = embeddingClient.embed(question);
        String vectorLiteral = toPgVector(queryVector);

        // Build IN clause with placeholders
        String placeholders = candidateVehicleIds.stream()
                .map(id -> "?")
                .collect(Collectors.joining(","));

        String sql = """
            SELECT
                doc_id,
                chunk_index,
                content,
                embedding_half <=> ?::halfvec AS distance
            FROM document_chunks
            WHERE doc_type = 'vehicle'
              AND doc_id IN (%s)
            ORDER BY embedding_half <=> ?::halfvec
            LIMIT ?
            """.formatted(placeholders);

        // Build parameter array: vectorLiteral, ...vehicleIds, vectorLiteral, topK
        List<Object> params = new ArrayList<>();
        params.add(vectorLiteral);
        params.addAll(candidateVehicleIds);
        params.add(vectorLiteral);
        params.add(Math.max(1, topK));

        List<SearchHit> hits = jdbcTemplate.query(sql, ps -> {
            for (int i = 0; i < params.size(); i++) {
                Object param = params.get(i);
                if (param instanceof Integer intVal) {
                    ps.setInt(i + 1, intVal);
                } else {
                    ps.setString(i + 1, String.valueOf(param));
                }
            }
        }, (rs, rowNum) -> {
            String docId = rs.getString("doc_id");
            int chunkIndex = rs.getInt("chunk_index");
            String chunkId = docId + ":" + chunkIndex;
            VectorRecord record = new VectorRecord(0L, chunkIndex, chunkId,
                    rs.getString("content"), List.of());
            double score = 1.0 - rs.getDouble("distance");
            return new SearchHit(record, score);
        });

        log.info("Tier 2 — found {} detail chunks across {} vehicles for '{}'",
                hits.size(), candidateVehicleIds.size(), question);
        return hits;
    }

    /**
     * Full two-tier retrieval: summary search → detail retrieval.
     *
     * This is the main entry point for fleet-wide queries.
     * Drop-in replacement for PgVectorStore.searchAllWithScores().
     */
    public TwoTierResult searchTwoTier(String question, int summaryTopK, int detailTopK) {
        long start = System.currentTimeMillis();

        // Tier 1: find candidate vehicles
        List<VehicleSummaryHit> summaryHits = searchSummaries(question, summaryTopK);
        long tier1Ms = System.currentTimeMillis() - start;

        List<String> candidateIds = summaryHits.stream()
                .map(VehicleSummaryHit::vehicleId)
                .toList();

        // Tier 2: detail retrieval within candidates
        List<SearchHit> detailHits = searchDetailChunks(candidateIds, question, detailTopK);
        long totalMs = System.currentTimeMillis() - start;

        log.info("Two-tier search: {} summary candidates in {}ms, {} detail chunks in {}ms total",
                summaryHits.size(), tier1Ms, detailHits.size(), totalMs);

        return new TwoTierResult(summaryHits, detailHits, tier1Ms, totalMs);
    }

    // ─── ParadeDB hybrid search ───────────────────────────────────────────

    /**
     * Hybrid search combining vector similarity + BM25 in a single SQL query.
     *
     * Uses ParadeDB's pg_search for BM25 scoring and pgvector's halfvec
     * for vector similarity. The scores are combined with a tunable alpha
     * weight (0.0 = pure BM25, 1.0 = pure vector).
     *
     * This replaces the app-level RRF fusion in PgVectorStore.hybridSearch().
     */
    public List<SearchHit> hybridSearchVehicles(String question, int topK, double vectorWeight) {
        List<Double> queryVector = embeddingClient.embed(question);
        String vectorLiteral = toPgVector(queryVector);

        String sql = """
            SELECT
                doc_id,
                chunk_index,
                content,
                embedding_half <=> ?::halfvec AS vec_distance,
                paradedb.score(id) AS bm25_score
            FROM chunks_vehicle
            WHERE content @@@ ?
            ORDER BY
                (? * (1.0 - (embedding_half <=> ?::halfvec)))
                + ((1.0 - ?) * paradedb.score(id))
                DESC
            LIMIT ?
            """;

        return jdbcTemplate.query(sql, ps -> {
            ps.setString(1, vectorLiteral);
            ps.setString(2, question);
            ps.setDouble(3, vectorWeight);
            ps.setString(4, vectorLiteral);
            ps.setDouble(5, vectorWeight);
            ps.setInt(6, Math.max(1, topK));
        }, (rs, rowNum) -> {
            String docId = rs.getString("doc_id");
            int chunkIndex = rs.getInt("chunk_index");
            String chunkId = docId + ":" + chunkIndex;
            VectorRecord record = new VectorRecord(0L, chunkIndex, chunkId,
                    rs.getString("content"), List.of());
            double vecScore = 1.0 - rs.getDouble("vec_distance");
            double bm25Score = rs.getDouble("bm25_score");
            double hybridScore = vectorWeight * vecScore + (1.0 - vectorWeight) * bm25Score;
            return new SearchHit(record, hybridScore);
        });
    }

    // ─── Result records ───────────────────────────────────────────────────

    public record VehicleSummaryHit(
            String vehicleId,
            String make,
            String model,
            int year,
            String trim,
            int chunkCount,
            double score
    ) {}

    public record TwoTierResult(
            List<VehicleSummaryHit> summaryHits,
            List<SearchHit> detailHits,
            long tier1Ms,
            long totalMs
    ) {}

    // ─── Admin / observability ──────────────────────────────────────────

    /**
     * List all vehicle summaries for dashboard visibility.
     */
    public List<VehicleSummaryInfo> getAllSummaries() {
        String sql = """
            SELECT vehicle_id, make, model, year, trim, chunk_count,
                   embedding IS NOT NULL AS has_embedding,
                   created_at::text AS created_at
            FROM vehicle_summaries
            ORDER BY vehicle_id
            """;

        return jdbcTemplate.query(sql, (rs, rowNum) -> new VehicleSummaryInfo(
                rs.getString("vehicle_id"),
                rs.getString("make"),
                rs.getString("model"),
                rs.getInt("year"),
                rs.getString("trim"),
                rs.getInt("chunk_count"),
                rs.getBoolean("has_embedding"),
                rs.getString("created_at")
        ));
    }

    public record VehicleSummaryInfo(
            String vehicleId,
            String make,
            String model,
            int year,
            String trim,
            int chunkCount,
            boolean hasEmbedding,
            String createdAt
    ) {}

    // ─── Helpers ──────────────────────────────────────────────────────────

    private String toPgVector(List<Double> vector) {
        return vector.stream()
                .map(String::valueOf)
                .collect(Collectors.joining(",", "[", "]"));
    }
}
