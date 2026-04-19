package com.example.airagassistant.rag;

import com.example.airagassistant.trace.TraceHelper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Component;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Component
public class PgVectorStore {

    private final JdbcTemplate jdbcTemplate;
    private final TraceHelper traceHelper;

    public PgVectorStore(JdbcTemplate jdbcTemplate, TraceHelper traceHelper) {
        this.jdbcTemplate = jdbcTemplate;
        this.traceHelper = traceHelper;
    }

    // ---------------- VECTOR SEARCH ----------------
    public List<SearchHit> searchWithScores(String docId, List<Double> queryVector, int topK) {

        return traceHelper.run(
                "vector-search",
                buildBaseAttrs(docId, topK),
                () -> {
                    int k = Math.max(1, topK);
                    String vectorLiteral = toPgVector(queryVector);

                    String sql = """
                            SELECT
                                doc_id,
                                chunk_index,
                                content,
                                embedding <=> ?::vector AS distance
                            FROM document_chunks
                            WHERE doc_id = ?
                            ORDER BY embedding <=> ?::vector
                            LIMIT ?
                            """;

                    List<SearchHit> results = jdbcTemplate.query(
                            sql,
                            ps -> {
                                ps.setString(1, vectorLiteral);
                                ps.setString(2, docId); // Append wildcard here
                                ps.setString(3, vectorLiteral);
                                ps.setInt(4, k);
                            },
                            searchHitRowMapper()
                    );

                    addHitAttributes(results);
                    return results;
                }
        );
    }

    // ---------------- KEYWORD SEARCH ----------------
    public List<SearchHit> keywordSearch(String docId, String question, int topK) {

        Map<String, Object> attrs = buildBaseAttrs(docId, topK);
        attrs.put("gen_ai.prompt.0.content", question);

        return traceHelper.run(
                "keyword-search",
                attrs,
                () -> {
                    String sql = """
                            SELECT
                                doc_id,
                                chunk_index,
                                content,
                                ts_rank_cd(content_tsv, plainto_tsquery('english', ?)) AS score
                            FROM document_chunks
                            WHERE doc_id = ?
                              AND content_tsv @@ plainto_tsquery('english', ?)
                            ORDER BY score DESC, chunk_index ASC
                            LIMIT ?
                            """;

                    List<SearchHit> results = jdbcTemplate.query(
                            sql,
                            ps -> {
                                ps.setString(1, question);
                                ps.setString(2, docId);
                                ps.setString(3, question);
                                ps.setInt(4, Math.max(1, topK));
                            },
                            (rs, rowNum) -> {
                                String foundDocId = rs.getString("doc_id");
                                int chunkIndex = rs.getInt("chunk_index");
                                String chunkId = foundDocId + ":" + chunkIndex;

                                VectorRecord record = new VectorRecord(
                                        0L,
                                        chunkIndex,
                                        chunkId,
                                        rs.getString("content"),
                                        List.of()
                                );

                                double score = rs.getDouble("score");
                                return new SearchHit(record, score);
                            }
                    );

                    addHitAttributes(results);
                    return results;
                }
        );
    }

    // ---------------- HYBRID SEARCH (RRF) ----------------
    public List<SearchHit> hybridSearch(String docId, List<Double> queryVector, String question, int topK) {

        return traceHelper.run(
                "hybrid-search",
                buildBaseAttrs(docId, topK),
                () -> {

                    int k = Math.max(1, topK);

                    List<SearchHit> vectorHits = searchWithScores(docId, queryVector, k * 3);
                    List<SearchHit> keywordHits = keywordSearch(docId, question, k * 3);

                    traceHelper.addAttributes(Map.of(
                            "langsmith.metadata.vector_hits", vectorHits.size(),
                            "langsmith.metadata.keyword_hits", keywordHits.size()
                    ));

                    return traceHelper.run(
                            "rrf-fusion",
                            null,
                            () -> fuseRRF(vectorHits, keywordHits, k)
                    );
                }
        );
    }

    // ---------------- RRF CORE ----------------
    private List<SearchHit> fuseRRF(List<SearchHit> vectorHits,
                                    List<SearchHit> keywordHits,
                                    int k) {

        Map<String, Integer> vectorRanks = new HashMap<>();
        for (int i = 0; i < vectorHits.size(); i++) {
            vectorRanks.put(vectorHits.get(i).record().id(), i + 1);
        }

        Map<String, Integer> keywordRanks = new HashMap<>();
        for (int i = 0; i < keywordHits.size(); i++) {
            keywordRanks.put(keywordHits.get(i).record().id(), i + 1);
        }

        Map<String, Double> rrfScores = new HashMap<>();
        Map<String, SearchHit> sourceRecords = new HashMap<>();

        int kConstant = 10;

        for (SearchHit hit : vectorHits) {
            String id = hit.record().id();
            sourceRecords.put(id, hit);
            int rank = vectorRanks.get(id);
            rrfScores.put(id, rrfScores.getOrDefault(id, 0.0) + (1.0 / (kConstant + rank)));
        }

        for (SearchHit hit : keywordHits) {
            String id = hit.record().id();
            sourceRecords.putIfAbsent(id, hit);
            int rank = keywordRanks.get(id);
            rrfScores.put(id, rrfScores.getOrDefault(id, 0.0) + (1.0 / (kConstant + rank)));
        }

        List<SearchHit> finalResults = rrfScores.entrySet().stream()
                .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                .limit(k)
                .map(entry -> new SearchHit(
                        sourceRecords.get(entry.getKey()).record(),
                        entry.getValue()
                ))
                .toList();

        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.metadata.final_hits", finalResults.size());

        if (!finalResults.isEmpty()) {
            attrs.put("langsmith.metadata.top_chunk", finalResults.get(0).record().id());
            attrs.put("langsmith.metadata.top_score", finalResults.get(0).score());
        }

        traceHelper.addAttributes(attrs);

        return finalResults;
    }

    // ---------------- COMMON ATTRS ----------------
    private Map<String, Object> buildBaseAttrs(String docId, int topK) {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.span.kind", "retriever");
        attrs.put("langsmith.metadata.doc_id", docId);
        attrs.put("langsmith.metadata.top_k", topK);
        return attrs;
    }

    private void addHitAttributes(List<SearchHit> results) {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.metadata.hit_count", results.size());

        if (!results.isEmpty()) {
            attrs.put("langsmith.metadata.top_chunk_id", results.get(0).record().id());
            attrs.put("langsmith.metadata.top_score", results.get(0).score());
        }

        traceHelper.addAttributes(attrs);
    }

    // ---------------- MAPPER ----------------
    private RowMapper<SearchHit> searchHitRowMapper() {
        return new RowMapper<>() {
            @Override
            public SearchHit mapRow(ResultSet rs, int rowNum) throws SQLException {
                String docId = rs.getString("doc_id");
                int chunkIndex = rs.getInt("chunk_index");
                String chunkId = docId + ":" + chunkIndex;

                VectorRecord record = new VectorRecord(
                        0L,
                        chunkIndex,
                        chunkId,
                        rs.getString("content"),
                        List.of()
                );

                double distance = rs.getDouble("distance");
                double score = 1.0 - distance;

                return new SearchHit(record, score);
            }
        };
    }

    private String toPgVector(List<Double> vector) {
        return vector.stream()
                .map(String::valueOf)
                .collect(Collectors.joining(",", "[", "]"));
    }

    // ---------------- INSERT ----------------
    public void add(VectorRecord record) {
        if (record == null || record.vector() == null || record.vector().isEmpty()) {
            throw new IllegalArgumentException("Record/vector cannot be null/empty");
        }
        insertChunk(record);
    }

    private void insertChunk(VectorRecord record) {
        String sql = """
                INSERT INTO document_chunks (doc_id, chunk_index, content, embedding)
                VALUES (?, ?, ?, ?::vector)
                """;

        String docId = extractDocId(record.id());

        jdbcTemplate.update(
                sql,
                docId,
                record.chunkIndex(),
                record.text(),
                toPgVector(record.vector())
        );
    }

    private String extractDocId(String chunkId) {
        int idx = chunkId.lastIndexOf(':');
        if (idx <= 0) {
            throw new IllegalArgumentException("Invalid chunkId: " + chunkId);
        }
        return chunkId.substring(0, idx);
    }

    /**
     * Derives doc_type from docId naming convention.
     * Articles → start with "motortrend-"
     * Vehicles → contain a 4-digit year  (e.g. bmw-m3-2025-competition)
     * PDFs     → everything else
     */
    private String inferDocType(String docId) {
        if (docId == null) return "pdf";
        if (docId.startsWith("motortrend-")) return "article";
        if (docId.matches(".*-\\d{4}(-.*)?$")) return "vehicle";
        return "pdf";
    }

// ─────────────────────────────────────────────────────────────────────────────
// Also add a cross-document vector search (no doc_id filter) for vehicle
// queries that span all ingested vehicles:
// ─────────────────────────────────────────────────────────────────────────────

    /**
     * Searches across ALL documents (no doc_id filter).
     * Used by VehicleRagService to compare or rank across multiple vehicles.
     */
    public List<SearchHit> searchAllWithScores(List<Double> queryVector, int topK) {
        int k = Math.max(1, topK);
        String vectorLiteral = toPgVector(queryVector);

        String sql = """
                SELECT
                    doc_id,
                    chunk_index,
                    content,
                    embedding <=> ?::vector AS distance
                FROM document_chunks
                ORDER BY embedding <=> ?::vector
                LIMIT ?
                """;

        return jdbcTemplate.query(
                sql,
                ps -> {
                    ps.setString(1, vectorLiteral);
                    ps.setString(2, vectorLiteral);
                    ps.setInt(3, k);
                },
                searchHitRowMapper()
        );
    }

    /**
     * Full-text keyword search across ALL documents (no doc_id filter).
     */
    public List<SearchHit> keywordSearchAll(String question, int topK) {
        String sql = """
                SELECT
                    doc_id,
                    chunk_index,
                    content,
                    ts_rank_cd(content_tsv, plainto_tsquery('english', ?)) AS score
                FROM document_chunks
                WHERE content_tsv @@ plainto_tsquery('english', ?)
                ORDER BY score DESC, chunk_index ASC
                LIMIT ?
                """;

        return jdbcTemplate.query(
                sql,
                ps -> {
                    ps.setString(1, question);
                    ps.setString(2, question);
                    ps.setInt(3, Math.max(1, topK));
                },
                (rs, rowNum) -> {
                    String foundDocId = rs.getString("doc_id");
                    int chunkIndex = rs.getInt("chunk_index");
                    String chunkId = foundDocId + ":" + chunkIndex;
                    VectorRecord record = new VectorRecord(0L, chunkIndex, chunkId,
                            rs.getString("content"), List.of());
                    return new SearchHit(record, rs.getDouble("score"));
                }
        );
    }

    // ---------------- DOC_TYPE SCOPED SEARCH ----------------

    /**
     * Vector search scoped to a specific doc_type.
     * Prevents article searches returning vehicle or pdf chunks.
     * Usage: vectorStore.searchAllWithScores(queryVector, topK, "article")
     */
    public List<SearchHit> searchAllWithScores(List<Double> queryVector, int topK, String docType) {
        int k = Math.max(1, topK);
        String vectorLiteral = toPgVector(queryVector);

        String sql = """
                SELECT
                    doc_id,
                    chunk_index,
                    content,
                    embedding <=> ?::vector AS distance
                FROM document_chunks
                WHERE doc_type = ?
                ORDER BY embedding <=> ?::vector
                LIMIT ?
                """;

        return jdbcTemplate.query(
                sql,
                ps -> {
                    ps.setString(1, vectorLiteral);
                    ps.setString(2, docType);
                    ps.setString(3, vectorLiteral);
                    ps.setInt(4, k);
                },
                searchHitRowMapper()
        );
    }

    /**
     * Keyword search scoped to a specific doc_type.
     * Usage: vectorStore.keywordSearchAll(question, topK, "article")
     */
    public List<SearchHit> keywordSearchAll(String question, int topK, String docType) {
        String sql = """
                SELECT
                    doc_id,
                    chunk_index,
                    content,
                    ts_rank_cd(content_tsv, plainto_tsquery('english', ?)) AS score
                FROM document_chunks
                WHERE doc_type = ?
                  AND content_tsv @@ plainto_tsquery('english', ?)
                ORDER BY score DESC, chunk_index ASC
                LIMIT ?
                """;

        return jdbcTemplate.query(
                sql,
                ps -> {
                    ps.setString(1, question);
                    ps.setString(2, docType);
                    ps.setString(3, question);
                    ps.setInt(4, Math.max(1, topK));
                },
                (rs, rowNum) -> {
                    String foundDocId = rs.getString("doc_id");
                    int chunkIndex = rs.getInt("chunk_index");
                    String chunkId = foundDocId + ":" + chunkIndex;
                    VectorRecord record = new VectorRecord(0L, chunkIndex, chunkId,
                            rs.getString("content"), List.of());
                    return new SearchHit(record, rs.getDouble("score"));
                }
        );
    }

    // ─── BATCH OPERATIONS ──────────────────────────────────────────────────
    //
    // Drop-in replacements for the serial addAll() and upsert() methods.
    // Replace the existing methods in PgVectorStore.java with these.

    /**
     * Insert a batch of VectorRecords using JDBC batch execution.
     * <p>
     * Previous implementation: loop calling add() → one INSERT per chunk.
     * This implementation: one jdbcTemplate.batchUpdate() call → one
     * round-trip to Postgres with all INSERTs batched.
     * <p>
     * For 50 chunks:
     * - Serial:  50 round-trips × ~2ms = ~100ms
     * - Batch:   1 round-trip   × ~10ms = ~10ms
     */
    public void addAll(List<VectorRecord> batch) {
        if (batch == null || batch.isEmpty()) return;
        log.debug("======Batch Add all VectorRecord size {} =======", batch.size());
        String sql = """
                INSERT INTO document_chunks (doc_id, chunk_index, content, embedding)
                VALUES (?, ?, ?, ?::vector)
                """;

        jdbcTemplate.batchUpdate(sql, new org.springframework.jdbc.core.BatchPreparedStatementSetter() {
            @Override
            public void setValues(java.sql.PreparedStatement ps, int i) throws java.sql.SQLException {
                VectorRecord record = batch.get(i);
                ps.setString(1, extractDocId(record.id()));
                ps.setInt(2, record.chunkIndex());
                ps.setString(3, record.text());
                ps.setString(4, toPgVector(record.vector()));
            }

            @Override
            public int getBatchSize() {
                return batch.size();
            }
        });
    }

    /**
     * Upsert a batch of VectorRecords using JDBC batch execution.
     * <p>
     * Previous implementation: loop calling jdbcTemplate.update() per record.
     * This implementation: one jdbcTemplate.batchUpdate() call.
     * <p>
     * Uses INSERT ... ON CONFLICT (doc_id, chunk_index) DO UPDATE so that
     * re-ingesting the same vehicle/article replaces its old embeddings
     * without leaving stale duplicates.
     */
    public void upsert(List<VectorRecord> batch) {
        log.debug("======Batch upsert VectorRecord size {} =======", batch.size());
        if (batch == null || batch.isEmpty()) return;

        String sql = """
                INSERT INTO document_chunks (doc_id, chunk_index, content, embedding, doc_type)
                VALUES (?, ?, ?, ?::vector, ?)
                ON CONFLICT (doc_id, chunk_index, doc_type)
                DO UPDATE SET
                    content   = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    doc_type  = EXCLUDED.doc_type
                """;

        jdbcTemplate.batchUpdate(sql, new org.springframework.jdbc.core.BatchPreparedStatementSetter() {
            @Override
            public void setValues(java.sql.PreparedStatement ps, int i) throws java.sql.SQLException {
                VectorRecord record = batch.get(i);
                String docId = extractDocId(record.id());
                ps.setString(1, docId);
                ps.setInt(2, record.chunkIndex());
                ps.setString(3, record.text());
                ps.setString(4, toPgVector(record.vector()));
                ps.setString(5, inferDocType(docId));
            }

            @Override
            public int getBatchSize() {
                return batch.size();
            }
        });
    }


}