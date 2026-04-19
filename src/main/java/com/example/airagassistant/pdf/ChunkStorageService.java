package com.example.airagassistant.pdf;

import lombok.RequiredArgsConstructor;
import org.springframework.jdbc.core.BatchPreparedStatementSetter;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class ChunkStorageService {

    private final JdbcTemplate jdbcTemplate;

    public void deleteByDocId(String docId) {
        String sql = "DELETE FROM document_chunks WHERE doc_id = ?";
        jdbcTemplate.update(sql, docId);
    }

    // ─── Original single-chunk insert (kept for backward compat) ──────────

    public void saveChunk(String docId, int chunkIndex, String content, List<Double> embedding) {

        String vectorLiteral = embedding.stream()
                .map(String::valueOf)
                .collect(Collectors.joining(",", "[", "]"));

        String sql = """
        INSERT INTO document_chunks (doc_id, chunk_index, content, embedding)
        VALUES (?, ?, ?, ?::vector)
        """;

        jdbcTemplate.update(sql, docId, chunkIndex, content, vectorLiteral);
    }

    // ─── Batch insert (new) ───────────────────────────────────────────────

    /**
     * Insert all chunks + embeddings in a single JDBC batch call.
     *
     * chunks and embeddings must be the same size and aligned by index:
     *   chunks.get(i) was embedded into embeddings.get(i).
     *
     * Performance:
     *   20 chunks × saveChunk():      20 INSERT round-trips × ~2ms = ~40ms
     *   20 chunks × batchSaveChunks(): 1 batch round-trip    × ~5ms = ~5ms
     */
    public void batchSaveChunks(String docId, List<String> chunks, List<List<Double>> embeddings) {
        if (chunks == null || chunks.isEmpty()) return;

        if (chunks.size() != embeddings.size()) {
            throw new IllegalArgumentException(
                    "chunks.size()=" + chunks.size() + " != embeddings.size()=" + embeddings.size());
        }

        String sql = """
            INSERT INTO document_chunks (doc_id, chunk_index, content, embedding)
            VALUES (?, ?, ?, ?::vector)
            """;

        jdbcTemplate.batchUpdate(sql, new BatchPreparedStatementSetter() {
            @Override
            public void setValues(PreparedStatement ps, int i) throws SQLException {
                ps.setString(1, docId);
                ps.setInt(2, i);
                ps.setString(3, chunks.get(i));
                ps.setString(4, toVectorLiteral(embeddings.get(i)));
            }

            @Override
            public int getBatchSize() {
                return chunks.size();
            }
        });
    }

    private String toVectorLiteral(List<Double> embedding) {
        return embedding.stream()
                .map(String::valueOf)
                .collect(Collectors.joining(",", "[", "]"));
    }
}
