package com.example.airagassistant.pdf;

import lombok.RequiredArgsConstructor;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

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
}