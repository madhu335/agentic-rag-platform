package com.example.airagassistant;

import com.example.airagassistant.rag.EmbeddingClient;
import com.example.airagassistant.rag.PgVectorStore;
import com.example.airagassistant.rag.SearchHit;
import com.example.airagassistant.rag.VectorIngestionService;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.support.GeneratedKeyHolder;
import org.springframework.jdbc.support.KeyHolder;
import org.springframework.web.bind.annotation.*;

import java.sql.PreparedStatement;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping("/api")
public class IngestController {

    private final JdbcTemplate jdbc;
    private final VectorIngestionService vectorIngestionService;
    private final PgVectorStore vectorStore;
    private final EmbeddingClient embeddingClient;

    public IngestController(JdbcTemplate jdbc,
                            VectorIngestionService vectorIngestionService,
                            PgVectorStore vectorStore,
                            EmbeddingClient embeddingClient) {
        this.jdbc = jdbc;
        this.vectorIngestionService = vectorIngestionService;
        this.vectorStore = vectorStore;
        this.embeddingClient = embeddingClient;
    }

    public record IngestRequest(String title, String source, String content) {}

    public record IngestResponse(long documentId, int chunksInserted) {}

    public record SearchRequest(String query, Integer topK) {}

    public record SearchResponse(String query, int topK, List<SearchHit> hits) {}

    @PostMapping("/ingest")
    public IngestResponse ingest(@RequestBody IngestRequest req) {
        if (req == null || req.content() == null || req.content().isBlank()) {
            throw new IllegalArgumentException("content is required");
        }

        long docId = insertDocument(
                req.title() == null ? "Untitled" : req.title(),
                req.source() == null ? "manual" : req.source()
        );

        List<String> chunks = chunk(req.content(), 800);

        String vectorDocId = "doc:" + docId;
        vectorIngestionService.ingestChunks(docId, vectorDocId, chunks);

        return new IngestResponse(docId, chunks.size());
    }

    @PostMapping("/search")
    public SearchResponse search(@RequestBody SearchRequest req) {
        if (req == null || req.query() == null || req.query().isBlank()) {
            throw new IllegalArgumentException("query is required");
        }

        int topK = (req.topK() == null || req.topK() <= 0) ? 3 : req.topK();

        var queryVector = vectorIngestionServiceQueryEmbed(req.query());
        var hits = List.<SearchHit>of();

        return new SearchResponse(req.query(), topK, hits);
    }

    private List<Double> vectorIngestionServiceQueryEmbed(String query) {
        return embeddingClient.embed(query);
    }

    private List<String> chunk(String text, int maxChars) {
        String cleaned = text.replace("\r\n", "\n").trim();
        List<String> out = new ArrayList<>();
        int i = 0;
        while (i < cleaned.length()) {
            int end = Math.min(i + maxChars, cleaned.length());
            out.add(cleaned.substring(i, end));
            i = end;
        }
        return out;
    }

    private long insertDocument(String title, String source) {
        KeyHolder keyHolder = new GeneratedKeyHolder();

        jdbc.update(con -> {
            PreparedStatement ps = con.prepareStatement(
                    "insert into documents(title, source) values(?, ?)",
                    Statement.RETURN_GENERATED_KEYS
            );
            ps.setString(1, title);
            ps.setString(2, source);
            return ps;
        }, keyHolder);

        return ((Number) keyHolder.getKeys().get("ID")).longValue();
    }
}