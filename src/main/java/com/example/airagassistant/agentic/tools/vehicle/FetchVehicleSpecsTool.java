package com.example.airagassistant.agentic.tools.vehicle;

import com.example.airagassistant.rag.EmbeddingClient;
import com.example.airagassistant.rag.PgVectorStore;
import com.example.airagassistant.rag.SearchHit;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Agentic tool: FetchVehicleSpecs
 *
 * Retrieves the top matching spec chunk(s) for a given vehicleId.
 * Used by the planner when it needs raw spec data before generating a summary
 * or comparison.
 *
 * Input:  vehicleId  (e.g. "tesla-model3-2025-long-range")
 * Output: list of SpecChunk (chunkId + text)
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class FetchVehicleSpecsTool {

    private final EmbeddingClient embeddingClient;
    private final PgVectorStore vectorStore;

    public record Input(String vehicleId, String question, int topK) {}
    public record SpecChunk(String chunkId, String text, double score) {}

    /**
     * Fetch top-K spec chunks for a vehicle.
     * If question is null/blank, uses a generic "vehicle specifications" query
     * so the embedding still returns the full spec.
     */
    public List<SpecChunk> execute(Input input) {
        if (input.vehicleId() == null || input.vehicleId().isBlank()) {
            throw new IllegalArgumentException("vehicleId is required");
        }

        String query = (input.question() == null || input.question().isBlank())
                ? "vehicle specifications engine horsepower features"
                : input.question();

        int topK = input.topK() <= 0 ? 3 : input.topK();

        log.debug("FetchVehicleSpecsTool — vehicleId='{}' query='{}' topK={}", input.vehicleId(), query, topK);

        List<Double> queryVector = embeddingClient.embed(query);
        List<SearchHit> hits = vectorStore.searchWithScores(input.vehicleId(), queryVector, topK);

        return hits.stream()
                .map(h -> new SpecChunk(h.record().id(), h.record().text(), h.score()))
                .toList();
    }
}