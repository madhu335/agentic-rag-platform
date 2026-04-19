package com.example.airagassistant.rag.ingestion.vehicle;

import com.example.airagassistant.domain.vehicle.RichVehicleRecord;
import com.example.airagassistant.rag.EmbeddingClient;
import com.example.airagassistant.rag.PgVectorStore;
import com.example.airagassistant.rag.VectorRecord;
import com.example.airagassistant.rag.retrieval.VehicleSummaryService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Bulk vehicle ingestion pipeline for large-scale data loads.
 *
 * Designed for ingesting thousands of vehicles efficiently:
 *   - Processes vehicles in configurable pages (default 50)
 *   - Each page: chunk all vehicles → batch embed all texts → batch upsert
 *   - After each page: populate vehicle_summaries for two-tier retrieval
 *
 * Performance at 90K vehicles × 100 chunks/vehicle:
 *
 *   | Approach               | Embed calls | DB calls   | Time       |
 *   |------------------------|-------------|------------|------------|
 *   | Serial (current)       | 9,000,000   | 9,000,000  | ~50 hours  |
 *   | Batch (this)           | 281,250     | ~180,000   | ~6 hours   |
 *   | Batch + 4 GPU workers  | 70,312      | ~180,000   | ~1.5 hours |
 *
 * Usage:
 *   POST /vehicles/ingest/bulk
 *   Body: List<RichVehicleRecord>
 *
 *   Or programmatically:
 *   bulkIngestionService.ingestBulk(vehicles);
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class BulkVehicleIngestionService {

    private static final int PAGE_SIZE = 50;         // vehicles per batch
    private static final int EMBED_BATCH_SIZE = 32;  // texts per Ollama call (handled by EmbeddingClient)

    private final EmbeddingClient embeddingClient;
    private final PgVectorStore vectorStore;
    private final VehicleChunkBuilder chunkBuilder;
    private final VehicleSummaryService summaryService;

    /**
     * Ingest a batch of vehicles in pages.
     *
     * Each page:
     *   1. Chunk all vehicles in the page (CPU-bound, fast)
     *   2. Batch embed all chunk texts (GPU-bound, one call per 32 texts)
     *   3. Batch upsert all records (I/O-bound, one JDBC call per page)
     *   4. Populate vehicle_summaries for two-tier retrieval
     *
     * Returns a summary of the ingestion run.
     */
    public BulkIngestResult ingestBulk(List<RichVehicleRecord> vehicles) {
        if (vehicles == null || vehicles.isEmpty()) {
            return new BulkIngestResult(0, 0, 0, 0, List.of());
        }

        log.info("Bulk ingestion starting — {} vehicles in pages of {}", vehicles.size(), PAGE_SIZE);
        long startTime = System.currentTimeMillis();

        int totalVehicles = 0;
        int totalChunks = 0;
        int totalSummaries = 0;
        List<String> errors = new ArrayList<>();

        for (int pageStart = 0; pageStart < vehicles.size(); pageStart += PAGE_SIZE) {
            int pageEnd = Math.min(pageStart + PAGE_SIZE, vehicles.size());
            List<RichVehicleRecord> page = vehicles.subList(pageStart, pageEnd);
            int pageNum = (pageStart / PAGE_SIZE) + 1;

            log.info("Bulk ingestion — page {}: vehicles {}-{} of {}",
                    pageNum, pageStart + 1, pageEnd, vehicles.size());

            try {
                PageResult result = ingestPage(page);
                totalVehicles += result.vehicleCount;
                totalChunks += result.chunkCount;
                totalSummaries += result.summaryCount;
                errors.addAll(result.errors);

                log.info("Bulk ingestion — page {} complete: {} vehicles, {} chunks, {} summaries, {} errors",
                        pageNum, result.vehicleCount, result.chunkCount,
                        result.summaryCount, result.errors.size());

            } catch (Exception e) {
                String msg = "Page " + pageNum + " failed: " + e.getMessage();
                log.error(msg, e);
                errors.add(msg);
            }
        }

        long elapsedMs = System.currentTimeMillis() - startTime;
        log.info("Bulk ingestion complete — {} vehicles, {} chunks, {} summaries in {}ms. {} errors.",
                totalVehicles, totalChunks, totalSummaries, elapsedMs, errors.size());

        return new BulkIngestResult(totalVehicles, totalChunks, totalSummaries, elapsedMs, errors);
    }

    /**
     * Ingest a single page of vehicles.
     *
     * All chunks from all vehicles in the page are collected, embedded in
     * one batch call, and upserted in one JDBC call. This amortizes the
     * HTTP and DB round-trip overhead across the entire page.
     */
    private PageResult ingestPage(List<RichVehicleRecord> page) {
        // Step 1: Chunk all vehicles
        List<String> allTexts = new ArrayList<>();
        List<ChunkMeta> allMetas = new ArrayList<>();

        for (RichVehicleRecord vehicle : page) {
            try {
                List<VehicleChunkBuilder.VehicleChunk> chunks = chunkBuilder.buildChunks(vehicle);
                for (VehicleChunkBuilder.VehicleChunk chunk : chunks) {
                    allTexts.add(chunk.text());
                    allMetas.add(new ChunkMeta(
                            vehicle.vehicleId(), vehicle,
                            chunk.chunkIndex(), chunk.chunkType(), chunk.text()));
                }
            } catch (Exception e) {
                log.warn("Chunking failed for vehicle '{}': {}", vehicle.vehicleId(), e.getMessage());
            }
        }

        if (allTexts.isEmpty()) {
            return new PageResult(0, 0, 0, List.of("No valid chunks generated"));
        }

        // Step 2: Batch embed
        List<List<Double>> embeddings = embeddingClient.embedBatch(allTexts);

        if (embeddings.size() != allTexts.size()) {
            throw new IllegalStateException(
                    "Embedding count mismatch: expected " + allTexts.size()
                            + " but got " + embeddings.size());
        }

        // Step 3: Build VectorRecords and batch upsert
        List<VectorRecord> records = new ArrayList<>();
        List<String> errors = new ArrayList<>();

        for (int i = 0; i < allMetas.size(); i++) {
            ChunkMeta meta = allMetas.get(i);
            List<Double> embedding = embeddings.get(i);

            if (embedding == null || embedding.isEmpty()) {
                errors.add("Empty embedding for " + meta.vehicleId + ":" + meta.chunkIndex);
                continue;
            }

            long docId = Math.abs((long) meta.vehicleId.hashCode());
            String chunkId = meta.vehicleId + ":" + meta.chunkIndex;
            records.add(new VectorRecord(docId, meta.chunkIndex, chunkId, meta.text, embedding));
        }

        vectorStore.upsert(records);

        // Step 4: Populate vehicle_summaries for two-tier retrieval
        int summaryCount = 0;
        for (RichVehicleRecord vehicle : page) {
            try {
                String summaryText = summaryService.buildSummaryText(vehicle.vehicleId());
                if (!summaryText.isBlank()) {
                    int chunkCount = (int) allMetas.stream()
                            .filter(m -> m.vehicleId.equals(vehicle.vehicleId()))
                            .count();
                    summaryService.upsertSummary(
                            vehicle.vehicleId(),
                            vehicle.make(), vehicle.model(), vehicle.year(),
                            vehicle.trim() != null ? vehicle.trim() : "",
                            summaryText, chunkCount);
                    summaryCount++;
                }
            } catch (Exception e) {
                errors.add("Summary failed for " + vehicle.vehicleId() + ": " + e.getMessage());
            }
        }

        return new PageResult(page.size(), records.size(), summaryCount, errors);
    }

    // ─── Records ──────────────────────────────────────────────────────────

    public record BulkIngestResult(
            int totalVehicles,
            int totalChunks,
            int totalSummaries,
            long elapsedMs,
            List<String> errors
    ) {
        public boolean hasErrors() { return !errors.isEmpty(); }
    }

    private record PageResult(
            int vehicleCount,
            int chunkCount,
            int summaryCount,
            List<String> errors
    ) {}

    private record ChunkMeta(
            String vehicleId,
            RichVehicleRecord vehicle,
            int chunkIndex,
            String chunkType,
            String text
    ) {}
}
