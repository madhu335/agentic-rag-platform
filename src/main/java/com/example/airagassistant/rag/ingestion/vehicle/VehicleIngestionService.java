package com.example.airagassistant.rag.ingestion.vehicle;

import com.example.airagassistant.domain.vehicle.RichVehicleRecord;
import com.example.airagassistant.domain.vehicle.VehicleRecord;
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
 * Ingests vehicles into pgvector — batch optimized + two-tier summary.
 *
 * After ingesting chunks, also populates the vehicle_summaries table
 * so two-tier fleet retrieval works immediately. This means a single
 * POST /vehicles/ingest/rich call makes the vehicle searchable both
 * by direct doc_id lookup AND by fleet-wide summary search.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class VehicleIngestionService {

    private final EmbeddingClient embeddingClient;
    private final PgVectorStore vectorStore;
    private final VehicleDocumentBuilder documentBuilder;
    private final VehicleChunkBuilder chunkBuilder;
    private final VehicleSummaryService summaryService;

    // ─── Simple ingestion (backward-compatible) ───────────────────────────

    public void ingestVehicle(VehicleRecord vehicle) {
        validateVehicle(vehicle);

        String docText = documentBuilder.buildDocument(vehicle);
        log.info("Ingesting vehicle '{}' — {} chars (single chunk)", vehicle.vehicleId(), docText.length());

        List<Double> embedding = embed(docText, vehicle.vehicleId() + ":1");

        vectorStore.upsert(List.of(new VectorRecord(
                stableDocumentId(vehicle.vehicleId()),
                1,
                vehicle.vehicleId() + ":1",
                docText,
                embedding
        )));

        log.info("Vehicle '{}' ingested — 1 chunk", vehicle.vehicleId());
    }

    // ─── Rich ingestion (batch embed + summary) ───────────────────────────

    public IngestResult ingestRichVehicle(RichVehicleRecord vehicle) {
        validateRichVehicle(vehicle);

        List<VehicleChunkBuilder.VehicleChunk> chunks = chunkBuilder.buildChunks(vehicle);
        log.info("Ingesting rich vehicle '{}' — {} semantic chunks", vehicle.vehicleId(), chunks.size());

        long documentId = stableDocumentId(vehicle.vehicleId());

        // Batch embed all chunks
        List<String> texts = chunks.stream()
                .map(VehicleChunkBuilder.VehicleChunk::text)
                .toList();

        List<List<Double>> embeddings;
        try {
            embeddings = embeddingClient.embedBatch(texts);
        } catch (Exception e) {
            log.error("Batch embedding failed for vehicle '{}': {}",
                    vehicle.vehicleId(), e.getMessage());
            throw new IllegalStateException(
                    "Batch embedding failed for vehicleId=" + vehicle.vehicleId(), e);
        }

        if (embeddings.size() != chunks.size()) {
            throw new IllegalStateException(
                    "Embedding count mismatch for vehicleId=" + vehicle.vehicleId());
        }

        // Build records
        List<VectorRecord> records = new ArrayList<>();
        List<String> chunkIds = new ArrayList<>();
        List<String> errors = new ArrayList<>();

        for (int i = 0; i < chunks.size(); i++) {
            VehicleChunkBuilder.VehicleChunk chunk = chunks.get(i);
            String chunkId = vehicle.vehicleId() + ":" + chunk.chunkIndex();
            List<Double> embedding = embeddings.get(i);

            if (embedding == null || embedding.isEmpty()) {
                errors.add("Empty embedding for " + chunkId);
                continue;
            }

            records.add(new VectorRecord(documentId, chunk.chunkIndex(), chunkId, chunk.text(), embedding));
            chunkIds.add(chunkId);
        }

        if (records.isEmpty()) {
            throw new IllegalStateException(
                    "All chunks had empty embeddings for vehicleId=" + vehicle.vehicleId());
        }

        // Batch upsert chunks
        vectorStore.upsert(records);

        // Populate vehicle_summaries for two-tier retrieval
        try {
            String summaryText = summaryService.buildSummaryText(vehicle.vehicleId());
            if (!summaryText.isBlank()) {
                summaryService.upsertSummary(
                        vehicle.vehicleId(),
                        vehicle.make(), vehicle.model(), vehicle.year(),
                        vehicle.trim() != null ? vehicle.trim() : "",
                        summaryText, records.size());
                log.info("Vehicle summary populated for '{}'", vehicle.vehicleId());
            }
        } catch (Exception e) {
            // Summary failure is non-fatal — chunks are already stored
            log.warn("Failed to populate summary for '{}': {}", vehicle.vehicleId(), e.getMessage());
            errors.add("Summary population failed: " + e.getMessage());
        }

        log.info("Rich vehicle '{}' — {}/{} chunks stored, {} errors",
                vehicle.vehicleId(), records.size(), chunks.size(), errors.size());

        return new IngestResult(vehicle.vehicleId(), chunkIds, errors);
    }

    // ─── Result record ────────────────────────────────────────────────────

    public record IngestResult(
            String vehicleId,
            List<String> storedChunkIds,
            List<String> errors
    ) {
        public boolean hasErrors() { return !errors.isEmpty(); }
        public int chunkCount() { return storedChunkIds.size(); }
    }

    // ─── Validation ───────────────────────────────────────────────────────

    private void validateVehicle(VehicleRecord v) {
        if (v == null) throw new IllegalArgumentException("VehicleRecord cannot be null");
        requireId(v.vehicleId());
        if (v.make() == null || v.make().isBlank()) throw new IllegalArgumentException("make is required");
        if (v.model() == null || v.model().isBlank()) throw new IllegalArgumentException("model is required");
        if (v.year() < 1886 || v.year() > 2100) throw new IllegalArgumentException("year invalid");
    }

    private void validateRichVehicle(RichVehicleRecord v) {
        if (v == null) throw new IllegalArgumentException("RichVehicleRecord cannot be null");
        requireId(v.vehicleId());
        if (v.make() == null || v.make().isBlank()) throw new IllegalArgumentException("make is required");
        if (v.model() == null || v.model().isBlank()) throw new IllegalArgumentException("model is required");
        if (v.year() < 1886 || v.year() > 2100) throw new IllegalArgumentException("year invalid");
    }

    private void requireId(String vehicleId) {
        if (vehicleId == null || vehicleId.isBlank())
            throw new IllegalArgumentException("vehicleId is required");
        if (vehicleId.contains(":"))
            throw new IllegalArgumentException("vehicleId must not contain ':'");
    }

    // ─── Helpers ──────────────────────────────────────────────────────────

    private List<Double> embed(String text, String chunkId) {
        List<Double> vec = embeddingClient.embed(text);
        if (vec == null || vec.isEmpty())
            throw new IllegalStateException("Embedding returned empty for chunkId=" + chunkId);
        return vec;
    }

    private long stableDocumentId(String vehicleId) {
        return Math.abs((long) vehicleId.hashCode());
    }
}
