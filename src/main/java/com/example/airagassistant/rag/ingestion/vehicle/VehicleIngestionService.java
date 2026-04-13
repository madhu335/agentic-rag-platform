package com.example.airagassistant.rag.ingestion.vehicle;

import com.example.airagassistant.domain.vehicle.RichVehicleRecord;
import com.example.airagassistant.domain.vehicle.VehicleRecord;
import com.example.airagassistant.rag.EmbeddingClient;
import com.example.airagassistant.rag.PgVectorStore;
import com.example.airagassistant.rag.VectorRecord;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Ingests vehicles into pgvector.
 *
 * Two modes:
 *
 * 1. ingestVehicle(VehicleRecord)         — simple flat record → 1 chunk
 *    Backward-compatible with existing /vehicles/ingest endpoint.
 *
 * 2. ingestRichVehicle(RichVehicleRecord) — nested record → N semantic chunks
 *    Uses VehicleChunkBuilder to split by question category:
 *    identity · performance · ownership_cost · rankings · safety ·
 *    features_trims · reviews · one chunk per maintenance interval ·
 *    one chunk per recall.
 *
 * doc_id    : vehicleId  (e.g. "bmw-m3-2025-competition")
 * chunkId   : vehicleId:chunkIndex  (e.g. "bmw-m3-2025-competition:5")
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class VehicleIngestionService {

    private final EmbeddingClient        embeddingClient;
    private final PgVectorStore          vectorStore;
    private final VehicleDocumentBuilder documentBuilder; // simple flat builder (existing)
    private final VehicleChunkBuilder    chunkBuilder;    // rich semantic builder (new)

    // ─── Simple ingestion (backward-compatible) ───────────────────────────────

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

    // ─── Rich ingestion (multi-chunk semantic) ────────────────────────────────

    /**
     * Ingests a RichVehicleRecord as multiple semantic chunks, one per question type.
     * Returns an IngestResult listing every stored chunkId and any per-chunk errors.
     *
     * Re-ingesting the same vehicle is safe — upsert uses ON CONFLICT (doc_id, chunk_index)
     * DO UPDATE so existing chunks are replaced in place, not duplicated.
     *
     * If you remove a sub-object between ingests (e.g. resolve all recalls),
     * the old recall chunks remain in the DB. Call deleteChunksByType(vehicleId, "recall")
     * before re-ingesting to clean them up.
     */
    public IngestResult ingestRichVehicle(RichVehicleRecord vehicle) {
        validateRichVehicle(vehicle);

        List<VehicleChunkBuilder.VehicleChunk> chunks = chunkBuilder.buildChunks(vehicle);
        log.info("Ingesting rich vehicle '{}' — {} semantic chunks", vehicle.vehicleId(), chunks.size());

        long         documentId = stableDocumentId(vehicle.vehicleId());
        List<VectorRecord> records  = new ArrayList<>();
        List<String>       chunkIds = new ArrayList<>();
        List<String>       errors   = new ArrayList<>();

        for (VehicleChunkBuilder.VehicleChunk chunk : chunks) {
            String chunkId = vehicle.vehicleId() + ":" + chunk.chunkIndex();
            try {
                List<Double> embedding = embed(chunk.text(), chunkId);
                records.add(new VectorRecord(documentId, chunk.chunkIndex(), chunkId, chunk.text(), embedding));
                chunkIds.add(chunkId);
                log.debug("  Embedded [{}] type={} len={}", chunkId, chunk.chunkType(), chunk.text().length());
            } catch (Exception e) {
                // One bad chunk should not abort the whole vehicle ingest — log and continue
                String msg = "Embedding failed for " + chunkId + ": " + e.getMessage();
                log.error(msg);
                errors.add(msg);
            }
        }

        if (records.isEmpty()) {
            throw new IllegalStateException(
                    "All chunks failed embedding for vehicleId=" + vehicle.vehicleId());
        }

        vectorStore.upsert(records);

        log.info("Rich vehicle '{}' — {}/{} chunks stored, {} errors",
                vehicle.vehicleId(), records.size(), chunks.size(), errors.size());

        return new IngestResult(vehicle.vehicleId(), chunkIds, errors);
    }

    // ─── Result record ────────────────────────────────────────────────────────

    public record IngestResult(
            String       vehicleId,
            List<String> storedChunkIds,
            List<String> errors
    ) {
        public boolean hasErrors()  { return !errors.isEmpty(); }
        public int     chunkCount() { return storedChunkIds.size(); }
    }

    // ─── Validation ───────────────────────────────────────────────────────────

    private void validateVehicle(VehicleRecord v) {
        if (v == null) throw new IllegalArgumentException("VehicleRecord cannot be null");
        requireId(v.vehicleId());
        if (v.make()  == null || v.make().isBlank())  throw new IllegalArgumentException("make is required for vehicleId="  + v.vehicleId());
        if (v.model() == null || v.model().isBlank()) throw new IllegalArgumentException("model is required for vehicleId=" + v.vehicleId());
        if (v.year() < 1886 || v.year() > 2100)      throw new IllegalArgumentException("year " + v.year() + " looks invalid for vehicleId=" + v.vehicleId());
    }

    private void validateRichVehicle(RichVehicleRecord v) {
        if (v == null) throw new IllegalArgumentException("RichVehicleRecord cannot be null");
        requireId(v.vehicleId());
        if (v.make()  == null || v.make().isBlank())  throw new IllegalArgumentException("make is required for vehicleId="  + v.vehicleId());
        if (v.model() == null || v.model().isBlank()) throw new IllegalArgumentException("model is required for vehicleId=" + v.vehicleId());
        if (v.year() < 1886 || v.year() > 2100)      throw new IllegalArgumentException("year " + v.year() + " looks invalid for vehicleId=" + v.vehicleId());
    }

    private void requireId(String vehicleId) {
        if (vehicleId == null || vehicleId.isBlank())
            throw new IllegalArgumentException("vehicleId is required");
        if (vehicleId.contains(":"))
            throw new IllegalArgumentException(
                    "vehicleId must not contain ':' — reserved for chunkId format. Got: " + vehicleId);
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    private List<Double> embed(String text, String chunkId) {
        List<Double> vec = embeddingClient.embed(text);
        if (vec == null || vec.isEmpty())
            throw new IllegalStateException("Embedding returned empty for chunkId=" + chunkId);
        return vec;
    }

    /** Deterministic stable ID from vehicleId — no timestamp drift on re-ingest.
     * HashCode added to have long values, abs func returns only positives.
     * */
    private long stableDocumentId(String vehicleId) {
        return Math.abs((long) vehicleId.hashCode());
    }
}