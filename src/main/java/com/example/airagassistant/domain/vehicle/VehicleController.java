package com.example.airagassistant.domain.vehicle;

import com.example.airagassistant.agentic.AgentSessionRunner;
import com.example.airagassistant.domain.vehicle.service.VehicleRagService;
import com.example.airagassistant.rag.RagAnswerService;
import com.example.airagassistant.rag.RetrievalMode;
import com.example.airagassistant.rag.SearchHit;
import com.example.airagassistant.rag.ingestion.vehicle.BulkVehicleIngestionService;
import com.example.airagassistant.rag.ingestion.vehicle.VehicleIngestionService;
import com.example.airagassistant.rag.retrieval.VehicleSummaryService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * Vehicle RAG REST API.
 * <p>
 * Endpoints:
 * <p>
 * POST /vehicles/ingest          — ingest a single vehicle
 * POST /vehicles/{id}/ask        — ask a question about one vehicle
 * POST /vehicles/ask             — ask a cross-vehicle question (fleet search)
 */
@RestController
@RequestMapping("/vehicles")
@RequiredArgsConstructor
public class VehicleController {

    private final VehicleIngestionService ingestionService;
    private final VehicleRagService vehicleRagService;
    private final AgentSessionRunner sessionRunner;
    private final VehicleSummaryService summaryService;
    private final BulkVehicleIngestionService bulkIngestionService;

    // ─── Request / Response DTOs ───────────────────────────────────────────

    public record VehicleAskRequest(
            String question,
            Integer topK,
            String mode   // optional: VECTOR | BM25 | HYBRID | HYBRID_RERANK  (default: HYBRID)
    ) {
    }

    public record VehicleAskResponse(
            String vehicleId,
            String answer,
            List<String> retrievedChunkIds,
            List<String> citedChunkIds,
            int usedChunks,
            Double bestScore
    ) {
    }

    public record FleetSearchRequest(
            String question,
            Integer topK
    ) {
    }

    public record FleetSearchHit(
            String chunkId,
            String vehicleId,
            int rank,          // 1-based position in results (clearer than raw RRF score)
            String excerpt     // first 200 chars of the chunk text
    ) {
    }

    // ─── Ingest ───────────────────────────────────────────────────────────

    /**
     * POST /vehicles/ingest
     * Body: VehicleRecord JSON
     */
    @PostMapping("/ingest")
    public String ingest(@RequestBody VehicleRecord vehicle) {
        ingestionService.ingestVehicle(vehicle);
        return "Vehicle ingested: " + vehicle.vehicleId();
    }
    // ── ADD to VehicleController.java ────────────────────────────────────────────
// 1. Add import at top:
//    import com.example.airagassistant.domain.vehicle.RichVehicleRecord;
//    import com.example.airagassistant.rag.ingestion.vehicle.VehicleIngestionService.IngestResult;
//
// 2. Add this endpoint alongside the existing POST /vehicles/ingest:

    /**
     * POST /vehicles/ingest/rich
     * Ingests a RichVehicleRecord as N semantic chunks (one per sub-object type).
     * Returns the list of stored chunkIds and any per-chunk errors.
     * <p>
     * Example response:
     * {
     * "vehicleId": "bmw-m3-2025-competition",
     * "chunkCount": 12,
     * "storedChunkIds": ["bmw-m3-2025-competition:1", ..., "bmw-m3-2025-competition:21"],
     * "errors": [],
     * "hasErrors": false
     * }
     */
    @PostMapping("/ingest/rich")
    public VehicleIngestionService.IngestResult ingestRich(@RequestBody RichVehicleRecord vehicle) {
        return ingestionService.ingestRichVehicle(vehicle);
    }
    // ─── Single-vehicle ask ────────────────────────────────────────────────

    /**
     * POST /vehicles/{vehicleId}/ask
     * Ask a question scoped to a single ingested vehicle.
     * <p>
     * Example:
     * POST /vehicles/tesla-model3-2025-long-range/ask
     * { "question": "What is the horsepower and drivetrain?" }
     */
    @PostMapping("/{vehicleId}/ask")
    public VehicleAskResponse askVehicle(
            @PathVariable String vehicleId,
            @RequestBody VehicleAskRequest req
    ) {
        int topK = (req.topK() == null || req.topK() <= 0) ? 5 : req.topK();
        RetrievalMode mode = parseMode(req.mode());

        RagAnswerService.RagResult result = sessionRunner.runRagWithSession(
                req.question(),
                vehicleId,
                () -> vehicleRagService.askVehicle(
                        vehicleId,
                        req.question(),
                        topK,
                        mode
                )
        );

        return new VehicleAskResponse(
                vehicleId,
                result.answer(),
                result.retrievedChunkIds(),
                result.citedChunkIds(),
                result.usedChunks(),
                result.bestScore()
        );
    }

    // ─── Cross-vehicle (fleet) ask ─────────────────────────────────────────

    /**
     * POST /vehicles/ask
     * Ask a question across ALL ingested vehicles.
     * Returns the top matching chunks from any vehicle.
     * <p>
     * Example:
     * POST /vehicles/ask
     * { "question": "Which vehicles have over 400 horsepower?", "topK": 5 }
     */
    @PostMapping("/ask")
    public List<FleetSearchHit> askFleet(@RequestBody FleetSearchRequest req) {
        int topK = (req.topK() == null || req.topK() <= 0) ? 5 : req.topK();

        List<SearchHit> hits = vehicleRagService.searchAllVehicles(req.question(), topK);

        var result = new java.util.ArrayList<FleetSearchHit>();
        for (int i = 0; i < hits.size(); i++) {
            SearchHit hit = hits.get(i);
            result.add(new FleetSearchHit(
                    hit.record().id(),
                    extractVehicleId(hit.record().id()),
                    i + 1,                           // rank: 1 = best match
                    excerpt(hit.record().text(), 200)
            ));
        }
        return result;
    }

    // ─── Helpers ──────────────────────────────────────────────────────────

    private RetrievalMode parseMode(String mode) {
        if (mode == null || mode.isBlank()) return RetrievalMode.HYBRID;
        try {
            return RetrievalMode.valueOf(mode.toUpperCase());
        } catch (IllegalArgumentException e) {
            return RetrievalMode.HYBRID;
        }
    }

    /**
     * chunkId format is  "vehicleId:chunkIndex"  e.g. "tesla-model3-2025-long-range:1"
     */
    private String extractVehicleId(String chunkId) {
        int idx = chunkId.lastIndexOf(':');
        return idx > 0 ? chunkId.substring(0, idx) : chunkId;
    }

    private String excerpt(String text, int maxLen) {
        if (text == null) return "";
        return text.length() <= maxLen ? text : text.substring(0, maxLen) + "…";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TWO-TIER FLEET SEARCH
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * POST /vehicles/ask/fleet
     * Two-tier fleet search: searches vehicle_summaries first (Tier 1),
     * then retrieves detailed chunks from candidates (Tier 2).
     *
     * At 90K vehicles: ~6ms total instead of 500ms-2s with single-tier.
     *
     * Example:
     *   POST /vehicles/ask/fleet
     *   { "question": "Which vehicle has the best performance?", "topK": 5 }
     */
    @PostMapping("/ask/fleet")
    public TwoTierFleetResponse askFleetTwoTier(@RequestBody FleetSearchRequest req) {
        int topK = (req.topK() == null || req.topK() <= 0) ? 5 : req.topK();

        VehicleSummaryService.TwoTierResult result =
                summaryService.searchTwoTier(req.question(), 10, topK);

        var summaryHits = result.summaryHits().stream()
                .map(h -> new SummaryHit(
                        h.vehicleId(),
                        h.year() + " " + h.make() + " " + h.model()
                                + (h.trim() != null ? " " + h.trim() : ""),
                        h.score(),
                        h.chunkCount()
                ))
                .toList();

        var detailHits = new java.util.ArrayList<FleetSearchHit>();
        for (int i = 0; i < result.detailHits().size(); i++) {
            SearchHit hit = result.detailHits().get(i);
            detailHits.add(new FleetSearchHit(
                    hit.record().id(),
                    extractVehicleId(hit.record().id()),
                    i + 1,
                    excerpt(hit.record().text(), 200)
            ));
        }

        return new TwoTierFleetResponse(
                summaryHits,
                detailHits,
                result.tier1Ms(),
                result.totalMs()
        );
    }

    public record SummaryHit(
            String vehicleId,
            String vehicleName,
            double score,
            int chunkCount
    ) {}

    public record TwoTierFleetResponse(
            List<SummaryHit> candidates,
            List<FleetSearchHit> results,
            long tier1Ms,
            long totalMs
    ) {}

    // ═══════════════════════════════════════════════════════════════════════
    // PARADEDB HYBRID SEARCH
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * POST /vehicles/ask/hybrid
     * Hybrid search combining vector similarity + BM25 (ParadeDB) in a
     * single SQL query. No app-level RRF — fusion happens in Postgres.
     *
     * vectorWeight: 0.0 = pure BM25, 1.0 = pure vector, 0.7 = default blend
     *
     * Example:
     *   POST /vehicles/ask/hybrid
     *   { "question": "sports sedan track performance", "topK": 5, "vectorWeight": 0.7 }
     */
    @PostMapping("/ask/hybrid")
    public List<HybridSearchHit> askHybrid(@RequestBody HybridSearchRequest req) {
        int topK = (req.topK() == null || req.topK() <= 0) ? 5 : req.topK();
        double vectorWeight = (req.vectorWeight() == null) ? 0.7 : req.vectorWeight();

        List<SearchHit> hits = summaryService.hybridSearchVehicles(
                req.question(), topK, vectorWeight);

        var result = new java.util.ArrayList<HybridSearchHit>();
        for (int i = 0; i < hits.size(); i++) {
            SearchHit hit = hits.get(i);
            result.add(new HybridSearchHit(
                    hit.record().id(),
                    extractVehicleId(hit.record().id()),
                    i + 1,
                    hit.score(),
                    excerpt(hit.record().text(), 200)
            ));
        }
        return result;
    }

    public record HybridSearchRequest(
            String question,
            Integer topK,
            Double vectorWeight     // 0.0 = pure BM25, 1.0 = pure vector
    ) {}

    public record HybridSearchHit(
            String chunkId,
            String vehicleId,
            int rank,
            double hybridScore,
            String excerpt
    ) {}

    // ═══════════════════════════════════════════════════════════════════════
    // BULK INGESTION
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * POST /vehicles/ingest/bulk
     * Batch-ingest multiple rich vehicles in one call.
     * Processes in pages of 50: chunk → batch embed → batch upsert → populate summaries.
     *
     * Example:
     *   POST /vehicles/ingest/bulk
     *   [{ "vehicleId": "...", ... }, { "vehicleId": "...", ... }]
     */
    @PostMapping("/ingest/bulk")
    public BulkVehicleIngestionService.BulkIngestResult ingestBulk(
            @RequestBody List<RichVehicleRecord> vehicles) {
        return bulkIngestionService.ingestBulk(vehicles);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ADMIN / OBSERVABILITY
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * GET /vehicles/admin/summaries
     * View all vehicle summaries (two-tier Tier 1 data).
     * Useful for verifying summary population after ingestion.
     */
    @GetMapping("/admin/summaries")
    public List<VehicleSummaryService.VehicleSummaryInfo> getVehicleSummaries() {
        return summaryService.getAllSummaries();
    }
}