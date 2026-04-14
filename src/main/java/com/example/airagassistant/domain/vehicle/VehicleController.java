package com.example.airagassistant.domain.vehicle;

import com.example.airagassistant.agentic.AgentSessionRunner;
import com.example.airagassistant.domain.vehicle.service.VehicleRagService;
import com.example.airagassistant.rag.RagAnswerService;
import com.example.airagassistant.rag.RetrievalMode;
import com.example.airagassistant.rag.SearchHit;
import com.example.airagassistant.rag.ingestion.vehicle.VehicleIngestionService;
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
}