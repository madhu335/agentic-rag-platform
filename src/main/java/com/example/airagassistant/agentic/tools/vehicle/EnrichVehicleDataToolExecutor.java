package com.example.airagassistant.agentic.tools.vehicle;

import com.example.airagassistant.agentic.dto.AgentTool;
import com.example.airagassistant.agentic.dto.PlanStep;
import com.example.airagassistant.agentic.state.AgentSessionState;
import com.example.airagassistant.domain.vehicle.VehicleRecord;
import com.example.airagassistant.rag.ingestion.vehicle.VehicleIngestionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

/**
 * AgentTool: "enrich_vehicle_data"
 *
 * Auto-generates a summary for a VehicleRecord using the LLM,
 * then re-ingests the enriched record so the vector store is updated.
 *
 * This tool is designed to be called BEFORE ingestion when summary is missing,
 * or as a standalone re-enrichment pass on an already-ingested vehicle.
 *
 * Plan step args:
 *   - vehicleId    (required)
 *   - year         (required if building from scratch)
 *   - make         (required)
 *   - model        (required)
 *   - trim         (optional)
 *   - engine       (optional)
 *   - horsepower   (optional)
 *   - msrp         (optional)
 *   ... (all VehicleRecord fields can be supplied; missing ones default to null)
 *
 * Minimal usage — planner can pass just vehicleId if the vehicle is already
 * in the vector store; the tool will fetch its raw text, regenerate the
 * summary from it, and upsert the improved embedding.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class EnrichVehicleDataToolExecutor implements AgentTool {

    private final EnrichVehicleDataTool enrichVehicleDataTool;
    private final VehicleIngestionService vehicleIngestionService;

    @Override
    public String name() {
        return "enrich_vehicle_data";
    }

    @Override
    public AgentSessionState execute(AgentSessionState state, PlanStep step) {
        Map<String, Object> args = step.args() != null ? step.args() : Map.of();

        String vehicleId = getString(args, "vehicleId", null);
        if (vehicleId == null || vehicleId.isBlank()) {
            throw new IllegalArgumentException("enrich_vehicle_data requires 'vehicleId' arg");
        }

        log.info("enrich_vehicle_data — vehicleId='{}'", vehicleId);

        // Build a VehicleRecord from whatever args are present
        // (fields not supplied will be null / defaults)
        VehicleRecord vehicle = buildVehicleFromArgs(vehicleId, args);

        // Enrich: generates summary via LLM if blank
        VehicleRecord enriched = enrichVehicleDataTool.execute(vehicle);

        // Re-ingest with the improved summary so the embedding is updated
        vehicleIngestionService.ingestVehicle(enriched);

        log.info("enrich_vehicle_data — '{}' re-ingested with summary: {}",
                vehicleId, enriched.summary());

        AgentSessionState.VehicleSnapshot updatedVehicle = new AgentSessionState.VehicleSnapshot(
                vehicleId,
                state.vehicle() != null ? state.vehicle().vehicleIds() : null,
                enriched.summary(),
                state.vehicle() != null ? state.vehicle().comparisonResult() : null,
                state.vehicle() != null ? state.vehicle().specChunkIds() : null,
                state.vehicle() != null ? state.vehicle().rawSpecText() : null,
                AgentSessionState.VehicleStepStatus.ENRICHED
        );

        // Surface the enriched summary as the agent's research result
        AgentSessionState.ResearchSnapshot updatedResearch = new AgentSessionState.ResearchSnapshot(
                "Vehicle '" + vehicleId + "' enriched and re-ingested. Summary: " + enriched.summary(),
                List.of(vehicleId + ":1"),
                1.0,
                null,
                List.of()
        );

        return state.withVehicle(updatedVehicle).withResearch(updatedResearch);
    }

    // ─── helpers ──────────────────────────────────────────────────────────

    private VehicleRecord buildVehicleFromArgs(String vehicleId, Map<String, Object> args) {
        return new VehicleRecord(
                vehicleId,
                getInt(args, "year", 0),
                getString(args, "make", ""),
                getString(args, "model", ""),
                getString(args, "trim", null),
                getString(args, "bodyStyle", null),
                getString(args, "engine", null),
                getIntOrNull(args, "horsepower"),
                getIntOrNull(args, "torque"),
                getString(args, "drivetrain", null),
                getString(args, "transmission", null),
                getString(args, "mpgCity", null),
                getString(args, "mpgHighway", null),
                getString(args, "msrp", null),
                List.of(),  // features — not typically in plan args
                null        // summary — intentionally blank so EnrichVehicleDataTool generates it
        );
    }

    private String getString(Map<String, Object> args, String key, String fallback) {
        Object v = args.get(key);
        if (v == null) return fallback;
        String s = String.valueOf(v).trim();
        return s.isEmpty() ? fallback : s;
    }

    private int getInt(Map<String, Object> args, String key, int fallback) {
        Object v = args.get(key);
        if (v == null) return fallback;
        try { return Integer.parseInt(String.valueOf(v).trim()); }
        catch (NumberFormatException e) { return fallback; }
    }

    private Integer getIntOrNull(Map<String, Object> args, String key) {
        Object v = args.get(key);
        if (v == null) return null;
        try { return Integer.parseInt(String.valueOf(v).trim()); }
        catch (NumberFormatException e) { return null; }
    }
}
