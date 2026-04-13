package com.example.airagassistant.agentic.tools.vehicle;

import com.example.airagassistant.agentic.dto.AgentTool;
import com.example.airagassistant.agentic.dto.PlanStep;
import com.example.airagassistant.agentic.state.AgentSessionState;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

/**
 * AgentTool: "generate_vehicle_summary"
 *
 * Reads spec chunks from the VehicleSnapshot already in session state
 * (populated by fetch_vehicle_specs) and calls the LLM to produce a
 * consumer-friendly narrative summary.
 *
 * Typical planner sequence:
 *   1. fetch_vehicle_specs   → args: vehicleId
 *   2. generate_vehicle_summary  (no additional args needed)
 *
 * The generated summary is stored back into VehicleSnapshot.summary
 * AND into ResearchSnapshot.summary so buildResponse() can surface it.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class GenerateVehicleSummaryToolExecutor implements AgentTool {

    private final GenerateVehicleSummaryTool generateVehicleSummaryTool;
    private final FetchVehicleSpecsTool fetchVehicleSpecsTool;

    @Override
    public String name() {
        return "generate_vehicle_summary";
    }

    @Override
    public AgentSessionState execute(AgentSessionState state, PlanStep step) {
        Map<String, Object> args = step.args() != null ? step.args() : Map.of();

        // vehicleId: prefer arg, fall back to whatever fetch_vehicle_specs put in state
        String vehicleId = getString(args, "vehicleId",
                state.vehicle() != null ? state.vehicle().vehicleId() : null);
        if (vehicleId == null || vehicleId.isBlank()) {
            throw new IllegalArgumentException("generate_vehicle_summary requires 'vehicleId' (arg or prior fetch step)");
        }

        log.info("generate_vehicle_summary — vehicleId='{}'", vehicleId);

        // If we already have spec chunks in state, reuse them; otherwise fetch on-demand
        List<FetchVehicleSpecsTool.SpecChunk> chunks;
        if (state.vehicle() != null
                && state.vehicle().specChunkIds() != null
                && !state.vehicle().specChunkIds().isEmpty()
                && state.vehicle().rawSpecText() != null) {

            // reconstruct SpecChunk list from snapshot
            chunks = state.vehicle().specChunkIds().stream()
                    .map(id -> new FetchVehicleSpecsTool.SpecChunk(id,
                            state.vehicle().rawSpecText(), 1.0))
                    .toList();
        } else {
            chunks = fetchVehicleSpecsTool.execute(
                    new FetchVehicleSpecsTool.Input(vehicleId, state.currentUserRequest(), 3));
        }

        String summary = generateVehicleSummaryTool.execute(
                new GenerateVehicleSummaryTool.Input(vehicleId, chunks));

        // Write summary into VehicleSnapshot
        AgentSessionState.VehicleSnapshot updatedVehicle = new AgentSessionState.VehicleSnapshot(
                vehicleId,
                state.vehicle() != null ? state.vehicle().vehicleIds() : null,
                summary,
                state.vehicle() != null ? state.vehicle().comparisonResult() : null,
                state.vehicle() != null ? state.vehicle().specChunkIds() : null,
                state.vehicle() != null ? state.vehicle().rawSpecText() : null,
                AgentSessionState.VehicleStepStatus.SUMMARY_GENERATED
        );

        // Also surface in ResearchSnapshot so buildResponse() returns it to the caller
        AgentSessionState.ResearchSnapshot updatedResearch = new AgentSessionState.ResearchSnapshot(
                summary,
                List.of(vehicleId + ":1"),
                1.0,
                null,
                List.of()
        );

        return state.withVehicle(updatedVehicle).withResearch(updatedResearch);
    }

    private String getString(Map<String, Object> args, String key, String fallback) {
        Object v = args.get(key);
        if (v == null) return fallback;
        String s = String.valueOf(v).trim();
        return s.isEmpty() ? fallback : s;
    }
}
