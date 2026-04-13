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
 * AgentTool: "fetch_vehicle_specs"
 *
 * Retrieves the top spec chunks for a vehicle from pgvector and stores them
 * in AgentSessionState.VehicleSnapshot for downstream tools
 * (generate_vehicle_summary, compare_vehicles).
 *
 * Plan step args:
 *   - vehicleId  (required) — e.g. "tesla-model3-2025-long-range"
 *   - question   (optional) — focus query; defaults to generic spec query
 *   - topK       (optional) — defaults to 3
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class FetchVehicleSpecsToolExecutor implements AgentTool {

    private final FetchVehicleSpecsTool fetchVehicleSpecsTool;

    @Override
    public String name() {
        return "fetch_vehicle_specs";
    }

    @Override
    public AgentSessionState execute(AgentSessionState state, PlanStep step) {
        Map<String, Object> args = step.args() != null ? step.args() : Map.of();

        String vehicleId = getString(args, "vehicleId",
                state.vehicle() != null ? state.vehicle().vehicleId() : null);
        if (vehicleId == null || vehicleId.isBlank()) {
            throw new IllegalArgumentException("fetch_vehicle_specs requires 'vehicleId' arg");
        }

        String question = getString(args, "question", state.currentUserRequest());
        int topK = getInt(args, "topK", 3);

        log.info("fetch_vehicle_specs — vehicleId='{}' topK={}", vehicleId, topK);

        List<FetchVehicleSpecsTool.SpecChunk> chunks =
                fetchVehicleSpecsTool.execute(new FetchVehicleSpecsTool.Input(vehicleId, question, topK));

        List<String> chunkIds = chunks.stream().map(FetchVehicleSpecsTool.SpecChunk::chunkId).toList();
        String rawText = chunks.stream()
                .map(FetchVehicleSpecsTool.SpecChunk::text)
                .reduce("", (a, b) -> a + "\n\n" + b)
                .trim();

        AgentSessionState.VehicleSnapshot snapshot = new AgentSessionState.VehicleSnapshot(
                vehicleId,
                state.vehicle() != null ? state.vehicle().vehicleIds() : null,
                null,   // summary not yet generated
                null,   // comparison not yet done
                chunkIds,
                rawText,
                AgentSessionState.VehicleStepStatus.SPECS_FETCHED
        );

        return withVehicle(state, snapshot);
    }

    // ─── helpers ──────────────────────────────────────────────────────────

    private AgentSessionState withVehicle(AgentSessionState s, AgentSessionState.VehicleSnapshot v) {
        return s.withVehicle(v);
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
}
