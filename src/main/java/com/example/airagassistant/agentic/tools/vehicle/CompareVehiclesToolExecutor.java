package com.example.airagassistant.agentic.tools.vehicle;

import com.example.airagassistant.agentic.dto.AgentTool;
import com.example.airagassistant.agentic.dto.PlanStep;
import com.example.airagassistant.agentic.state.AgentSessionState;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

@Slf4j
@Component
@RequiredArgsConstructor
public class CompareVehiclesToolExecutor implements AgentTool {

    private final CompareVehiclesTool compareVehiclesTool;

    @Override
    public String name() {
        return "compare_vehicles";
    }

    @Override
    public AgentSessionState execute(AgentSessionState state, PlanStep step) {
        Map<String, Object> args = step.args() != null ? step.args() : Map.of();

        List<String> vehicleIds = parseVehicleIds(args, state);
        if (vehicleIds.size() < 2) {
            throw new IllegalArgumentException("compare_vehicles requires at least 2 vehicleIds");
        }

        String question = getString(args, "question", state.currentUserRequest());

        log.info("compare_vehicles — vehicles={} question='{}'", vehicleIds, question);

        CompareVehiclesTool.ComparisonResult result =
                compareVehiclesTool.execute(new CompareVehiclesTool.Input(vehicleIds, question));

        AgentSessionState.VehicleSnapshot updatedVehicle = new AgentSessionState.VehicleSnapshot(
                vehicleIds.get(0),
                vehicleIds,
                null,
                result.answer(),
                null,
                null,
                AgentSessionState.VehicleStepStatus.COMPARISON_DONE
        );

        AgentSessionState.ResearchSnapshot updatedResearch = new AgentSessionState.ResearchSnapshot(
                result.answer(),
                vehicleIds.stream().map(id -> id + ":1").toList(),
                1.0,
                null,
                List.of()
        );

        return state.withVehicle(updatedVehicle).withResearch(updatedResearch);
    }

    @SuppressWarnings("unchecked")
    private List<String> parseVehicleIds(Map<String, Object> args, AgentSessionState state) {
        Object raw = args.get("vehicleIds");

        if (raw instanceof List<?> list) {
            return list.stream().map(String::valueOf).toList();
        }

        if (raw instanceof String s && !s.isBlank()) {
            return Arrays.stream(s.split(","))
                    .map(String::trim)
                    .filter(v -> !v.isEmpty())
                    .toList();
        }

        if (state.vehicle() != null && state.vehicle().vehicleIds() != null) {
            return state.vehicle().vehicleIds();
        }

        return List.of();
    }

    private String getString(Map<String, Object> args, String key, String fallback) {
        Object v = args.get(key);
        if (v == null) return fallback;
        String s = String.valueOf(v).trim();
        return s.isEmpty() ? fallback : s;
    }
}
