package com.example.airagassistant.agentic.multi.agents;

import com.example.airagassistant.agentic.multi.SubAgentResult;
import com.example.airagassistant.agentic.tools.vehicle.CompareVehiclesTool;
import com.example.airagassistant.agentic.tools.vehicle.EnrichVehicleDataTool;
import com.example.airagassistant.agentic.tools.vehicle.FetchVehicleSpecsTool;
import com.example.airagassistant.agentic.tools.vehicle.GenerateVehicleSummaryTool;
import com.example.airagassistant.domain.vehicle.VehicleRecord;
import com.example.airagassistant.rag.ingestion.vehicle.VehicleIngestionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Sub-agent: Vehicle
 *
 * Owns: vehicle spec retrieval, summary generation, comparison, enrichment.
 *
 * This agent has its own mini-router that decides which tool chain to run
 * based on the task and args. It does NOT use the LLM for internal planning —
 * keyword matching is sufficient because the supervisor already decomposed
 * the request into a vehicle-specific task.
 *
 * The key insight: the supervisor says "compare performance" and this agent
 * knows that means fetch_specs + compare. The supervisor doesn't need to
 * know the individual steps.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class VehicleSubAgent {

    private final FetchVehicleSpecsTool fetchSpecsTool;
    private final GenerateVehicleSummaryTool summaryTool;
    private final CompareVehiclesTool compareTool;
    private final EnrichVehicleDataTool enrichTool;
    private final VehicleIngestionService ingestionService;

    public SubAgentResult execute(String task, Map<String, Object> args) {
        String lowerTask = task != null ? task.toLowerCase() : "";

        // Route based on task + args
        if (hasArg(args, "vehicleIds") || lowerTask.contains("compare")) {
            return executeComparison(task, args);
        }

        if (lowerTask.contains("enrich") || lowerTask.contains("improve")) {
            return executeEnrichment(args);
        }

        // Default: fetch specs + generate summary
        return executeFetchAndSummarize(task, args);
    }

    // ─── Fetch + Summarize ────────────────────────────────────────────────

    private SubAgentResult executeFetchAndSummarize(String task, Map<String, Object> args) {
        String vehicleId = getStringArg(args, "vehicleId", null);
        if (vehicleId == null || vehicleId.isBlank()) {
            return SubAgentResult.failure("vehicle", "vehicleId is required");
        }

        log.info("VehicleSubAgent — fetch + summarize for '{}'", vehicleId);

        try {
            // Step 1: Fetch specs
            String question = getStringArg(args, "question", task);
            int topK = getIntArg(args, "topK", 3);

            List<FetchVehicleSpecsTool.SpecChunk> chunks =
                    fetchSpecsTool.execute(new FetchVehicleSpecsTool.Input(vehicleId, question, topK));

            if (chunks.isEmpty()) {
                return SubAgentResult.failure("vehicle",
                        "No spec data found for vehicle: " + vehicleId);
            }

            // Step 2: Generate summary
            String summary = summaryTool.execute(
                    new GenerateVehicleSummaryTool.Input(vehicleId, chunks));

            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("vehicleId", vehicleId);
            metadata.put("chunkCount", chunks.size());
            metadata.put("operation", "fetch_and_summarize");

            return SubAgentResult.success(
                    "vehicle", summary,
                    List.of(vehicleId + ":1"), 1.0, null, metadata
            );

        } catch (Exception e) {
            log.error("VehicleSubAgent — fetch+summarize failed for '{}': {}",
                    vehicleId, e.getMessage());
            return SubAgentResult.failure("vehicle",
                    "Failed to process vehicle " + vehicleId + ": " + e.getMessage());
        }
    }

    // ─── Comparison ───────────────────────────────────────────────────────

    private SubAgentResult executeComparison(String task, Map<String, Object> args) {
        List<String> vehicleIds = parseVehicleIds(args);
        if (vehicleIds.size() < 2) {
            // Fallback: if the supervisor only passed one vehicleId via docId,
            // we can't compare — need at least 2
            log.warn("VehicleSubAgent — comparison needs >= 2 vehicleIds, got {}", vehicleIds);
            return SubAgentResult.failure("vehicle",
                    "compare_vehicles requires at least 2 vehicleIds. Got: " + vehicleIds);
        }

        String question = getStringArg(args, "question", task);

        log.info("VehicleSubAgent — comparing {} on '{}'", vehicleIds, question);

        try {
            CompareVehiclesTool.ComparisonResult result =
                    compareTool.execute(new CompareVehiclesTool.Input(vehicleIds, question));

            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("vehicleIds", vehicleIds);
            metadata.put("operation", "compare");

            return SubAgentResult.success(
                    "vehicle", result.answer(),
                    vehicleIds.stream().map(id -> id + ":1").toList(),
                    1.0, null, metadata
            );

        } catch (Exception e) {
            log.error("VehicleSubAgent — comparison failed: {}", e.getMessage());
            return SubAgentResult.failure("vehicle",
                    "Comparison failed: " + e.getMessage());
        }
    }

    // ─── Enrichment ───────────────────────────────────────────────────────

    private SubAgentResult executeEnrichment(Map<String, Object> args) {
        String vehicleId = getStringArg(args, "vehicleId", null);
        if (vehicleId == null || vehicleId.isBlank()) {
            return SubAgentResult.failure("vehicle",
                    "vehicleId is required for enrichment");
        }

        log.info("VehicleSubAgent — enriching '{}'", vehicleId);

        try {
            // Build minimal VehicleRecord from args
            VehicleRecord vehicle = new VehicleRecord(
                    vehicleId,
                    getIntArg(args, "year", 0),
                    getStringArg(args, "make", ""),
                    getStringArg(args, "model", ""),
                    getStringArg(args, "trim", null),
                    getStringArg(args, "bodyStyle", null),
                    getStringArg(args, "engine", null),
                    getIntegerArg(args, "horsepower"),
                    getIntegerArg(args, "torque"),
                    getStringArg(args, "drivetrain", null),
                    getStringArg(args, "transmission", null),
                    getStringArg(args, "mpgCity", null),
                    getStringArg(args, "mpgHighway", null),
                    getStringArg(args, "msrp", null),
                    List.of(),
                    null  // summary intentionally null — enrichment generates it
            );

            VehicleRecord enriched = enrichTool.execute(vehicle);
            ingestionService.ingestVehicle(enriched);

            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("vehicleId", vehicleId);
            metadata.put("summary", enriched.summary());
            metadata.put("operation", "enrich");

            return SubAgentResult.success(
                    "vehicle",
                    "Vehicle '" + vehicleId + "' enriched and re-ingested. Summary: " + enriched.summary(),
                    List.of(vehicleId + ":1"), 1.0, null, metadata
            );

        } catch (Exception e) {
            log.error("VehicleSubAgent — enrichment failed for '{}': {}",
                    vehicleId, e.getMessage());
            return SubAgentResult.failure("vehicle",
                    "Enrichment failed: " + e.getMessage());
        }
    }

    // ─── Helpers ──────────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private List<String> parseVehicleIds(Map<String, Object> args) {
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
        return List.of();
    }

    private boolean hasArg(Map<String, Object> args, String key) {
        if (args == null) return false;
        Object v = args.get(key);
        if (v == null) return false;
        if (v instanceof String s) return !s.isBlank();
        return true;
    }

    private String getStringArg(Map<String, Object> args, String key, String fallback) {
        if (args == null) return fallback;
        Object v = args.get(key);
        if (v == null) return fallback;
        String s = String.valueOf(v).trim();
        return s.isEmpty() ? fallback : s;
    }

    private int getIntArg(Map<String, Object> args, String key, int fallback) {
        if (args == null) return fallback;
        Object v = args.get(key);
        if (v == null) return fallback;
        try { return Integer.parseInt(String.valueOf(v).trim()); }
        catch (NumberFormatException e) { return fallback; }
    }

    private Integer getIntegerArg(Map<String, Object> args, String key) {
        if (args == null) return null;
        Object v = args.get(key);
        if (v == null) return null;
        try { return Integer.parseInt(String.valueOf(v).trim()); }
        catch (NumberFormatException e) { return null; }
    }
}
