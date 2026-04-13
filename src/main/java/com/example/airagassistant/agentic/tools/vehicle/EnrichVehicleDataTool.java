package com.example.airagassistant.agentic.tools.vehicle;

import com.example.airagassistant.LlmClient;
import com.example.airagassistant.domain.vehicle.VehicleRecord;
import com.example.airagassistant.rag.ingestion.vehicle.VehicleDocumentBuilder;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Agentic tool: EnrichVehicleData
 *
 * Auto-generates or improves the `summary` field of a VehicleRecord before ingestion.
 * This is useful when the incoming data has a blank or low-quality summary.
 *
 * Workflow:
 *   1. Build the structured document text from the vehicle's fields.
 *   2. Ask the LLM to write a single-sentence summary.
 *   3. Return a new VehicleRecord with the enriched summary.
 *
 * Input:  VehicleRecord (summary may be null/blank)
 * Output: VehicleRecord with summary populated
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class EnrichVehicleDataTool {

    private final LlmClient llmClient;
    private final VehicleDocumentBuilder documentBuilder;

    public VehicleRecord execute(VehicleRecord vehicle) {
        if (vehicle.summary() != null && !vehicle.summary().isBlank()) {
            log.debug("EnrichVehicleDataTool — '{}' already has summary, skipping", vehicle.vehicleId());
            return vehicle;
        }

        log.info("EnrichVehicleDataTool — generating summary for '{}'", vehicle.vehicleId());

        String specText = documentBuilder.buildDocument(vehicle);

        String prompt = """
                You are an automotive copywriter.
                Write ONE concise sentence (max 25 words) summarising this vehicle's key selling points.
                Use only the data provided. Output only the sentence — no preamble, no punctuation at start.
                """;

        String generatedSummary = llmClient.answer(prompt, List.of(specText));
        String cleaned = generatedSummary == null ? "" : generatedSummary.trim();

        log.info("EnrichVehicleDataTool — generated summary for '{}': {}", vehicle.vehicleId(), cleaned);

        // Return a new record (VehicleRecord is immutable) with the enriched summary
        return new VehicleRecord(
                vehicle.vehicleId(),
                vehicle.year(),
                vehicle.make(),
                vehicle.model(),
                vehicle.trim(),
                vehicle.bodyStyle(),
                vehicle.engine(),
                vehicle.horsepower(),
                vehicle.torque(),
                vehicle.drivetrain(),
                vehicle.transmission(),
                vehicle.mpgCity(),
                vehicle.mpgHighway(),
                vehicle.msrp(),
                vehicle.features(),
                cleaned
        );
    }
}