package com.example.airagassistant.agentic.tools.vehicle;

import com.example.airagassistant.LlmClient;
import com.example.airagassistant.domain.vehicle.service.VehicleRagService;
import com.example.airagassistant.rag.SearchHit;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Agentic tool: CompareVehicles
 *
 * Compares two or more vehicles on a specific dimension (e.g. "horsepower",
 * "price vs range", "family-friendly features").
 *
 * Strategy:
 *   1. For each vehicleId, fetch its top spec chunk via targeted vector search.
 *   2. Feed all chunks as context to the LLM with a comparison prompt.
 *
 * Input:  list of vehicleIds + comparison question/dimension
 * Output: LLM-generated comparison text
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class CompareVehiclesTool {

    private final VehicleRagService vehicleRagService;
    private final LlmClient llmClient;

    public record Input(List<String> vehicleIds, String comparisonQuestion) {}
    public record ComparisonResult(String answer, List<String> vehicleIds) {}

    public ComparisonResult execute(Input input) {
        if (input.vehicleIds() == null || input.vehicleIds().size() < 2) {
            throw new IllegalArgumentException("At least 2 vehicleIds required for comparison");
        }
        if (input.comparisonQuestion() == null || input.comparisonQuestion().isBlank()) {
            throw new IllegalArgumentException("comparisonQuestion is required");
        }

        log.debug("CompareVehiclesTool — vehicles={} question='{}'",
                input.vehicleIds(), input.comparisonQuestion());

        // Fetch top spec chunk for each vehicle
        List<String> contextChunks = new ArrayList<>();
        for (String vehicleId : input.vehicleIds()) {
            List<SearchHit> hits = vehicleRagService.searchAllVehicles(
                    vehicleId + " " + input.comparisonQuestion(), 1);

            if (!hits.isEmpty()) {
                String chunkId = hits.get(0).record().id();
                String text    = hits.get(0).record().text();
                contextChunks.add("[" + chunkId + "] " + text);
            } else {
                contextChunks.add("[" + vehicleId + "] No data available.");
            }
        }

        String vehicleList = String.join(", ", input.vehicleIds());
        String prompt = buildComparisonPrompt(vehicleList, input.comparisonQuestion());

        String answer = llmClient.answer(prompt, contextChunks);

        return new ComparisonResult(answer, input.vehicleIds());
    }

    private String buildComparisonPrompt(String vehicleList, String question) {
        return """
                You are an automotive analyst.
                
                Compare the following vehicles: %s
                
                Focus on: %s
                
                Rules:
                - Use ONLY the provided spec data — do not invent figures.
                - Be concise: 3–5 sentences total.
                - State clear winners/trade-offs where the data supports it.
                - Do NOT include chunk IDs or citation markers in the output.
                """.formatted(vehicleList, question);
    }
}