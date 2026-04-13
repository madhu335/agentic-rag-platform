package com.example.airagassistant.agentic.tools.vehicle;

import com.example.airagassistant.LlmClient;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Agentic tool: GenerateVehicleSummary
 *
 * Takes a list of raw spec chunks (from FetchVehicleSpecsTool) and asks the LLM
 * to write a concise, consumer-friendly vehicle summary paragraph.
 *
 * Input:  vehicleId + list of SpecChunk
 * Output: narrative summary string
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class GenerateVehicleSummaryTool {

    private final LlmClient llmClient;

    public record Input(String vehicleId, List<FetchVehicleSpecsTool.SpecChunk> chunks) {}

    public String execute(Input input) {
        if (input.chunks() == null || input.chunks().isEmpty()) {
            return "No specification data available for vehicle: " + input.vehicleId();
        }

        List<String> contextLines = IntStream.range(0, input.chunks().size())
                .mapToObj(i -> "[" + input.chunks().get(i).chunkId() + "] " + input.chunks().get(i).text())
                .toList();

        String prompt = buildPrompt(input.vehicleId(), contextLines);

        log.debug("GenerateVehicleSummaryTool — vehicleId='{}' chunks={}", input.vehicleId(), input.chunks().size());

        return llmClient.answer(prompt, contextLines);
    }

    private String buildPrompt(String vehicleId, List<String> contextLines) {
        return """
                You are an automotive content writer.
                
                Write a concise, consumer-friendly summary (2–4 sentences) for the vehicle below.
                Use ONLY the provided specification data. Do not invent facts.
                Focus on: powertrain, key performance figures, standout features, and price.
                Do NOT include citations or chunk IDs in the output.
                
                Vehicle ID: %s
                """.formatted(vehicleId);
    }
}