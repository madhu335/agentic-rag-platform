package com.example.airagassistant.rag.ingestion.vehicle;

import com.example.airagassistant.domain.vehicle.VehicleRecord;
import org.springframework.stereotype.Component;

import java.util.StringJoiner;

/**
 * Converts a VehicleRecord into a plain-text document suitable for embedding.
 *
 * Design notes:
 * - Uses "Field: value" format — clear field labels help semantic search
 *   associate queries like "what's the horsepower of the Model 3" with the right chunk.
 * - Adds a natural-language preamble sentence so the embedding captures the
 *   entity identity (year + make + model) even for fuzzy/conversational queries.
 * - The summary field is placed last because it often repeats other fields; keeping
 *   it at the end prevents it from dominating the embedding.
 */
@Component
public class VehicleDocumentBuilder {

    public String buildDocument(VehicleRecord v) {
        StringJoiner sj = new StringJoiner("\n");

        // Natural-language preamble — improves semantic match for conversational queries
        sj.add(v.year() + " " + v.make() + " " + v.model()
                + (v.trim() != null ? " " + v.trim() : "")
                + " vehicle specification.");

        sj.add("Vehicle ID: " + v.vehicleId());
        sj.add("Year: " + v.year());
        sj.add("Make: " + v.make());
        sj.add("Model: " + v.model());
        sj.add("Trim: " + safe(v.trim()));
        sj.add("Body Style: " + safe(v.bodyStyle()));
        sj.add("Engine: " + safe(v.engine()));
        sj.add("Horsepower: " + safeUnit(v.horsepower(), "hp"));
        sj.add("Torque: " + safeUnit(v.torque(), "lb-ft"));
        sj.add("Drivetrain: " + safe(v.drivetrain()));
        sj.add("Transmission: " + safe(v.transmission()));
        sj.add("Fuel Economy City: " + safe(v.mpgCity()) + " MPGe/MPG");
        sj.add("Fuel Economy Highway: " + safe(v.mpgHighway()) + " MPGe/MPG");
        sj.add("MSRP: " + safe(v.msrp()));

        if (v.features() != null && !v.features().isEmpty()) {
            sj.add("Key Features: " + String.join(", ", v.features()));
        }

        if (v.summary() != null && !v.summary().isBlank()) {
            sj.add("Summary: " + v.summary());
        }

        return sj.toString();
    }

    private String safe(Object o) {
        return (o == null) ? "N/A" : String.valueOf(o).trim();
    }

    private String safeUnit(Object o, String unit) {
        if (o == null) return "N/A";
        String val = String.valueOf(o).trim();
        return val.isEmpty() ? "N/A" : val + " " + unit;
    }
}