package com.example.airagassistant.domain.vehicle.service;

import com.example.airagassistant.rag.SearchHit;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Collapses near-duplicate variants of the same model so they don't crowd a fleet result.
 *
 * EPA lists one vehicle per powertrain/wheel/model-year combination: 46 Rivian R1T
 * variants, 64 Silverados. Without collapsing, "truck with good off-road capability"
 * fills top-K with R1Ts that differ only in wheel size, and "family car with low
 * running costs" returns three Corolla hatchbacks before any other model.
 *
 * Family = everything before the model year in the vehicle id, i.e. make + model:
 *   toyota-corolla-2024-hatchback-fwd-...  -> toyota-corolla
 *   rivian-r1t-2025-adventure              -> rivian-r1t
 * Model year is deliberately excluded, so 2024 and 2025 of the same model are one family.
 *
 * Within a family the highest-ranked vehicle represents it; the ranking already
 * expresses which variant best matches the question.
 */
public final class VariantCollapser {

    private VariantCollapser() {}

    private static final Pattern FAMILY = Pattern.compile("^(.*?)-(?:19|20)\\d{2}(?:-|$)");

    /** "toyota-corolla-2024-hatchback-fwd" -> "toyota-corolla"; ids without a year are their own family. */
    public static String familyOf(String vehicleId) {
        if (vehicleId == null) return "";
        Matcher m = FAMILY.matcher(vehicleId);
        return m.find() ? m.group(1) : vehicleId;
    }

    /**
     * Keeps only chunks of the first-seen (best-ranked) vehicle of each family, in order.
     * If that leaves fewer than {@code minResults} chunks, refills from the dropped chunks
     * in rank order, so collapsing never shrinks a result below what the caller needs.
     */
    public static List<SearchHit> collapse(List<SearchHit> hits, int minResults) {
        if (hits == null || hits.isEmpty()) return hits;

        Map<String, String> representative = new HashMap<>();   // family -> chosen vehicle id
        List<SearchHit> kept = new ArrayList<>();
        List<SearchHit> dropped = new ArrayList<>();

        for (SearchHit hit : hits) {
            String vehicle = vehicleIdOf(hit.record().id());
            String chosen = representative.putIfAbsent(familyOf(vehicle), vehicle);
            if (chosen == null || chosen.equals(vehicle)) {
                kept.add(hit);
            } else {
                dropped.add(hit);
            }
        }

        for (SearchHit hit : dropped) {
            if (kept.size() >= minResults) break;
            kept.add(hit);
        }
        return kept;
    }

    /** "bmw-m3-2025-competition:21" -> "bmw-m3-2025-competition" */
    static String vehicleIdOf(String chunkId) {
        if (chunkId == null) return "";
        int idx = chunkId.lastIndexOf(':');
        return idx >= 0 ? chunkId.substring(0, idx) : chunkId;
    }
}
