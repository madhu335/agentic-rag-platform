package com.example.airagassistant.rag.ingestion.vehicle;

import com.example.airagassistant.domain.vehicle.MaintenanceInterval;
import com.example.airagassistant.domain.vehicle.RankingRecord;
import com.example.airagassistant.domain.vehicle.ReviewSnippet;
import com.example.airagassistant.domain.vehicle.RichVehicleRecord;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.StringJoiner;

/**
 * Converts a RichVehicleRecord into a list of typed semantic chunks.
 *
 * Each chunk is:
 *  - scoped to one question category (performance, cost, rankings, etc.)
 *  - prefixed with the identity anchor so the LLM always knows which vehicle it's reading
 *  - written in natural prose so embeddings match conversational queries
 *
 * chunk_index assignment:
 *   1 = identity
 *   2 = performance
 *   3 = ownership cost
 *   4 = rankings
 *   5 = safety
 *   6 = features + trims
 *   7 = reviews
 *   10+ = maintenance intervals (one per milestone)
 *   20+ = recall records (one per recall)
 */
@Component
public class VehicleChunkBuilder {

    public record VehicleChunk(
            String vehicleId,
            int    chunkIndex,
            String chunkType,
            String text
    ) {}

    public List<VehicleChunk> buildChunks(RichVehicleRecord v) {
        List<VehicleChunk> chunks = new ArrayList<>();

        // Anchor: repeated in every chunk so retrieval never loses vehicle identity
        String anchor = v.year() + " " + v.make() + " " + v.model()
                + (v.trim() != null ? " " + v.trim() : "") + " "
                + safe(v.vehicleClass()) + ". ";

        chunks.add(chunk(v, 1,  "identity",       buildIdentity(anchor, v)));
        chunks.add(chunk(v, 2,  "performance",     buildPerformance(anchor, v)));
        chunks.add(chunk(v, 3,  "ownership_cost",  buildOwnershipCost(anchor, v)));

        if (v.rankings() != null && !v.rankings().isEmpty()) {
            chunks.add(chunk(v, 4, "rankings", buildRankings(anchor, v.rankings())));
        }

        if (v.safety() != null) {
            chunks.add(chunk(v, 5, "safety", buildSafety(anchor, v)));
        }

        if ((v.features() != null && !v.features().isEmpty())
                || (v.trimLevels() != null && !v.trimLevels().isEmpty())) {
            chunks.add(chunk(v, 6, "features_trims", buildFeaturesAndTrims(anchor, v)));
        }

        if (v.reviews() != null && !v.reviews().isEmpty()) {
            chunks.add(chunk(v, 7, "reviews", buildReviews(anchor, v.reviews())));
        }

        // Maintenance — one chunk per interval so "30k service cost" retrieves exactly that row
        if (v.maintenanceSchedule() != null) {
            int idx = 10;
            for (MaintenanceInterval interval : v.maintenanceSchedule()) {
                chunks.add(chunk(v, idx++, "maintenance", buildMaintenanceInterval(anchor, interval)));
            }
        }

        // Recalls — one chunk per recall for precise retrieval
        if (v.recalls() != null) {
            int idx = 20;
            for (var recall : v.recalls()) {
                String text = anchor + "Recall " + recall.recallId()
                        + " (" + recall.date() + "): " + recall.component() + ". "
                        + recall.description() + " Remedy: " + recall.remedy() + ". "
                        + (recall.remedyAvailable() ? "Remedy is available." : "Remedy not yet available.");
                chunks.add(chunk(v, idx++, "recall", text));
            }
        }

        return chunks;
    }

    // ── Chunk builders ────────────────────────────────────────────────────────

    private String buildIdentity(String anchor, RichVehicleRecord v) {
        return anchor
                + "Body style: " + safe(v.bodyStyle()) + ". "
                + "Class: " + safe(v.vehicleClass()) + ". "
                + "Fuel type: " + safe(v.fuelType()) + ". "
                + (v.exteriorColor() != null ? "Exterior: " + v.exteriorColor() + ". " : "")
                + "MSRP: " + safe(v.msrp()) + ". "
                + "Invoice: " + safe(v.invoicePrice()) + ".";
    }

    private String buildPerformance(String anchor, RichVehicleRecord v) {
        StringJoiner sj = new StringJoiner(" ");
        sj.add(anchor + "Performance specs:");
        sj.add("Engine: " + safe(v.engine()) + ".");
        sj.add("Horsepower: " + safeUnit(v.horsepower(), "hp") + ".");
        sj.add("Torque: " + safeUnit(v.torque(), "lb-ft") + ".");
        if (v.zeroToSixty() != null) sj.add("0-60 mph: " + v.zeroToSixty() + " seconds.");
        if (v.topSpeed() != null)    sj.add("Top speed: " + v.topSpeed() + " mph.");
        sj.add("Drivetrain: " + safe(v.drivetrain()) + ".");
        sj.add("Transmission: " + safe(v.transmission()) + ".");
        sj.add("Fuel economy: " + safe(v.mpgCity()) + " city / " + safe(v.mpgHighway()) + " highway MPG.");
        if (v.electricRangeMiles() != null) sj.add("Electric range: " + v.electricRangeMiles() + " miles.");
        return sj.toString();
    }

    private String buildOwnershipCost(String anchor, RichVehicleRecord v) {
        if (v.ownershipCost() == null) return anchor + "Ownership cost data not available.";
        var c = v.ownershipCost();
        return anchor + "5-year ownership costs: "
                + "Fuel $" + c.fuelCostPerYear() + "/year. "
                + "Insurance $" + c.insurancePerYear() + "/year. "
                + "Maintenance $" + c.maintenancePerYear() + "/year. "
                + "5-year depreciation: $" + c.depreciationFiveYear() + ". "
                + "Total 5-year cost: $" + c.totalFiveYear() + ". "
                + "Resale value after 3 years: " + c.resaleValuePct() + "% of MSRP ("
                + c.resaleRating() + ").";
    }

    private String buildRankings(String anchor, List<RankingRecord> rankings) {
        // Key insight: convert rank numbers to narrative prose.
        // "ranked 2nd of 18 in sports sedans" embeds far better than "rank:2 total:18"
        // because the embedding model learned from natural language, not data tables.
        StringJoiner sj = new StringJoiner(" ");
        sj.add(anchor + "Industry rankings and awards:");
        for (RankingRecord r : rankings) {
            sj.add("Ranked " + ordinal(r.rank()) + " of " + r.total()
                    + " in " + r.category()
                    + " by " + r.source() + " (" + r.year() + "), score " + r.score() + "/100.");
            if (r.notes() != null && !r.notes().isBlank()) sj.add(r.notes());
        }
        return sj.toString();
    }

    private String buildSafety(String anchor, RichVehicleRecord v) {
        var s = v.safety();
        return anchor + "Safety ratings: "
                + "NHTSA overall " + safe(s.nhtsaOverall()) + " "
                + "(frontal: " + safe(s.nhtsaFrontal())
                + ", side: " + safe(s.nhtsaSide())
                + ", rollover: " + safe(s.nhtsaRollover()) + "). "
                + "IIHS overall: " + safe(s.iihsOverall()) + " "
                + "(small overlap: " + safe(s.iihsSmallOverlap())
                + ", headlights: " + safe(s.iihsHeadlights()) + "). "
                + (s.hasAutomaticEmergencyBraking() ? "Automatic emergency braking standard. " : "")
                + (s.hasLaneDepartureWarning()      ? "Lane departure warning standard. "       : "")
                + (s.hasBlindSpotMonitoring()        ? "Blind spot monitoring standard."         : "");
    }

    private String buildFeaturesAndTrims(String anchor, RichVehicleRecord v) {
        StringJoiner sj = new StringJoiner(" ");
        sj.add(anchor + "Features and trim levels:");

        if (v.features() != null && !v.features().isEmpty()) {
            sj.add("Standard features include: " + String.join(", ", v.features()) + ".");
        }

        if (v.trimLevels() != null) {
            for (var t : v.trimLevels()) {
                sj.add("The " + t.trim() + " trim is priced at " + t.msrp());
                if (t.extraHorsepower() != null && t.extraHorsepower() > 0) {
                    sj.add("and adds " + t.extraHorsepower() + "hp over base.");
                } else {
                    sj.add(".");
                }
                if (t.addedFeatures() != null && !t.addedFeatures().isEmpty()) {
                    sj.add("Added features: " + String.join(", ", t.addedFeatures()) + ".");
                }
            }
        }
        return sj.toString();
    }

    private String buildReviews(String anchor, List<ReviewSnippet> reviews) {
        StringJoiner sj = new StringJoiner(" ");
        sj.add(anchor + "Expert reviews:");
        for (ReviewSnippet r : reviews) {
            sj.add(r.source() + " (" + r.year() + ") rated it " + r.score() + "/10.");
            if (r.summary() != null) sj.add(r.summary());
            if (r.pros()    != null) sj.add("Pros: " + r.pros() + ".");
            if (r.cons()    != null) sj.add("Cons: " + r.cons() + ".");
        }
        return sj.toString();
    }

    private String buildMaintenanceInterval(String anchor, MaintenanceInterval m) {
        // One chunk per milestone — "60k service cost" retrieves exactly this chunk
        return anchor + "Maintenance at " + m.mileage() + " miles: "
                + m.service() + ". "
                + "Parts cost: $" + m.partsCost() + ". "
                + "Labor: " + m.laborHours() + " hours. "
                + "Dealer estimate: $" + m.dealerAvg() + ". "
                + "DIY estimate: $" + m.diyAvg() + "."
                + (m.notes() != null && !m.notes().isBlank() ? " Note: " + m.notes() : "");
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private VehicleChunk chunk(RichVehicleRecord v, int idx, String type, String text) {
        return new VehicleChunk(v.vehicleId(), idx, type, text);
    }

    private String safe(Object o) {
        return o == null ? "N/A" : String.valueOf(o).trim();
    }

    private String safeUnit(Object o, String unit) {
        if (o == null) return "N/A";
        return String.valueOf(o).trim() + " " + unit;
    }

    private String ordinal(int n) {
        return switch (n) {
            case 1 -> "1st"; case 2 -> "2nd"; case 3 -> "3rd"; default -> n + "th";
        };
    }
}
