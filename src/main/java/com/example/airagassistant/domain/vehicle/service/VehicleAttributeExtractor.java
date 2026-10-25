package com.example.airagassistant.domain.vehicle.service;

import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Extracts {@link VehicleAttributes} from a vehicle's indexed chunk text.
 *
 * Parsing the chunk text (instead of the ingest payload) gives one code path for
 * both new ingestion and backfilling vehicles already in the database, and keeps
 * the attributes consistent with what retrieval actually sees.
 *
 * Every pattern is anchored on a field LABEL ("Horsepower:", "MSRP:", ...), never
 * on a bare number. Trim names such as "S500" or "M40i", and the model year that
 * opens every chunk, therefore cannot leak into a value. That was the bug in the
 * old NumericFilter, which took the first number on the line: always the year.
 *
 * Handles both chunk formats in the database:
 *   - VehicleChunkBuilder (rich):  "Horsepower: 503 hp."  "Fuel economy: 16 city / 23 highway MPG."
 *   - VehicleDocumentBuilder (basic, multi-line): "Fuel Economy City: 16 MPGe/MPG"
 */
public final class VehicleAttributeExtractor {

    private VehicleAttributeExtractor() {}

    private static final int FIRST_RECALL_CHUNK_INDEX = 20;

    private static final Pattern HORSEPOWER = label("Horsepower:\\s*(\\d+)");
    private static final Pattern TORQUE = label("Torque:\\s*(\\d+)");
    private static final Pattern MSRP = label("MSRP:\\s*\\$?\\s*(\\d[\\d,]*)");
    private static final Pattern MPG_RICH = label("Fuel economy:\\s*(\\d+)\\s*city\\s*/\\s*(\\d+)\\s*highway");
    private static final Pattern MPG_CITY_BASIC = label("Fuel Economy City:\\s*(\\d+)");
    private static final Pattern MPG_HWY_BASIC = label("Fuel Economy Highway:\\s*(\\d+)");
    private static final Pattern EPA_COMBINED = label("EPA combined fuel economy (\\d+) (MPGe|MPG)");
    private static final Pattern EPA_FUEL_COST = label("annual fuel cost \\$(\\d[\\d,]*)");
    private static final Pattern ELECTRIC_RANGE = label("Electric range:\\s*(\\d+)");
    private static final Pattern FUEL_TYPE = label("Fuel type:\\s*([^.\\n]+)");
    private static final Pattern ENGINE = label("Engine:\\s*([^.\\n]+(?:\\.\\d[^.\\n]*)?)");
    private static final Pattern DRIVETRAIN = label("Drivetrain:\\s*([^.\\n]+)");
    private static final Pattern BODY_STYLE = label("Body style:\\s*([^.\\n]+)");
    private static final Pattern OWN_FUEL = label("Fuel \\$(\\d[\\d,]*)/year");
    private static final Pattern OWN_MAINT = label("Maintenance \\$(\\d[\\d,]*)/year");
    private static final Pattern OWN_TOTAL = label("Total 5-year cost:\\s*\\$(\\d[\\d,]*)");
    private static final Pattern OWN_RESALE = label("Resale value after 3 years:\\s*(\\d+(?:\\.\\d+)?)%");
    private static final Pattern NHTSA_OVERALL = label("NHTSA overall (\\d)-star");

    private static Pattern label(String regex) {
        return Pattern.compile(regex, Pattern.CASE_INSENSITIVE);
    }

    /**
     * @param vehicleId the vehicle
     * @param chunksByIndex chunk_index -> chunk text, for every chunk of this vehicle
     */
    public static VehicleAttributes extract(String vehicleId, Map<Integer, String> chunksByIndex) {
        Map<Integer, String> chunks = new TreeMap<>(chunksByIndex);
        String all = String.join("\n", chunks.values());

        String fuelType = text(FUEL_TYPE, all);
        String engine = text(ENGINE, all);
        Integer range = integer(ELECTRIC_RANGE, all);
        boolean electric = isElectric(fuelType, engine);

        Integer city = null;
        Integer highway = null;
        Matcher rich = MPG_RICH.matcher(all);
        if (rich.find()) {
            city = Integer.parseInt(rich.group(1));
            highway = Integer.parseInt(rich.group(2));
        } else {
            city = integer(MPG_CITY_BASIC, all);
            highway = integer(MPG_HWY_BASIC, all);
        }

        // Prefer EPA's own combined figure; otherwise use EPA's 55% city / 45% highway harmonic mean.
        Integer combined = integer(EPA_COMBINED, all);
        if (combined == null && city != null && highway != null && city > 0 && highway > 0) {
            combined = (int) Math.round(1.0 / (0.55 / city + 0.45 / highway));
        }

        Integer annualFuelCost = integer(EPA_FUEL_COST, all);
        if (annualFuelCost == null) annualFuelCost = integer(OWN_FUEL, all);

        int recalls = (int) chunks.keySet().stream().filter(i -> i >= FIRST_RECALL_CHUNK_INDEX).count();

        return new VehicleAttributes(
                vehicleId,
                fuelType,
                text(DRIVETRAIN, all),
                text(BODY_STYLE, all),
                electric,
                integer(HORSEPOWER, all),
                integer(TORQUE, all),
                city,
                highway,
                combined,
                range,
                integer(MSRP, all),
                annualFuelCost,
                integer(OWN_MAINT, all),
                integer(OWN_TOTAL, all),
                decimal(OWN_RESALE, all),
                integer(NHTSA_OVERALL, all),
                recalls
        );
    }

    /** Battery-electric only. Hybrids and plug-in hybrids burn fuel, so MPG filters apply to them. */
    static boolean isElectric(String fuelType, String engine) {
        String f = fuelType == null ? "" : fuelType.toLowerCase();
        if (!f.isBlank()) {
            return f.contains("electric") && !f.contains("hybrid");
        }
        String e = engine == null ? "" : engine.toLowerCase();
        return (e.contains("electric") || e.contains("motor")) && !e.contains("hybrid") && !e.matches(".*\\d\\.\\dl.*");
    }

    private static String text(Pattern p, String s) {
        Matcher m = p.matcher(s);
        if (!m.find()) return null;
        String v = m.group(1).trim();
        return v.isEmpty() || v.equalsIgnoreCase("N/A") ? null : v;
    }

    private static Integer integer(Pattern p, String s) {
        Matcher m = p.matcher(s);
        if (!m.find()) return null;
        try {
            return Integer.parseInt(m.group(1).replace(",", ""));
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private static Double decimal(Pattern p, String s) {
        Matcher m = p.matcher(s);
        return m.find() ? Double.parseDouble(m.group(1)) : null;
    }
}
