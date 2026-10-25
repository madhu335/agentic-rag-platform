package com.example.airagassistant.domain.vehicle.service;

import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A fleet question that is really a structured query over vehicle_attributes.
 *
 *   "which vehicles have over 500 horsepower"   -> horsepower >  500   ORDER BY horsepower DESC
 *   "vehicles under $40000"                      -> msrp       < 40000  ORDER BY msrp ASC
 *   "vehicles with at least 40 mpg"              -> mpg_combined >= 40 (non-EV) ORDER BY mpg DESC
 *   "which car has the most torque"              -> ORDER BY torque DESC
 *   "vehicle with lowest total 5 year cost"      -> ORDER BY total_five_year_cost ASC
 *
 * Only two shapes qualify: a THRESHOLD (comparison word + number + field) or a
 * SUPERLATIVE (most/least/best/... + field). Everything else returns empty and
 * stays on hybrid text retrieval, so subjective questions are unaffected.
 *
 * Deliberately conservative: a false positive replaces good text retrieval with
 * the wrong SQL, so anything ambiguous is left to hybrid search.
 */
public record StructuredVehicleQuery(Field field, Op op, Double value, boolean descending) {

    public enum Op { GT, GTE, LT, LTE, SUPERLATIVE }

    /** Filterable column, its natural "better" direction, and the chunk that shows it. */
    public enum Field {
        HORSEPOWER("horsepower", true, 2, 50, 2_000),
        TORQUE("torque", true, 2, 50, 2_500),
        MPG("mpg_combined", true, 2, 5, 200),
        ELECTRIC_RANGE("electric_range", true, 2, 10, 1_000),
        MSRP("msrp", false, 1, 1_000, 5_000_000),
        ANNUAL_FUEL_COST("annual_fuel_cost", false, 6, 50, 20_000),
        MAINTENANCE_PER_YEAR("maintenance_per_year", false, 3, 50, 50_000),
        TOTAL_FIVE_YEAR_COST("total_five_year_cost", false, 3, 1_000, 2_000_000),
        RESALE_VALUE_PCT("resale_value_pct", true, 3, 1, 100);

        public final String column;
        /** true if a larger value is "better" (best resale = highest %); false for costs. */
        public final boolean higherIsBetter;
        /** chunk_index that states this value (1 identity, 2 performance, 3 ownership, 6 features). */
        public final int chunkIndex;
        /** Plausible range for a threshold. Rejects "over 5 years" being read as cost > 5. */
        final double min;
        final double max;

        Field(String column, boolean higherIsBetter, int chunkIndex, double min, double max) {
            this.column = column;
            this.higherIsBetter = higherIsBetter;
            this.chunkIndex = chunkIndex;
            this.min = min;
            this.max = max;
        }

        /** MPG of an EV is MPGe on a different scale, so MPG questions exclude EVs. */
        public boolean excludesElectric() {
            return this == MPG;
        }
    }

    // A concrete vehicle id in the question ("bmw-m3-2025-competition ...") means a
    // single-vehicle lookup, e.g. from CompareVehiclesTool. Never turn that into a fleet query.
    private static final Pattern VEHICLE_ID = Pattern.compile("\\b[a-z0-9]+(?:-[a-z0-9]+)*-(19|20)\\d{2}(?:-[a-z0-9]+)*\\b");

    private static final Pattern GT = Pattern.compile("\\b(over|more than|above|greater than|exceeding)\\b");
    private static final Pattern GTE = Pattern.compile("\\b(at least|minimum of|or more|no less than)\\b");
    private static final Pattern LT = Pattern.compile("\\b(under|less than|below|cheaper than)\\b");
    private static final Pattern LTE = Pattern.compile("\\b(at most|maximum of|or less|no more than)\\b");

    // Superlatives. "best"/"worst" take the field's own direction (best resale = highest, best cost = lowest).
    private static final Pattern SUP_BEST = Pattern.compile("\\b(best|top|ideal)\\b");
    private static final Pattern SUP_WORST = Pattern.compile("\\bworst\\b");
    private static final Pattern SUP_HIGH = Pattern.compile("\\b(most|highest|greatest|largest|biggest|longest|maximum)\\b");
    private static final Pattern SUP_LOW = Pattern.compile("\\b(least|lowest|cheapest|smallest|shortest|fewest|minimum)\\b");

    // A number that is a duration or distance, not the threshold: "over 5 years", "30000 mile service".
    private static final Pattern NUMBER = Pattern.compile("\\$?\\s*(\\d[\\d,]*(?:\\.\\d+)?)\\s*(k\\b)?(?!\\s*(?:year|yr|month|mile|mi\\b|-year|day))");

    public static Optional<StructuredVehicleQuery> parse(String question) {
        if (question == null || question.isBlank()) return Optional.empty();
        String q = question.toLowerCase().replace("’", "'");

        if (VEHICLE_ID.matcher(q).find()) return Optional.empty();

        Field field = detectField(q);
        if (field == null) return Optional.empty();

        Op op = detectThresholdOp(q);
        if (op != null) {
            Double value = thresholdValue(q, op);
            if (value != null && value >= field.min && value <= field.max) {
                boolean desc = op == Op.GT || op == Op.GTE;
                return Optional.of(new StructuredVehicleQuery(field, op, value, desc));
            }
            // A comparison word without a usable number ("over 5 years") is not a threshold;
            // it may still be a superlative, so fall through.
        }

        Boolean desc = superlativeDirection(q, field);
        if (desc != null) {
            return Optional.of(new StructuredVehicleQuery(field, Op.SUPERLATIVE, null, desc));
        }
        return Optional.empty();
    }

    /** Most specific phrases first: "most expensive to maintain" is maintenance cost, not price. */
    static Field detectField(String q) {
        if (q.matches(".*\\bresale\\b.*")) return Field.RESALE_VALUE_PCT;
        if (q.matches(".*\\b(maintain|maintenance|upkeep)\\b.*")) return Field.MAINTENANCE_PER_YEAR;
        if (q.matches(".*\\b(5|five)[ -]year\\b.*") || q.matches(".*\\b(ownership cost|cost of ownership|cost to own|total cost)\\b.*"))
            return Field.TOTAL_FIVE_YEAR_COST;
        if (q.matches(".*\\b(fuel cost|gas cost|spend on (gas|fuel))\\b.*")) return Field.ANNUAL_FUEL_COST;
        if (q.matches(".*\\b(horsepower|hp|most powerful)\\b.*")) return Field.HORSEPOWER;
        if (q.matches(".*\\b(torque|lb-ft)\\b.*")) return Field.TORQUE;
        if (q.matches(".*\\b(mpg|fuel economy|fuel[- ]efficient|fuel efficiency|gas mileage)\\b.*")) return Field.MPG;
        if (q.matches(".*\\b(electric range|ev range|driving range|range)\\b.*")) return Field.ELECTRIC_RANGE;
        if (q.matches(".*(\\$|\\b\\d+k\\b|\\b(price|priced|msrp|cost|costs|dollars|cheapest|cheap|expensive|affordable|budget)\\b).*"))
            return Field.MSRP;
        return null;
    }

    static Op detectThresholdOp(String q) {
        if (GTE.matcher(q).find()) return Op.GTE;   // "or more" before "more than"
        if (LTE.matcher(q).find()) return Op.LTE;
        if (GT.matcher(q).find()) return Op.GT;
        if (LT.matcher(q).find()) return Op.LT;
        return null;
    }

    /** The first number after the comparison phrase ("40k" = 40000); null if none qualifies. */
    static Double thresholdValue(String q, Op op) {
        Pattern opPattern = switch (op) {
            case GT -> GT;
            case GTE -> GTE;
            case LT -> LT;
            case LTE -> LTE;
            default -> null;
        };
        int from = 0;
        if (opPattern != null) {
            Matcher om = opPattern.matcher(q);
            if (om.find()) from = om.start();
        }
        // "or more"/"or less" follow the number, so search the whole string for those.
        if (op == Op.GTE && q.contains("or more") || op == Op.LTE && q.contains("or less")) from = 0;

        Matcher m = NUMBER.matcher(q);
        while (m.find()) {
            if (m.start() < from) continue;
            double v = Double.parseDouble(m.group(1).replace(",", ""));
            if (m.group(2) != null) v *= 1_000;
            return v;
        }
        return null;
    }

    /** true = descending, false = ascending, null = not a superlative. */
    static Boolean superlativeDirection(String q, Field field) {
        // "most expensive", "least expensive" talk about cost, where "expensive" flips the meaning
        boolean costWord = q.matches(".*\\b(expensive|costly|pricey)\\b.*");
        boolean cheapWord = q.matches(".*\\b(cheapest|cheap|affordable|inexpensive)\\b.*");

        if (SUP_BEST.matcher(q).find()) return field.higherIsBetter;
        if (SUP_WORST.matcher(q).find()) return !field.higherIsBetter;
        if (cheapWord && q.matches(".*\\b(cheapest|most affordable|least expensive)\\b.*")) return false;
        if (SUP_HIGH.matcher(q).find()) return !(cheapWord) || costWord;
        if (SUP_LOW.matcher(q).find()) return false;
        return null;
    }

    public boolean isThreshold() {
        return op != Op.SUPERLATIVE;
    }

    public String sqlOperator() {
        return switch (op) {
            case GT -> ">";
            case GTE -> ">=";
            case LT -> "<";
            case LTE -> "<=";
            case SUPERLATIVE -> throw new IllegalStateException("superlative has no operator");
        };
    }
}
