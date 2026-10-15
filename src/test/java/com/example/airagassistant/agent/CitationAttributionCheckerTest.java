package com.example.airagassistant.agent;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class CitationAttributionCheckerTest {

    private static final String M3_SPECS = "bmw-m3-2025-competition:2";
    private static final String M3_TRIMS = "bmw-m3-2025-competition:6";
    private static final String M3_PRICE = "bmw-m3-2025-competition:1";
    private static final String CRV_SPECS = "honda-crv-2025-sport-hybrid:2";

    private static Map<String, String> chunks() {
        Map<String, String> c = new LinkedHashMap<>();
        c.put(M3_SPECS, "2025 BMW M3 Competition sports sedan. Performance specs: Engine: 3.0L twin-turbo "
                + "inline-6 (S58). Horsepower: 503 hp. Torque: 479 lb-ft. 0-60 mph: 3.4 seconds.");
        c.put(M3_TRIMS, "2025 BMW M3 Competition sports sedan. The Base M3 trim is priced at $72,800 . "
                + "The Competition trim is priced at $76,900 and adds 30hp over base.");
        c.put(M3_PRICE, "2025 BMW M3 Competition sports sedan. Body style: Sedan. Fuel type: gasoline. "
                + "MSRP: $76,900. Invoice: $72,840.");
        c.put(CRV_SPECS, "2025 Honda CR-V Sport Hybrid compact SUV. Horsepower: 204 hp. Torque: 247 lb-ft.");
        return c;
    }

    @Test
    void repointsMisattributedCitationToChunkContainingTheNumber() {
        // Real answer from /ask on 2026-10-15: 503 hp cited to the price chunk.
        var result = CitationAttributionChecker.check(
                "The 2025 BMW M3 Competition produces 503 hp [" + M3_PRICE + "] from a 3.0L "
                        + "twin-turbo inline-6 engine [" + M3_SPECS + "].",
                chunks());

        assertThat(result.answer()).isEqualTo(
                "The 2025 BMW M3 Competition produces 503 hp [" + M3_SPECS + "] from a 3.0L "
                        + "twin-turbo inline-6 engine [" + M3_SPECS + "].");
        assertThat(result.correctedCount()).isEqualTo(1);
        assertThat(result.droppedCount()).isZero();
    }

    @Test
    void keepsCorrectCitation() {
        var result = CitationAttributionChecker.check("It costs $76,900 [" + M3_PRICE + "].", chunks());

        assertThat(result.corrections()).isEmpty();
    }

    @Test
    void leavesClaimsWithoutNumbersUnchanged() {
        var result = CitationAttributionChecker.check("It is a sports sedan [" + M3_TRIMS + "].", chunks());

        assertThat(result.corrections()).isEmpty();
    }

    @Test
    void dropsCitationWhenNoChunkSupportsTheNumber() {
        var result = CitationAttributionChecker.check("It makes 620 hp [" + M3_SPECS + "].", chunks());

        assertThat(result.answer()).isEqualTo("It makes 620 hp.");
        assertThat(result.droppedCount()).isEqualTo(1);
    }

    @Test
    void repointsAcrossVehiclesWhenSameVehicleHasNoMatch() {
        var result = CitationAttributionChecker.check("The CR-V makes 204 hp [" + M3_SPECS + "].", chunks());

        assertThat(result.answer()).isEqualTo("The CR-V makes 204 hp [" + CRV_SPECS + "].");
    }

    @Test
    void numbersFromAPreviousSentenceDoNotLeakIntoTheClaim() {
        var result = CitationAttributionChecker.check(
                "Base is $72,800. The Competition makes 503 hp [" + M3_SPECS + "].", chunks());

        assertThat(result.corrections()).isEmpty();
    }

    @Test
    void extractNumbersIgnoresModelNamesAndNormalisesDecimals() {
        assertThat(CitationAttributionChecker.extractNumbers("M3 with S58, 3.0L, $76,900, inline-6"))
                .containsExactly("3", "76900", "6");
    }
}
