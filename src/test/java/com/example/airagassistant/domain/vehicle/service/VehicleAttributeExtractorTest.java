package com.example.airagassistant.domain.vehicle.service;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class VehicleAttributeExtractorTest {

    @Test
    void epaGasVehicle_trimNumbersNeverLeakIntoValues() {
        // "S500" in the trim and the model year at the start must not be read as horsepower or MPG
        String anchor = "2024 Mercedes-Benz S-Class S500 4matic 3.0L 6-cylinder turbocharged large cars. ";
        Map<Integer, String> chunks = Map.of(
                1, anchor + "Body style: Car. Class: large cars. Fuel type: premium gasoline. ",
                2, anchor + "Performance specs: Engine: 3.0L 6-cylinder turbocharged. Drivetrain: AWD. "
                        + "Transmission: 9-speed automatic. Fuel economy: 20 city / 29 highway MPG.",
                6, anchor + "Features and trim levels: Standard features include: EPA estimated annual fuel cost "
                        + "$2,750 (15,000 miles/year), EPA combined fuel economy 23 MPG, Tailpipe CO2 385 grams per mile.",
                20, anchor + "Recall 24V001000 (2024-01-05): Electrical System. Something. Remedy: Fix.");

        VehicleAttributes a = VehicleAttributeExtractor.extract("merc-s500", chunks);

        assertNull(a.horsepower(), "EPA data has no horsepower; S500 must not become 500 hp");
        assertNull(a.torque());
        assertNull(a.msrp());
        assertEquals(20, a.mpgCity());
        assertEquals(29, a.mpgHighway());
        assertEquals(23, a.mpgCombined(), "EPA's own combined figure wins over the harmonic estimate");
        assertEquals(2750, a.annualFuelCost());
        assertEquals("premium gasoline", a.fuelType());
        assertEquals("AWD", a.drivetrain());
        assertEquals("Car", a.bodyStyle());
        assertFalse(a.electric());
        assertEquals(1, a.recallCount());
    }

    @Test
    void epaElectricVehicle() {
        String anchor = "2025 Tesla Model 3 Long Range AWD midsize cars. ";
        Map<Integer, String> chunks = Map.of(
                1, anchor + "Body style: Car. Class: midsize cars. Fuel type: electric. ",
                2, anchor + "Performance specs: Engine: Electric motor (158 and 208 kW AC PMSM). Drivetrain: AWD. "
                        + "Transmission: single-speed direct drive. Fuel economy: 134 city / 126 highway MPG. "
                        + "Electric range: 346 miles.");

        VehicleAttributes a = VehicleAttributeExtractor.extract("tesla", chunks);

        assertTrue(a.electric());
        assertEquals(346, a.electricRange());
        assertEquals(130, a.mpgCombined(), "harmonic 55/45 of 134 and 126");
    }

    @Test
    void hybridIsNotElectric() {
        Map<Integer, String> chunks = Map.of(
                1, "2024 Toyota Camry. Body style: Car. Fuel type: hybrid. ",
                2, "2024 Toyota Camry. Performance specs: Engine: 2.5L 4-cylinder hybrid (88 kW AC PMSM). "
                        + "Fuel economy: 51 city / 49 highway MPG.");
        assertFalse(VehicleAttributeExtractor.extract("camry", chunks).electric());
    }

    @Test
    void curatedRichVehicle_allCostFields() {
        String anchor = "2025 BMW M3 Competition sports sedan. ";
        Map<Integer, String> chunks = Map.of(
                1, anchor + "Body style: Sedan. Class: sports sedan. Fuel type: gasoline. MSRP: $76,900. Invoice: $71,200.",
                2, anchor + "Performance specs: Engine: 3.0L twin-turbo inline-6 (S58). Horsepower: 503 hp. "
                        + "Torque: 479 lb-ft. 0-60 mph: 3.4 seconds. Drivetrain: RWD. Transmission: 8-speed automatic. "
                        + "Fuel economy: 16 city / 23 highway MPG.",
                3, anchor + "5-year ownership costs: Fuel $2800/year. Insurance $3400/year. Maintenance $1100/year. "
                        + "5-year depreciation: $28000. Total 5-year cost: $52000. "
                        + "Resale value after 3 years: 58.0% of MSRP (excellent).",
                5, anchor + "Safety ratings: NHTSA overall 5-star (frontal: 5-star). ",
                6, anchor + "Features and trim levels: The CS trim is priced at $119,900 and adds 40hp over base.");

        VehicleAttributes a = VehicleAttributeExtractor.extract("bmw-m3-2025-competition", chunks);

        assertEquals(503, a.horsepower());
        assertEquals(479, a.torque());
        assertEquals(76_900, a.msrp(), "the MSRP label, not the CS trim price");
        assertEquals(1100, a.maintenancePerYear());
        assertEquals(52_000, a.totalFiveYearCost());
        assertEquals(58.0, a.resaleValuePct());
        assertEquals(2800, a.annualFuelCost(), "ownership fuel cost when no EPA figure");
        assertEquals(5, a.nhtsaOverall());
        assertEquals(19, a.mpgCombined());
        assertEquals(0, a.recallCount());
    }

    @Test
    void legacyMultiLineDocument() {
        String doc = String.join("\n",
                "2025 Rivian R1T Adventure vehicle specification.",
                "Vehicle ID: rivian-r1t-2025-adventure",
                "Engine: Quad motor electric",
                "Horsepower: 835 hp",
                "Torque: 908 lb-ft",
                "Drivetrain: AWD",
                "Fuel Economy City: N/A MPGe/MPG",
                "Fuel Economy Highway: N/A MPGe/MPG",
                "MSRP: $73,000");

        VehicleAttributes a = VehicleAttributeExtractor.extract("rivian", Map.of(0, doc));

        assertEquals(835, a.horsepower());
        assertEquals(908, a.torque());
        assertEquals(73_000, a.msrp());
        assertNull(a.mpgCombined(), "N/A is unknown, not zero");
        assertTrue(a.electric(), "no fuel type line: inferred from an electric engine");
        assertEquals("AWD", a.drivetrain());
    }
}
