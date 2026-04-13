// ─────────────────────────────────────────────────────────────────────────────
// Domain records — drop these in domain/vehicle/
// ─────────────────────────────────────────────────────────────────────────────

package com.example.airagassistant.domain.vehicle;

import java.util.List;

// ── Root ─────────────────────────────────────────────────────────────────────
public record RichVehicleRecord(

        // identity
        String vehicleId,
        int    year,
        String make,
        String model,
        String trim,
        String bodyStyle,
        String vehicleClass,      // "sports sedan", "full-size truck", etc.
        String exteriorColor,
        String interiorColor,

        // powertrain
        String  engine,
        Integer horsepower,
        Integer torque,
        String  drivetrain,
        String  transmission,
        Double  zeroToSixty,      // seconds
        Integer topSpeed,         // mph
        String  fuelType,         // "gasoline", "hybrid", "electric"
        String  mpgCity,
        String  mpgHighway,
        Integer electricRangeMiles,

        // pricing
        String msrp,
        String invoicePrice,
        String destinationCharge,

        // sub-objects
        List<String>              features,
        List<TrimLevel>           trimLevels,
        List<RankingRecord>       rankings,
        List<MaintenanceInterval> maintenanceSchedule,
        OwnershipCost             ownershipCost,
        SafetyRecord              safety,
        List<RecallRecord>        recalls,
        List<ReviewSnippet>       reviews,

        String summary
) {}

