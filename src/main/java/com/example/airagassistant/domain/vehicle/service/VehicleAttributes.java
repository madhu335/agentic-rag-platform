package com.example.airagassistant.domain.vehicle.service;

/**
 * Typed, filterable facts about one vehicle, stored in the vehicle_attributes table.
 *
 * Text retrieval (BM25 + vectors) cannot answer "over 500 hp" or "most torque":
 * it matches tokens and meaning, not values. These columns let fleet questions
 * with a threshold or a superlative become a SQL WHERE / ORDER BY instead.
 *
 * Every field is nullable: EPA data has no horsepower, torque, MSRP or cost
 * data, and curated vehicles lack some fields. A null never matches a filter.
 */
public record VehicleAttributes(
        String vehicleId,
        String fuelType,
        String drivetrain,
        String bodyStyle,
        boolean electric,
        Integer horsepower,
        Integer torque,
        Integer mpgCity,
        Integer mpgHighway,
        Integer mpgCombined,
        Integer electricRange,
        Integer msrp,
        Integer annualFuelCost,
        Integer maintenancePerYear,
        Integer totalFiveYearCost,
        Double resaleValuePct,
        Integer nhtsaOverall,
        int recallCount
) {}
