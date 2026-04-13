package com.example.airagassistant.domain.vehicle;

import java.util.List;

public record VehicleRecord(
        String vehicleId,
        int year,
        String make,
        String model,
        String trim,
        String bodyStyle,
        String engine,
        Integer horsepower,
        Integer torque,
        String drivetrain,
        String transmission,
        String mpgCity,
        String mpgHighway,
        String msrp,
        List<String> features,
        String summary
) {}