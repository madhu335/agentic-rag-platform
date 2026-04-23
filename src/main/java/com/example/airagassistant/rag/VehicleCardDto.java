package com.example.airagassistant.rag;

import java.util.Map;

public record VehicleCardDto(
        String vehicleId,
        String title,
        String imageUrl,
        Double score,
        Map<String, Object> attributes
) {}