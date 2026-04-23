package com.example.airagassistant.agentic.mapper;

import com.example.airagassistant.rag.VehicleCardDto;
import com.example.airagassistant.rag.retrieval.VehicleSummaryService;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

@Component
public class VehicleSummaryMapper {

    public VehicleCardDto toCard(VehicleSummaryService.VehicleSummaryHit hit) {
        Map<String, Object> attributes = new HashMap<>();

        attributes.put("year", hit.year());
        attributes.put("make", hit.make());
        attributes.put("model", hit.model());
        attributes.put("trim", hit.trim());

        // TODO: enrich these if available in DB
        attributes.put("chunkCount", hit.chunkCount());

        return new VehicleCardDto(
                hit.vehicleId(),
                buildTitle(hit),
                null, // imageUrl -> join later if needed
                hit.score(),
                attributes
        );
    }

    private String buildTitle(VehicleSummaryService.VehicleSummaryHit hit) {
        return "%d %s %s %s".formatted(
                hit.year(),
                hit.make(),
                hit.model(),
                hit.trim() != null ? hit.trim() : ""
        ).trim();
    }
}