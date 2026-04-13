package com.example.airagassistant.eval;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.InputStream;
import java.util.List;

@Component
public class GoldenSetLoader {

    private final ObjectMapper objectMapper;

    public GoldenSetLoader(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public List<EvalCase> load() {
        try {
            ClassPathResource resource = new ClassPathResource("eval/golden-set.json");
            try (InputStream in = resource.getInputStream()) {
                return objectMapper.readValue(in, new TypeReference<List<EvalCase>>() {});
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to load eval dataset", e);
        }
    }
}