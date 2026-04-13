package com.example.airagassistant.pdf;

import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class TextChunker {

    private static final int CHUNK_SIZE = 800;
    private static final int CHUNK_OVERLAP = 100;

    public List<String> chunk(String text) {
        List<String> chunks = new ArrayList<>();

        if (text == null || text.isBlank()) {
            return chunks;
        }

        String[] sections = text.split("(?=Q\\d+\\.)");

        for (String section : sections) {
            String cleaned = section.trim();
            if (!cleaned.isBlank()) {
                chunks.add(cleaned);
            }
        }

        return chunks;
    }
}