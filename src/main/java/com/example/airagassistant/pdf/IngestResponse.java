package com.example.airagassistant.pdf;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class IngestResponse {
    private String docId;
    private String fileName;
    private int chunkCount;
    private int totalCharacters;
    private String status;
}