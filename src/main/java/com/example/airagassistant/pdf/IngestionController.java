package com.example.airagassistant.pdf;

import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@CrossOrigin(origins = "http://localhost:5173")
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class IngestionController {

    private final IngestionService ingestionService;

    @PostMapping(value = "/ingest-pdf", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<IngestResponse> ingestPdf(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "docId", required = false) String docId
    ) {
        IngestResponse response = ingestionService.ingestPdf(file, docId);
        return ResponseEntity.ok(response);
    }
}