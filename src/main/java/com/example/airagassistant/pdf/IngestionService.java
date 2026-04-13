package com.example.airagassistant.pdf;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@Service
@RequiredArgsConstructor
@Transactional
public class IngestionService {

    private final PdfExtractorService pdfExtractorService;
    private final TextChunker textChunker;
    private final EmbeddingService embeddingService;
    private final ChunkStorageService chunkStorageService;

    public IngestResponse ingestPdf(MultipartFile file, String docId) {
        if (file == null || file.isEmpty()) {
            throw new IllegalArgumentException("Uploaded file is empty");
        }

        String resolvedDocId = (docId != null && !docId.isBlank())
                ? docId
                : generateDocId(file.getOriginalFilename());

        String extractedText = pdfExtractorService.extractText(file);
        List<String> chunks = textChunker.chunk(extractedText);

        if (chunks.isEmpty()) {
            throw new IllegalArgumentException("No chunks generated from PDF");
        }

        chunkStorageService.deleteByDocId(resolvedDocId);

        for (int i = 0; i < chunks.size(); i++) {
            String chunkText = chunks.get(i);
            List<Double> embedding = embeddingService.embed(chunkText);
            chunkStorageService.saveChunk(resolvedDocId, i, chunkText, embedding);
        }

        return IngestResponse.builder()
                .docId(resolvedDocId)
                .fileName(file.getOriginalFilename())
                .chunkCount(chunks.size())
                .totalCharacters(extractedText.length())
                .status("INGESTED")
                .build();
    }

    private String generateDocId(String fileName) {
        String safeName = (fileName == null || fileName.isBlank())
                ? "pdf-doc"
                : fileName.replaceAll("[^a-zA-Z0-9-_\\.]", "_");

        return safeName + "_" + System.currentTimeMillis();
    }
}