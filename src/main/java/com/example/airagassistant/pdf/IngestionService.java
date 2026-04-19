package com.example.airagassistant.pdf;

import com.example.airagassistant.rag.EmbeddingClient;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

/**
 * PDF ingestion service — batch optimized.
 *
 * Previous flow (serial):
 *   for each chunk:
 *     embed(chunk)          → 1 HTTP round-trip to Ollama
 *     saveChunk(chunk, vec) → 1 INSERT round-trip to Postgres
 *   Total for 20 chunks: 20 embed calls + 20 inserts = 40 round-trips
 *
 * New flow (batch):
 *   embedBatch(allChunks)   → 1 HTTP call to Ollama /api/embed
 *   batchSaveChunks(all)    → 1 JDBC batch INSERT to Postgres
 *   Total for 20 chunks: 1 embed call + 1 batch insert = 2 round-trips
 *
 * Note: EmbeddingClient is used instead of EmbeddingService (which was
 * just a pass-through wrapper). If you want to keep the wrapper, add
 * embedBatch() to EmbeddingService as well.
 */
@Slf4j
@Service
@RequiredArgsConstructor
@Transactional
public class IngestionService {

    private final PdfExtractorService pdfExtractorService;
    private final TextChunker textChunker;
    private final EmbeddingClient embeddingClient;
    private final ChunkStorageService chunkStorageService;

    public IngestResponse ingestPdf(MultipartFile file, String docId) {
        if (file == null || file.isEmpty()) {
            throw new IllegalArgumentException("Uploaded file is empty");
        }

        String resolvedDocId = (docId != null && !docId.isBlank())
                ? docId
                : generateDocId(file.getOriginalFilename());

        // Step 1: Extract text from PDF
        String extractedText = pdfExtractorService.extractText(file);

        // Step 2: Chunk the text
        List<String> chunks = textChunker.chunk(extractedText);
        if (chunks.isEmpty()) {
            throw new IllegalArgumentException("No chunks generated from PDF");
        }

        log.info("Ingesting PDF '{}' — {} chunks, {} chars",
                resolvedDocId, chunks.size(), extractedText.length());

        // Step 3: Delete old chunks for this docId (idempotent re-ingest)
        chunkStorageService.deleteByDocId(resolvedDocId);

        // Step 4: Batch embed all chunks in one Ollama call
        List<List<Double>> embeddings = embeddingClient.embedBatch(chunks);

        if (embeddings.size() != chunks.size()) {
            throw new IllegalStateException(
                    "Embedding count mismatch: expected " + chunks.size()
                            + " but got " + embeddings.size());
        }

        // Step 5: Batch insert all chunks into Postgres
        chunkStorageService.batchSaveChunks(resolvedDocId, chunks, embeddings);

        log.info("PDF '{}' ingested — {} chunks", resolvedDocId, chunks.size());

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
