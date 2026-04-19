package com.example.airagassistant.rag;

import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Generic vector ingestion — batch optimized.
 *
 * Previous: serial embed() per chunk → N HTTP calls to Ollama.
 * New: embedBatch() → 1 HTTP call to Ollama /api/embed (or ceil(N/32) calls
 *      if N > BATCH_SIZE, handled internally by OllamaEmbeddingClient).
 */
@Service
public class VectorIngestionService {

    private final EmbeddingClient embeddingClient;
    private final PgVectorStore vectorStore;

    public VectorIngestionService(EmbeddingClient embeddingClient, PgVectorStore vectorStore) {
        this.embeddingClient = embeddingClient;
        this.vectorStore = vectorStore;
    }

    public void ingestChunks(long documentId, String docId, List<String> chunks) {
        if (documentId <= 0) {
            throw new IllegalArgumentException("documentId must be positive");
        }
        if (docId == null || docId.isBlank()) {
            throw new IllegalArgumentException("docId cannot be blank");
        }
        if (chunks == null || chunks.isEmpty()) {
            return;
        }

        // Filter out blank chunks and collect texts for batch embedding
        List<String> validTexts = new ArrayList<>();
        List<Integer> validIndices = new ArrayList<>();

        for (int i = 0; i < chunks.size(); i++) {
            String chunkText = chunks.get(i);
            if (chunkText != null && !chunkText.isBlank()) {
                validTexts.add(chunkText);
                validIndices.add(i + 1);  // chunk_index is 1-based
            }
        }

        if (validTexts.isEmpty()) return;

        // Batch embed — one call instead of N
        List<List<Double>> embeddings = embeddingClient.embedBatch(validTexts);

        if (embeddings.size() != validTexts.size()) {
            throw new IllegalStateException(
                    "Embedding count mismatch: expected " + validTexts.size()
                            + " but got " + embeddings.size());
        }

        // Build VectorRecords
        List<VectorRecord> batch = new ArrayList<>(validTexts.size());
        for (int i = 0; i < validTexts.size(); i++) {
            List<Double> vector = embeddings.get(i);
            if (vector == null || vector.isEmpty()) {
                throw new IllegalStateException(
                        "Embedding failed for chunkId=" + docId + ":" + validIndices.get(i));
            }

            batch.add(new VectorRecord(
                    documentId,
                    validIndices.get(i),
                    docId + ":" + validIndices.get(i),
                    validTexts.get(i),
                    vector
            ));
        }

        vectorStore.addAll(batch);
    }
}
