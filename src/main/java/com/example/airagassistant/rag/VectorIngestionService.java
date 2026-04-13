package com.example.airagassistant.rag;

import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

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

        List<VectorRecord> batch = new ArrayList<>(chunks.size());

        for (int i = 0; i < chunks.size(); i++) {
            String chunkText = chunks.get(i);
            if (chunkText == null || chunkText.isBlank()) {
                continue;
            }

            int chunkIndex = i + 1;
            String chunkId = docId + ":" + chunkIndex;

            var vector = embeddingClient.embed(chunkText);
            if (vector == null || vector.isEmpty()) {
                throw new IllegalStateException("Embedding failed for chunkId=" + chunkId);
            }

            batch.add(new VectorRecord(
                    documentId,
                    chunkIndex,
                    chunkId,
                    chunkText,
                    vector
            ));
        }

        vectorStore.addAll(batch);
    }
}