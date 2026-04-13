package com.example.airagassistant.rag;

import com.example.airagassistant.trace.TraceHelper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class RagRetriever {

    private final EmbeddingClient embeddingClient;
    private final PgVectorStore vectorStore;
    private final ReRankService reRankService;
    private final TraceHelper traceHelper;

    public List<SearchHit> retrieve(String docId, String question, int topK) {
        validateQuestion(question);

        List<Double> qVec = embedQuestion(docId, question);

        return traceHelper.run(
                "retrieve-hybrid-default",
                buildRetrieveAttributes(docId, question, topK, "HYBRID"),
                () -> {
                    List<SearchHit> hits = vectorStore.hybridSearch(docId, qVec, question, topK);
                    addHitAttributes(hits);
                    return hits;
                }
        );
    }

    public List<SearchHit> retrieve(String docId, String question, int topK, RetrievalMode mode) {
        validateQuestion(question);

        Map<String, Object> attrs = buildRetrieveAttributes(docId, question, topK, mode.name());

        return traceHelper.run(
                "retrieve-" + mode.name().toLowerCase(),
                attrs,
                () -> {
                    List<SearchHit> hits = switch (mode) {
                        case VECTOR -> {
                            List<Double> qVec = embedQuestion(docId, question);
                            yield vectorStore.searchWithScores(docId, qVec, topK);
                        }
                        case BM25 -> {
                            // Postgres full-text search ignores tokens < 3 chars (e.g. "M3", "V8").
                            // If BM25 returns nothing, fall back to vector search so short-token
                            // queries like "What engine does the M3 use?" still get an answer.
                            List<SearchHit> bm25Hits = vectorStore.keywordSearch(docId, question, topK);
                            if (bm25Hits.isEmpty()) {
                                log.debug("BM25 returned no hits for docId={}, falling back to vector", docId);
                                List<Double> qVec = embedQuestion(docId, question);
                                yield vectorStore.searchWithScores(docId, qVec, topK);
                            }
                            yield bm25Hits;
                        }
                        case HYBRID -> {
                            List<Double> qVec = embedQuestion(docId, question);
                            yield vectorStore.hybridSearch(docId, qVec, question, topK);
                        }
                        case HYBRID_RERANK -> {
                            List<Double> qVec = embedQuestion(docId, question);
                            List<SearchHit> hybridHits = vectorStore.hybridSearch(docId, qVec, question, topK);

                            traceHelper.addAttributes(Map.of(
                                    "langsmith.metadata.hybrid_hit_count", hybridHits.size()
                            ));

                            List<SearchHit> reranked = reRankService.rerank(question, hybridHits);

                            Map<String, Object> rerankAttrs = new LinkedHashMap<>();
                            rerankAttrs.put("langsmith.metadata.rerank_input_count", hybridHits.size());
                            rerankAttrs.put("langsmith.metadata.rerank_output_count", reranked.size());
                            traceHelper.addAttributes(rerankAttrs);

                            yield reranked;
                        }
                    };

                    addHitAttributes(hits);
                    return hits;
                }
        );
    }

    private List<Double> embedQuestion(String docId, String question) {
        return traceHelper.run(
                "embedding-call",
                buildEmbeddingAttributes(docId, question),
                () -> {
                    List<Double> qVec = embeddingClient.embed(question);

                    Map<String, Object> attrs = new LinkedHashMap<>();
                    attrs.put("langsmith.metadata.embedding_size", qVec.size());
                    traceHelper.addAttributes(attrs);

                    return qVec;
                }
        );
    }

    private void validateQuestion(String question) {
        if (question == null || question.isBlank()) {
            throw new IllegalArgumentException("question cannot be blank");
        }
    }

    private Map<String, Object> buildEmbeddingAttributes(String docId, String question) {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.span.kind", "embedding");
        attrs.put("langsmith.metadata.doc_id", docId);
        attrs.put("gen_ai.prompt.0.role", "user");
        attrs.put("gen_ai.prompt.0.content", question);
        return attrs;
    }

    private Map<String, Object> buildRetrieveAttributes(String docId, String question, int topK, String mode) {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.span.kind", "retriever");
        attrs.put("langsmith.metadata.doc_id", docId);
        attrs.put("langsmith.metadata.top_k", topK);
        attrs.put("langsmith.metadata.retrieval_mode", mode);
        attrs.put("gen_ai.prompt.0.content", question);
        return attrs;
    }

    private void addHitAttributes(List<SearchHit> hits) {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("langsmith.metadata.hit_count", hits.size());

        if (!hits.isEmpty()) {
            attrs.put("langsmith.metadata.top_chunk_id", hits.get(0).record().id());
            attrs.put("langsmith.metadata.top_score", hits.get(0).score());
        }

        traceHelper.addAttributes(attrs);
    }
}