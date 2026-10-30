package com.example.airagassistant.rag;

import java.util.List;

/**
 * Text embedding client.
 *
 * Retrieval embedders such as Nemotron 3.5 Embed are asymmetric: a search query and a stored
 * passage get different prefixes ("query: " vs "passage: "). Leaving them out costs recall, so
 * callers say which role a text plays:
 *
 *   embed(text)            -> a search QUERY (all retrieval/eval call sites use this)
 *   embedQuery(text)       -> same as embed(text), explicit name
 *   embedPassage(text)     -> one stored PASSAGE/chunk/summary (ingestion)
 *   embedBatch(texts)      -> stored PASSAGES in bulk (ingestion)
 *   embedPassages(texts)   -> same as embedBatch(texts), explicit name
 *
 * Symmetric models (bge-base, nomic) ignore the distinction, so the defaults below keep
 * existing implementations (Ollama) working unchanged.
 *
 * Batch performance (why embedBatch exists):
 *   - 20 chunks x serial embed():   20 HTTP round-trips x ~20ms = ~400ms
 *   - 20 chunks x embedBatch():      1 HTTP round-trip  x ~80ms  = ~80ms
 */
public interface EmbeddingClient {

    /** Embeds a search query. */
    List<Double> embed(String text);

    /** Embeds stored passages in bulk. Default calls embed() serially for backward compatibility. */
    default List<List<Double>> embedBatch(List<String> texts) {
        return texts.stream().map(this::embed).toList();
    }

    default List<Double> embedQuery(String text) {
        return embed(text);
    }

    default List<Double> embedPassage(String text) {
        List<List<Double>> result = embedPassages(List.of(text == null ? "" : text));
        if (result.isEmpty()) {
            throw new IllegalStateException("Empty embedding response");
        }
        return result.get(0);
    }

    default List<List<Double>> embedPassages(List<String> texts) {
        return embedBatch(texts);
    }
}
