package com.example.airagassistant.agent;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Deterministic citation attribution check.
 *
 * An LLM can cite a chunk that exists in the context but does not contain the fact it is
 * attached to (e.g. "503 hp [m3:1]" when the horsepower is in [m3:2]). The judge catches
 * this only sometimes, so this check does it in code:
 *
 *  1. For each [chunk-id] citation, take the claim it supports: the text since the previous
 *     citation, trimmed to the current sentence.
 *  2. Extract the numbers in that claim (503, 3.0, 76900, ...).
 *  3. If the cited chunk contains all of them, keep the citation.
 *  4. Otherwise re-point it to a retrieved chunk that contains all of them (same vehicle
 *     first, then retrieval order). If no chunk does, drop the citation.
 *
 * Claims without numbers are left unchanged: they can't be verified deterministically.
 */
public final class CitationAttributionChecker {

    private static final Pattern CITATION = Pattern.compile("\\[([a-zA-Z0-9\\-]+:\\d+)]");

    /** Numbers not glued to a preceding letter/digit/dot, so "M3" and "S58" are ignored. */
    private static final Pattern NUMBER = Pattern.compile("(?<![A-Za-z0-9.])\\d[\\d,]*(?:\\.\\d+)?");

    /** Sentence end: terminator followed by whitespace (so "3.0L" is not a break), or a newline. */
    private static final Pattern SENTENCE_BREAK = Pattern.compile("[.!?]\\s+|\\n");

    private CitationAttributionChecker() {
    }

    public record Correction(String fromId, String toId, Set<String> claimNumbers) {}

    public record Result(String answer, List<Correction> corrections) {
        public long correctedCount() {
            return corrections.stream().filter(c -> c.toId() != null).count();
        }

        public long droppedCount() {
            return corrections.stream().filter(c -> c.toId() == null).count();
        }
    }

    /**
     * @param answer         LLM answer containing [chunk-id] citations
     * @param chunkTextById  retrieved chunks in retrieval order (id -> text)
     */
    public static Result check(String answer, Map<String, String> chunkTextById) {
        if (answer == null || answer.isBlank() || chunkTextById == null || chunkTextById.isEmpty()) {
            return new Result(answer, List.of());
        }

        Map<String, Set<String>> numbersByChunk = new LinkedHashMap<>();
        chunkTextById.forEach((id, text) -> numbersByChunk.put(id, extractNumbers(text)));

        List<Correction> corrections = new ArrayList<>();
        StringBuilder out = new StringBuilder();
        Matcher m = CITATION.matcher(answer);
        int prevEnd = 0;

        while (m.find()) {
            String citedId = m.group(1);
            Set<String> claimNumbers = extractNumbers(claimText(answer.substring(prevEnd, m.start())));
            String replacement = m.group();

            if (!claimNumbers.isEmpty()
                    && !numbersByChunk.getOrDefault(citedId, Set.of()).containsAll(claimNumbers)) {

                String supportingId = findSupportingChunk(citedId, claimNumbers, numbersByChunk);
                corrections.add(new Correction(citedId, supportingId, claimNumbers));
                replacement = supportingId != null ? "[" + supportingId + "]" : "";
            }

            m.appendReplacement(out, Matcher.quoteReplacement(replacement));
            prevEnd = m.end();
        }
        m.appendTail(out);

        String result = out.toString();
        if (corrections.stream().anyMatch(c -> c.toId() == null)) {
            // tidy whitespace left behind by dropped citations
            result = result.replaceAll(" {2,}", " ").replaceAll(" +([.,;:])", "$1");
        }

        return new Result(result, corrections);
    }

    /** Text the citation supports: from the last sentence break in the segment onwards. */
    static String claimText(String segment) {
        Matcher m = SENTENCE_BREAK.matcher(segment);
        int start = 0;
        while (m.find()) {
            start = m.end();
        }
        return segment.substring(start);
    }

    static Set<String> extractNumbers(String text) {
        Set<String> numbers = new LinkedHashSet<>();
        if (text == null) {
            return numbers;
        }
        Matcher m = NUMBER.matcher(text);
        while (m.find()) {
            String raw = m.group().replace(",", "");
            if (raw.isEmpty()) {
                continue;
            }
            try {
                // normalise so "3.0" and "3" (or "503.00" and "503") compare equal
                numbers.add(new BigDecimal(raw).stripTrailingZeros().toPlainString());
            } catch (NumberFormatException ignored) {
                // not a clean number; skip
            }
        }
        return numbers;
    }

    private static String findSupportingChunk(String citedId,
                                              Set<String> claimNumbers,
                                              Map<String, Set<String>> numbersByChunk) {
        String citedPrefix = docPrefix(citedId);

        // Prefer a chunk from the same document/vehicle, then any retrieved chunk.
        for (boolean sameDocOnly : new boolean[]{true, false}) {
            for (Map.Entry<String, Set<String>> e : numbersByChunk.entrySet()) {
                String id = e.getKey();
                if (id.equals(citedId)) {
                    continue;
                }
                if (sameDocOnly && !docPrefix(id).equals(citedPrefix)) {
                    continue;
                }
                if (e.getValue().containsAll(claimNumbers)) {
                    return id;
                }
            }
        }
        return null;
    }

    private static String docPrefix(String chunkId) {
        int idx = chunkId.lastIndexOf(':');
        return idx >= 0 ? chunkId.substring(0, idx) : chunkId;
    }
}
