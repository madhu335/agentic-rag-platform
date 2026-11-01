package com.example.airagassistant.judge;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Deterministic check: every number in the answer must appear somewhere in the context. No LLM, sub-millisecond.
 * Used as an escalation trigger for the judge cascade and as a risk signal in the orchestrator.
 *
 * judge_compare.py (2026-11-01): caught 3/4 bad answers (invented 0-60 time, wrong sum, answer from the wrong
 * vehicle's context); missed swapped hp/torque (both numbers are in the context); one false alarm on a CORRECT
 * derived total ($83,075 = $81,900 + $1,175). A false alarm only costs one thorough judge call.
 *
 * Ignored before extracting (judge_eval.py replay on 29 real answers, 2026-11-01):
 *  - digits glued to letters: model names like "M3", "A3B"
 *  - list markers at line start: "1. ", "2) ", "- 3. "
 *  - references to context entries: "Entry 1", "context 5", "Entries 3, 4, and 5", "item 2"
 *    (the RAG prompt numbers the chunks, so answers cite them by position)
 * These removed 3 of 4 false alarms on real answers. "1,175" and "1175" compare equal, as do "503" and "503.0".
 */
public final class NumbersCheck {

    private static final Pattern NUMBER = Pattern.compile("(?<![A-Za-z\\d])\\d[\\d,]*(?:\\.\\d+)?");
    private static final Pattern LIST_MARKER = Pattern.compile("(?m)^\\s*(?:[-*]\\s*)?\\d+[.)]\\s");
    private static final Pattern ENTRY_REFERENCE = Pattern.compile(
            "(?i)\\b(?:entry|entries|context|item|items|chunk|chunks|source|sources|excerpt|excerpts)"
                    + "\\s*#?\\s*(?:\\d+\\s*(?:,\\s*and|,|&|and|-|to)?\\s*)+");

    private NumbersCheck() {
    }

    /** Numbers in the answer that do not appear in any context chunk, in the order found. */
    public static List<String> missingNumbers(String answer, List<String> contextChunks) {
        Set<BigDecimal> context = new TreeSet<>(extract(contextChunks == null ? "" : String.join(" ", contextChunks)));
        List<String> missing = new ArrayList<>();
        Set<BigDecimal> seen = new TreeSet<>();
        for (BigDecimal n : extract(stripNonFacts(answer))) {
            if (!context.contains(n) && seen.add(n)) {
                missing.add(n.toPlainString());
            }
        }
        return missing;
    }

    /** Removes list markers and context-entry references, which are formatting, not factual numbers. */
    static String stripNonFacts(String answer) {
        if (answer == null) {
            return "";
        }
        String out = LIST_MARKER.matcher(answer).replaceAll(" ");
        return ENTRY_REFERENCE.matcher(out).replaceAll(" ");
    }

    private static List<BigDecimal> extract(String text) {
        List<BigDecimal> out = new ArrayList<>();
        if (text == null || text.isEmpty()) {
            return out;
        }
        Matcher m = NUMBER.matcher(text);
        while (m.find()) {
            String s = m.group().replace(",", "");
            if (s.endsWith(".")) {
                s = s.substring(0, s.length() - 1);
            }
            try {
                out.add(new BigDecimal(s).stripTrailingZeros());
            } catch (NumberFormatException ignored) {
                // not a number after cleanup; skip
            }
        }
        return out;
    }
}
