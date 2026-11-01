package com.example.airagassistant.judge;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Deterministic escalation trigger for the judge cascade: every number in the answer must appear somewhere
 * in the context. No LLM, sub-millisecond.
 *
 * judge_compare.py (2026-11-01): caught 3/4 bad answers (invented 0-60 time, wrong sum, answer from the wrong
 * vehicle's context); missed swapped hp/torque (both numbers are in the context); one false alarm on a
 * CORRECT derived total ($83,075 = $81,900 + $1,175). A false alarm only costs one thorough judge call,
 * which then accepts the answer. The fast judge and this check catch different failures; together they
 * caught all four.
 *
 * Digits glued to letters are ignored (model names like "M3", "A3B"). "1,175" and "1175" compare equal,
 * as do "503" and "503.0".
 */
public final class NumbersCheck {

    private static final Pattern NUMBER = Pattern.compile("(?<![A-Za-z\\d])\\d[\\d,]*(?:\\.\\d+)?");

    private NumbersCheck() {
    }

    /** Numbers in the answer that do not appear in any context chunk, in the order found. */
    public static List<String> missingNumbers(String answer, List<String> contextChunks) {
        Set<BigDecimal> context = extract(contextChunks == null ? "" : String.join(" ", contextChunks));
        List<String> missing = new ArrayList<>();
        Set<BigDecimal> seen = new TreeSet<>();
        for (BigDecimal n : extractOrdered(answer)) {
            if (!context.contains(n) && seen.add(n)) {
                missing.add(n.toPlainString());
            }
        }
        return missing;
    }

    private static Set<BigDecimal> extract(String text) {
        return new TreeSet<>(extractOrdered(text));
    }

    private static List<BigDecimal> extractOrdered(String text) {
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
