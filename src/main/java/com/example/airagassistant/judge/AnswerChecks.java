package com.example.airagassistant.judge;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Deterministic answer checks (no LLM, sub-millisecond). Any check that fires means the answer may contain
 * something the context does not support, so it should get the thorough judge:
 *
 *   numbers  a number in the answer is not in the context                      ({@link NumbersCheck})
 *   quotes   text the answer puts in quotation marks is not in the context
 *            (and not in the question) - catches invented "quotes", e.g. completing a truncated excerpt
 *   short    the answer is under 25 characters and is not an honest "not in context" answer
 *
 * judge_eval.py replay on 29 real answers (2026-11-01): the fast judge passed invented non-numeric claims
 * (S2-004 "razor-sharp handling", a one-word answer "M3"); the quote and short checks catch those.
 */
public final class AnswerChecks {

    private static final Pattern QUOTE = Pattern.compile("[\"“]([^\"“”\\n]{8,80})[\"”]");
    private static final Pattern HONEST_NO_ANSWER = Pattern.compile(
            "(?i)(don.?t know|not (?:mention|say|provide|contain|include|specif)|no information|"
                    + "context (?:does not|doesn.?t|lacks))");
    private static final int SHORT_ANSWER_CHARS = 25;

    private AnswerChecks() {
    }

    public record Result(List<String> missingNumbers, List<String> unsupportedQuotes, boolean tooShort) {

        public boolean fired() {
            return !missingNumbers.isEmpty() || !unsupportedQuotes.isEmpty() || tooShort;
        }

        /** Short label for logs/traces, e.g. "numbers_not_in_context:52,54" or "quote_not_in_context". */
        public String reason() {
            List<String> parts = new ArrayList<>();
            if (!missingNumbers.isEmpty()) {
                parts.add("numbers_not_in_context:" + String.join(",", missingNumbers));
            }
            if (!unsupportedQuotes.isEmpty()) {
                parts.add("quote_not_in_context");
            }
            if (tooShort) {
                parts.add("short_answer");
            }
            return parts.isEmpty() ? "none" : String.join("+", parts);
        }
    }

    public static Result check(String question, String answer, List<String> contextChunks) {
        String context = contextChunks == null ? "" : String.join(" ", contextChunks);
        return new Result(
                NumbersCheck.missingNumbers(answer, contextChunks),
                unsupportedQuotes(question, answer, context),
                tooShort(answer)
        );
    }

    static List<String> unsupportedQuotes(String question, String answer, String context) {
        List<String> out = new ArrayList<>();
        if (answer == null) {
            return out;
        }
        String ctx = normalize(context);
        String q = normalize(question);
        Matcher m = QUOTE.matcher(answer);
        while (m.find()) {
            String quoted = m.group(1);
            if (quoted.trim().split("\\s+").length < 2) {
                continue;                                   // single words are usually emphasis, not quotes
            }
            String n = normalize(quoted);
            if (!n.isEmpty() && !ctx.contains(n) && !q.contains(n)) {
                out.add(quoted.trim());
            }
        }
        return out;
    }

    static boolean tooShort(String answer) {
        String a = answer == null ? "" : answer.strip();
        return a.length() < SHORT_ANSWER_CHARS && !HONEST_NO_ANSWER.matcher(a).find();
    }

    private static String normalize(String s) {
        if (s == null) {
            return "";
        }
        String out = s.toLowerCase(Locale.ROOT)
                .replace('‑', '-').replace('‐', '-').replace('–', '-').replace('—', '-')
                .replace('’', '\'').replace(' ', ' ').replace(' ', ' ')
                .replaceAll("\\s+", " ")
                .strip();
        // drop trailing punctuation/ellipsis so "engineering." matches "engineering"
        return out.replaceAll("[\\s.,;:!?…]+$", "");
    }
}
