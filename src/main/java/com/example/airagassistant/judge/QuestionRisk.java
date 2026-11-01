package com.example.airagassistant.judge;

import java.util.Locale;
import java.util.regex.Pattern;

/**
 * Question-level risk: questions whose answers depend on numbers, thresholds or rankings are where a wrong
 * answer is most likely and most costly (judge_compare.py / judge_eval.py, 2026-11-01: invented mpg figures,
 * wrong totals, swapped hp/torque, unsupported "lowest cost" claims).
 *
 * Risky when the question contains a number (not a model name like "M3" or a year like 2025), a threshold
 * phrase ("over", "under", "at least", ...) or a comparative/superlative ("best", "most", "lowest", "rank", "vs", ...). Comparison questions mostly already
 * go to the AGENT route via QueryRouter; this catches the ones that stay on RAG.
 */
public final class QuestionRisk {

    /** A standalone number: not glued to letters ("M3", "CX-5" model names) and not a 4-digit year (2025). */
    private static final Pattern NUMBER = Pattern.compile("(?<![A-Za-z\\d-])\\d[\\d,]*(?:\\.\\d+)?(?![A-Za-z\\d])");
    private static final Pattern YEAR = Pattern.compile("^(?:19|20)\\d{2}$");
    private static final Pattern THRESHOLD = Pattern.compile(
            "\\b(over|under|above|below|at least|at most|more than|less than|fewer than|greater than|"
                    + "up to|between|exceed|exceeds|within)\\b");
    private static final Pattern RANKING = Pattern.compile(
            "\\b(best|worst|most|least|top|lowest|highest|cheapest|fastest|slowest|quickest|largest|smallest|"
                    + "biggest|longest|shortest|better|worse|cheaper|faster|rank|ranked|ranking|rankings|compare|"
                    + "compared|comparison|versus|vs)\\b");

    private QuestionRisk() {
    }

    /** Returns the risk reason ("question_numeric", "question_threshold", "question_ranking") or null. */
    public static String reason(String question) {
        if (question == null || question.isBlank()) {
            return null;
        }
        String q = question.toLowerCase(Locale.ROOT);
        var m = NUMBER.matcher(q);
        while (m.find()) {
            if (!YEAR.matcher(m.group()).matches()) {
                return "question_numeric";
            }
        }
        if (THRESHOLD.matcher(q).find()) {
            return "question_threshold";
        }
        if (RANKING.matcher(q).find()) {
            return "question_ranking";
        }
        return null;
    }
}
