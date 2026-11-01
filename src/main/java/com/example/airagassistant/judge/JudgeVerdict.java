package com.example.airagassistant.judge;

/**
 * Single verdict returned by the judge. Replaces the separate grounded/correct/complete booleans as the
 * source of truth: the booleans could contradict each other and the score, and "grounded" was ambiguous
 * (judge_compare.py, 2026-11-01: both Nemotron-30B and gpt-oss-120b split on it; with this contract both
 * scored 24/24 decisions, 0 verdict/score conflicts).
 */
public enum JudgeVerdict {
    /** Every claim supported and correct, question answered (or honestly "not in context"). */
    PASS,
    /** States a fact/number/detail the context does not state, or attaches a number to the wrong quantity. */
    UNSUPPORTED,
    /** Facts used are in the context, but something is derived wrongly (total, comparison, conclusion). */
    INCORRECT,
    /** Everything stated is supported and correct, but part of the question is left unanswered. */
    INCOMPLETE,
    /** The judge could not produce a usable verdict (timeout, unparseable output, provider down). */
    UNAVAILABLE;

    /** Lenient parse of the model's verdict string. Returns null when it is not a known verdict. */
    public static JudgeVerdict parse(String raw) {
        if (raw == null) {
            return null;
        }
        String v = raw.trim().toUpperCase().replace(' ', '_');
        for (JudgeVerdict verdict : values()) {
            if (verdict != UNAVAILABLE && verdict.name().equals(v)) {
                return verdict;
            }
        }
        return null;
    }

    public boolean accepted() {
        return this == PASS;
    }
}
