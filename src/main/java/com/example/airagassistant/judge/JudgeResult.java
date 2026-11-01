package com.example.airagassistant.judge;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Judge outcome. {@code verdict} is the source of truth; grounded/correct/complete are derived from it so
 * existing callers, logs and LangSmith attributes keep working.
 *
 * @param verdict PASS | UNSUPPORTED | INCORRECT | INCOMPLETE | UNAVAILABLE
 * @param score   0.0 - 1.0 (1.0 = fully correct and supported)
 * @param reason  short explanation from the judge (or "judge_unavailable")
 * @param stage   how the verdict was reached: fast_pass, fast_reject, escalated:&lt;why&gt;, thorough, legacy, unavailable
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record JudgeResult(
        JudgeVerdict verdict,
        double score,
        String reason,
        String stage
) {

    public static final String UNAVAILABLE_REASON = "judge_unavailable";

    public JudgeResult {
        if (verdict == null) {
            verdict = JudgeVerdict.UNAVAILABLE;
        }
    }

    /**
     * Backward-compatible constructor for code that still builds results from the old booleans.
     * Precedence: unavailable -> UNSUPPORTED (not grounded) -> INCORRECT -> INCOMPLETE -> PASS.
     */
    public JudgeResult(boolean grounded, boolean correct, boolean complete, double score, String reason) {
        this(legacyVerdict(grounded, correct, complete, reason), score, reason, "legacy");
    }

    /**
     * JSON entry point. Accepts the new shape ({"verdict",...}) and the old one ({"grounded","correct",
     * "complete",...}) so results serialized before the verdict change still read back correctly.
     */
    @JsonCreator
    public static JudgeResult fromJson(
            @JsonProperty("verdict") JudgeVerdict verdict,
            @JsonProperty("score") Double score,
            @JsonProperty("reason") String reason,
            @JsonProperty("stage") String stage,
            @JsonProperty("grounded") Boolean grounded,
            @JsonProperty("correct") Boolean correct,
            @JsonProperty("complete") Boolean complete
    ) {
        double s = score == null ? 0.0 : score;
        if (verdict == null && grounded != null && correct != null) {
            return new JudgeResult(grounded, correct, complete == null || complete, s, reason)
                    .withStage(stage == null ? "legacy" : stage);
        }
        return new JudgeResult(verdict, s, reason, stage);
    }

    public static JudgeResult unavailable() {
        return new JudgeResult(JudgeVerdict.UNAVAILABLE, 0.0, UNAVAILABLE_REASON, "unavailable");
    }

    public JudgeResult withStage(String newStage) {
        return new JudgeResult(verdict, score, reason, newStage);
    }

    @JsonIgnore
    public boolean accepted() {
        return verdict.accepted();
    }

    @JsonIgnore
    public boolean isUnavailable() {
        return verdict == JudgeVerdict.UNAVAILABLE;
    }

    /** False only when the answer states something the context does not support. */
    @JsonProperty("grounded")
    public boolean grounded() {
        return verdict == JudgeVerdict.PASS || verdict == JudgeVerdict.INCORRECT || verdict == JudgeVerdict.INCOMPLETE;
    }

    /** True when nothing stated is wrong (PASS, or INCOMPLETE = correct but partial). */
    @JsonProperty("correct")
    public boolean correct() {
        return verdict == JudgeVerdict.PASS || verdict == JudgeVerdict.INCOMPLETE;
    }

    /** False only for INCOMPLETE (and UNAVAILABLE). */
    @JsonProperty("complete")
    public boolean complete() {
        return verdict != JudgeVerdict.INCOMPLETE && verdict != JudgeVerdict.UNAVAILABLE;
    }

    private static JudgeVerdict legacyVerdict(boolean grounded, boolean correct, boolean complete, String reason) {
        if (UNAVAILABLE_REASON.equalsIgnoreCase(reason)) {
            return JudgeVerdict.UNAVAILABLE;
        }
        if (!grounded) {
            return JudgeVerdict.UNSUPPORTED;
        }
        if (!correct) {
            return JudgeVerdict.INCORRECT;
        }
        if (!complete) {
            return JudgeVerdict.INCOMPLETE;
        }
        return JudgeVerdict.PASS;
    }
}
