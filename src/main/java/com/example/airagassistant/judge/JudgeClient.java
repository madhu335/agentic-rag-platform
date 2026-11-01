package com.example.airagassistant.judge;

import java.util.List;

public interface JudgeClient {

    /** Thorough evaluation (the judge's full-quality mode; for Nemotron: reasoning on). */
    String evaluate(String prompt, List<String> contextChunks);

    /**
     * True when the client has a cheaper fast mode that JudgeService can use as the first stage of the
     * cascade. Clients without one (Halo/gpt-oss always reasons, Claude, OpenAI, default) are judged thorough-only.
     */
    default boolean supportsFastMode() {
        return false;
    }

    /** Fast evaluation (for Nemotron: reasoning off). Defaults to the thorough evaluation. */
    default String evaluateFast(String prompt, List<String> contextChunks) {
        return evaluate(prompt, contextChunks);
    }
}
