package com.example.airagassistant.judge;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class JudgeScoreNormalizationTest {

    @Test
    void leavesDecimalScoreAlone() {
        double score = normalize(0.7);
        assertEquals(0.7, score, 0.0001);
    }

    @Test
    void convertsTenScaleToDecimal() {
        double score = normalize(7.0);
        assertEquals(0.7, score, 0.0001);
    }

    private double normalize(double score) {
        return score > 1.0 ? score / 10.0 : score;
    }
}