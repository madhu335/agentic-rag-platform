package com.example.airagassistant.domain.article;

// ── Ratings ───────────────────────────────────────────────────────────────────
public record ArticleRating(
        double overall,          // 0-10
        double performance,
        double comfort,
        double interior,
        double technology,
        double practicality,
        double value,
        String verdictLabel      // "Recommended", "Top Pick", "Best Buy"
) {}
