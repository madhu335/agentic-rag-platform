package com.example.airagassistant.domain.article;

// ── Article sections ──────────────────────────────────────────────────────────
// Each named section becomes its own chunk — same principle as maintenance intervals
public record ArticleSection(
        String sectionTitle,     // "Performance", "Interior", "Technology", "Value"
        String content,          // section body text
        int    orderIndex        // position in article
) {}
