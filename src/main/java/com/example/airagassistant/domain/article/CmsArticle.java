package com.example.airagassistant.domain.article;

import java.util.List;

// ── Root ──────────────────────────────────────────────────────────────────────
public record CmsArticle(

        // identity
        String articleId,        // "motortrend-bmw-m3-2025-review"
        String slug,             // "2025-bmw-m3-competition-review"
        String title,            // "2025 BMW M3 Competition Review: The Benchmark Returns"
        String articleType,      // "first_drive", "long_term", "comparison", "buyers_guide"
        String author,
        String publishDate,      // "2025-03-15"
        String updatedDate,

        // vehicle references — many-to-many
        List<VehicleReference> vehicles,

        // structured content
        String verdict,          // 2-3 sentence editor summary
        String pros,             // comma-separated or narrative
        String cons,
        String bodyText,         // full article body — long form

        // structured sections
        List<ArticleSection> sections,

        // ratings
        ArticleRating rating,

        // metadata
        List<String> tags,       // "sports sedan", "track car", "daily driver"
        String priceAsTestedMsrp,
        int wordCount
) {}

