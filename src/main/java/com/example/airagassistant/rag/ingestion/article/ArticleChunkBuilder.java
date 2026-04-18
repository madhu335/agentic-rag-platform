package com.example.airagassistant.rag.ingestion.article;

import com.example.airagassistant.domain.article.ArticleSection;
import com.example.airagassistant.domain.article.CmsArticle;
import com.example.airagassistant.domain.article.VehicleReference;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.StringJoiner;
import java.util.stream.Collectors;

/**
 * Converts a CmsArticle into semantic chunks.
 *
 * Key differences from VehicleChunkBuilder:
 *
 * 1. Body text is LONG (2000-5000 words) — cannot embed as one chunk.
 *    Split into overlapping windows so boundary sentences aren't missed.
 *
 * 2. Many-to-many vehicle references — every chunk carries vehicle
 *    identity anchors for ALL referenced vehicles so retrieval works
 *    regardless of which vehicle the user queries by.
 *
 * 3. Opinion vs fact — verdict/pros/cons chunks are semantic opinion,
 *    rating chunks are structured numeric. Both need different prose styles.
 *
 * 4. Temporal anchor — every chunk includes publish date so "recent reviews"
 *    queries retrieve newer articles over older ones.
 *
 * chunk_index assignment:
 *   1   = identity + verdict (always retrieve together — the summary)
 *   2   = ratings narrative (structured scores as prose)
 *   3   = pros and cons
 *   4   = vehicle references summary (who is in this article)
 *   10+ = article sections (one per section — Performance, Interior, etc.)
 *   50+ = body text windows (overlapping chunks of long-form text)
 */
@Component
public class ArticleChunkBuilder {

    // Recursive semantic splitter settings
    // Tries paragraph → sentence → word boundaries in order
    // Only falls back to the next level if the current chunk is still too large
    private static final int MAX_CHUNK_WORDS    = 600;  // hard ceiling per chunk
    private static final int MIN_CHUNK_WORDS    = 100;  // don't create tiny fragments
    private static final int OVERLAP_SENTENCES  = 2;    // sentences shared between chunks

    public record ArticleChunk(
            String articleId,
            int    chunkIndex,
            String chunkType,
            String text
    ) {}

    public List<ArticleChunk> buildChunks(CmsArticle article) {
        List<ArticleChunk> chunks = new ArrayList<>();

        // Anchor: repeated in every chunk
        // Includes ALL vehicle names so any vehicle query retrieves this article
        String vehicleAnchor = buildVehicleAnchor(article);
        String timeAnchor    = article.publishDate() != null
                ? "Published " + article.publishDate() + ". " : "";
        String anchor        = vehicleAnchor + timeAnchor;

        // Fixed chunks
        chunks.add(chunk(article, 1,  "identity_verdict",  buildIdentityVerdict(anchor, article)));
        chunks.add(chunk(article, 2,  "ratings",           buildRatings(anchor, article)));
        chunks.add(chunk(article, 3,  "pros_cons",         buildProsCons(anchor, article)));
        chunks.add(chunk(article, 4,  "vehicle_references",buildVehicleReferences(anchor, article)));

        // Article sections — one chunk per section (index 10+)
        // Same principle as maintenance intervals — each section answers different questions
        // "Performance" section answers "how does it drive?"
        // "Interior" section answers "what is the cabin like?"
        if (article.sections() != null) {
            int idx = 10;
            for (ArticleSection section : article.sections()) {
                if (section.content() != null && !section.content().isBlank()) {
                    chunks.add(chunk(article, idx++, "section_" + normalize(section.sectionTitle()),
                            buildSection(anchor, section)));
                }
            }
        }

        // Body text sliding windows — index 50+
        // Long-form text needs overlapping windows so no sentence is cut off
        // at a chunk boundary without being retrievable
        if (article.bodyText() != null && !article.bodyText().isBlank()) {
            List<String> windows = slidingWindows(article.bodyText());
            int idx = 50;
            for (int i = 0; i < windows.size(); i++) {
                String windowAnchor = anchor + "Article excerpt " + (i + 1) + " of " + windows.size() + ". ";
                chunks.add(chunk(article, idx++, "body_window", windowAnchor + windows.get(i)));
            }
        }

        return chunks;
    }

    // ── Chunk builders ────────────────────────────────────────────────────────

    private String buildIdentityVerdict(String anchor, CmsArticle article) {
        // This chunk answers: "what is this article about?" and "what is the verdict?"
        // It's the first chunk retrieved for any article — the summary
        StringJoiner sj = new StringJoiner(" ");
        sj.add(anchor);
        sj.add("Article: " + safe(article.title()) + ".");
        sj.add("Type: " + safe(article.articleType()) + ".");
        sj.add("Author: " + safe(article.author()) + ".");
        if (article.priceAsTestedMsrp() != null)
            sj.add("Price as tested: " + article.priceAsTestedMsrp() + ".");
        if (article.verdict() != null && !article.verdict().isBlank())
            sj.add("Verdict: " + article.verdict());
        if (article.rating() != null)
            sj.add("Overall rating: " + article.rating().overall() + " out of 10.");
        if (article.rating() != null && article.rating().verdictLabel() != null)
            sj.add(article.rating().verdictLabel() + ".");
        return sj.toString();
    }

    private String buildRatings(String anchor, CmsArticle article) {
        // Convert numeric ratings to prose — same principle as vehicle rankings
        // "performance: 9.2" embeds poorly
        // "MotorTrend rated performance 9.2 out of 10" embeds well
        if (article.rating() == null)
            return anchor + "No ratings available for this article.";

        var r = article.rating();
        StringJoiner sj = new StringJoiner(" ");
        sj.add(anchor + "MotorTrend ratings:");
        sj.add("Overall score " + r.overall() + " out of 10.");
        if (r.performance()  > 0) sj.add("Performance rated " + r.performance() + " out of 10.");
        if (r.comfort()      > 0) sj.add("Comfort and ride rated " + r.comfort() + " out of 10.");
        if (r.interior()     > 0) sj.add("Interior quality rated " + r.interior() + " out of 10.");
        if (r.technology()   > 0) sj.add("Technology and infotainment rated " + r.technology() + " out of 10.");
        if (r.practicality() > 0) sj.add("Practicality rated " + r.practicality() + " out of 10.");
        if (r.value()        > 0) sj.add("Value for money rated " + r.value() + " out of 10.");
        if (r.verdictLabel() != null)
            sj.add("Verdict: " + r.verdictLabel() + ".");
        return sj.toString();
    }

    private String buildProsCons(String anchor, CmsArticle article) {
        // Pros/cons answer: "what do reviewers like/dislike?"
        // Also critical for comparison queries — "which car has better interior?"
        StringJoiner sj = new StringJoiner(" ");
        sj.add(anchor + "Expert assessment:");
        if (article.pros() != null && !article.pros().isBlank())
            sj.add("Strengths: " + article.pros() + ".");
        if (article.cons() != null && !article.cons().isBlank())
            sj.add("Weaknesses: " + article.cons() + ".");
        if (article.tags() != null && !article.tags().isEmpty())
            sj.add("Best suited for: " + String.join(", ", article.tags()) + ".");
        return sj.toString();
    }

    private String buildVehicleReferences(String anchor, CmsArticle article) {
        // Critical for many-to-many — answers "which articles mention BMW M3 and Porsche 911?"
        // Also enables cross-article comparison queries
        if (article.vehicles() == null || article.vehicles().isEmpty())
            return anchor + "Vehicle references not specified.";

        StringJoiner sj = new StringJoiner(" ");
        sj.add(anchor + "Vehicles featured in this article:");
        for (VehicleReference v : article.vehicles()) {
            sj.add("The " + v.year() + " " + v.make() + " " + v.model()
                    + (v.trim() != null ? " " + v.trim() : "")
                    + " (" + safe(v.role()) + ").");
        }

        // Primary vehicle highlight — helps single-vehicle queries
        article.vehicles().stream()
                .filter(v -> "primary".equals(v.role()))
                .findFirst()
                .ifPresent(v -> sj.add(
                        "Primary subject: " + v.year() + " " + v.make() + " " + v.model() + "."));

        return sj.toString();
    }

    private String buildSection(String anchor, ArticleSection section) {
        // One chunk per section — same principle as maintenance intervals
        // "Performance" section answers "how does it drive?"
        // "Interior" answers "what is the cabin like?"
        // Each is a different user question → separate chunk
        //
        // UAC-first: section title must include vocabulary users actually query with.
        // "Performance" alone doesn't embed "track" — use enriched label.
        String enrichedTitle = enrichSectionTitle(section.sectionTitle());
        return anchor
                + "Section: " + enrichedTitle + ". "
                + section.content();
    }

    /**
     * Enriches section titles with query vocabulary so embeddings match user queries.
     *
     * UAC examples:
     *   "How does it perform on track?"     → needs "track" in Performance chunk
     *   "What is the cabin like?"           → needs "cabin" or "interior quality"
     *   "Is the infotainment easy to use?"  → needs "infotainment" in Technology chunk
     *   "Is it good value for money?"       → needs "value for money" in Value chunk
     */
    private String enrichSectionTitle(String title) {
        if (title == null) return "General";
        return switch (title.toLowerCase().trim()) {
            case "performance"  -> "Performance and track testing";
            case "interior"     -> "Interior cabin quality and comfort";
            case "technology"   -> "Technology infotainment and features";
            case "value"        -> "Value pricing and cost of ownership";
            case "safety"       -> "Safety ratings and driver assistance";
            case "design"       -> "Design styling and exterior";
            case "driving"      -> "Driving dynamics handling and ride";
            case "efficiency"   -> "Fuel efficiency and range";
            default             -> title;  // pass through for custom section names
        };
    }

    // ── Sliding window for body text ──────────────────────────────────────────

    /**
     * Recursive semantic text splitter.
     *
     * Strategy — try semantic boundaries in priority order:
     *   Level 1: paragraph boundaries (\n\n)
     *   Level 2: sentence boundaries (. ! ?)
     *   Level 3: clause boundaries (, ; :)
     *   Level 4: word boundaries (fallback — same as old sliding window)
     *
     * Why this is better than fixed word windows:
     *
     *   Fixed window cuts here (exactly 600 words):
     *     "...The M3 feels engineered to a consistent stan | dard — every system..."
     *      mid-word, mid-thought, embedding captures incomplete concept
     *
     *   Recursive splitter cuts here (nearest sentence end before 600 words):
     *     "...The M3 feels engineered to a consistent standard — every system works."
     *      ↑ complete sentence, complete thought, embedding captures full concept
     *
     * Overlap is sentence-based not word-based:
     *   Last 2 sentences of chunk N = first 2 sentences of chunk N+1
     *   → A thought spanning a boundary always appears complete in at least one chunk
     *   → Same guarantee as word overlap but semantically cleaner
     *
     * Recursive means: if a paragraph is still > MAX_CHUNK_WORDS after splitting
     * at paragraph boundaries, split that paragraph at sentence boundaries.
     * If a sentence is > MAX_CHUNK_WORDS, split at clause boundaries. Etc.
     */
    private List<String> slidingWindows(String text) {
        if (wordCount(text) <= MAX_CHUNK_WORDS) {
            return List.of(text.trim());
        }
        // Start recursive split at paragraph level
        return recursiveSplit(text, SEPARATORS, 0);
    }

    // Separator hierarchy — try each level before falling back to the next
    private static final String[] SEPARATORS = {
            "\n\n",          // Level 1: paragraphs
            "(?<=[.!?])\s+",  // Level 2: sentence endings
            "(?<=[,;:])\s+",  // Level 3: clause boundaries
            "\s+"             // Level 4: word boundaries (last resort)
    };

    private List<String> recursiveSplit(String text, String[] separators, int level) {
        if (wordCount(text) <= MAX_CHUNK_WORDS) {
            return List.of(text.trim());
        }

        if (level >= separators.length) {
            // Hit word-level fallback — just truncate at MAX_CHUNK_WORDS
            return hardSplit(text);
        }

        // Split at this level's separator
        String[] parts = text.split(separators[level]);
        List<String> chunks   = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        List<String>  currentSentences = new ArrayList<>();

        for (String part : parts) {
            String candidate = current.isEmpty()
                    ? part
                    : current + separators[level].replace("(?<=[.!?])\\s+", " ")
                    .replace("(?<=[,;:])\\s+", " ")
                    .replace("\\s+", " ")
                    .replace("\\n\\n", "\n\n") + part;

            if (wordCount(candidate) <= MAX_CHUNK_WORDS) {
                // Still fits — accumulate
                current = new StringBuilder(candidate);
                currentSentences.add(part);
            } else {
                // Current chunk is full
                if (!current.isEmpty()) {
                    String chunk = current.toString().trim();
                    if (wordCount(chunk) > MAX_CHUNK_WORDS) {
                        // Chunk itself is too big — recurse deeper
                        chunks.addAll(recursiveSplit(chunk, separators, level + 1));
                    } else if (wordCount(chunk) >= MIN_CHUNK_WORDS) {
                        chunks.add(chunk);
                    } else if (!chunks.isEmpty()) {
                        // Too small — merge into previous chunk
                        chunks.set(chunks.size() - 1,
                                chunks.get(chunks.size() - 1) + " " + chunk);
                    } else {
                        chunks.add(chunk);
                    }
                }

                // Start new chunk with overlap from end of previous
                List<String> overlapSentences = currentSentences.size() >= OVERLAP_SENTENCES
                        ? currentSentences.subList(
                        currentSentences.size() - OVERLAP_SENTENCES,
                        currentSentences.size())
                        : currentSentences;

                String overlap = String.join(" ", overlapSentences);
                current = new StringBuilder(overlap.isEmpty() ? part : overlap + " " + part);
                currentSentences = new ArrayList<>(overlapSentences);
                currentSentences.add(part);
            }
        }

        // Add remaining text
        if (!current.isEmpty()) {
            String remaining = current.toString().trim();
            if (!remaining.isBlank()) {
                if (wordCount(remaining) > MAX_CHUNK_WORDS) {
                    chunks.addAll(recursiveSplit(remaining, separators, level + 1));
                } else {
                    chunks.add(remaining);
                }
            }
        }

        return chunks.isEmpty() ? List.of(text.trim()) : chunks;
    }

    /**
     * Last-resort hard split at word boundaries when no semantic separator works.
     * Produces overlapping word windows — same as old approach but only used
     * when the text has no punctuation or paragraph breaks at all.
     */
    private List<String> hardSplit(String text) {
        String[] words = text.split("\\s+");
        List<String> windows = new ArrayList<>();
        int overlap = 50; // words
        int start = 0;
        while (start < words.length) {
            int end = Math.min(start + MAX_CHUNK_WORDS, words.length);
            windows.add(String.join(" ", java.util.Arrays.copyOfRange(words, start, end)));
            if (end == words.length) break;
            start += (MAX_CHUNK_WORDS - overlap);
        }
        return windows;
    }

    private int wordCount(String text) {
        if (text == null || text.isBlank()) return 0;
        return text.trim().split("\\s+").length;
    }

    // ── Anchor builder ────────────────────────────────────────────────────────

    /**
     * Builds the vehicle identity anchor for every chunk.
     *
     * For many-to-many articles ALL vehicles are in the anchor:
     *   "BMW M3 Competition, Porsche 911 Carrera S, Mercedes-AMG C63 comparison review."
     *
     * This means a query for ANY referenced vehicle retrieves this article.
     * Without this anchor, only the primary vehicle's chunks would be retrieved.
     */
    private String buildVehicleAnchor(CmsArticle article) {
        if (article.vehicles() == null || article.vehicles().isEmpty())
            return "MotorTrend article. ";

        String vehicleList = article.vehicles().stream()
                .map(v -> v.year() + " " + v.make() + " " + v.model()
                        + (v.trim() != null ? " " + v.trim() : ""))
                .collect(Collectors.joining(", "));

        // Include vehicleId values so downstream consumers (ArticleSubAgent)
        // can extract them from retrieved chunk text without needing a separate
        // lookup table. Format: vehicleId:xxx for each vehicle.
        String vehicleIdList = article.vehicles().stream()
                .filter(v -> v.vehicleId() != null && !v.vehicleId().isBlank())
                .map(v -> "vehicleId:" + v.vehicleId())
                .collect(Collectors.joining(", "));

        String type = article.articleType() != null
                ? article.articleType().replace("_", " ") : "review";

        String anchor = "MotorTrend " + type + " featuring " + vehicleList + ". ";
        if (!vehicleIdList.isEmpty()) {
            anchor += "Vehicles: " + vehicleIdList + ". ";
        }
        return anchor;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private ArticleChunk chunk(CmsArticle article, int idx, String type, String text) {
        return new ArticleChunk(article.articleId(), idx, type, text);
    }

    private String safe(Object o) {
        return o == null ? "N/A" : String.valueOf(o).trim();
    }

    private String normalize(String title) {
        if (title == null) return "unknown";
        return title.toLowerCase().replaceAll("[^a-z0-9]", "_");
    }
}
