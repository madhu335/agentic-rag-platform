package com.example.airagassistant.agentic.multi.agents;

import com.example.airagassistant.LlmClient;
import com.example.airagassistant.agentic.multi.SubAgentResult;
import com.example.airagassistant.agentic.tools.vehicle.FetchVehicleSpecsTool;
import com.example.airagassistant.domain.article.service.ArticleRagService;
import com.example.airagassistant.judge.JudgeResult;
import com.example.airagassistant.judge.JudgeService;
import com.example.airagassistant.rag.RagAnswerService;
import com.example.airagassistant.rag.RetrievalMode;
import com.example.airagassistant.rag.SearchHit;
import com.example.airagassistant.rag.VectorRecord;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class ArticleSubAgentTest {

    private ArticleRagService articleRagService;
    private RagAnswerService ragAnswerService;
    private FetchVehicleSpecsTool fetchSpecsTool;
    private LlmClient llm;
    private JudgeService judgeService;
    private ArticleSubAgent agent;

    @BeforeEach
    void setUp() {
        articleRagService = mock(ArticleRagService.class);
        ragAnswerService = mock(RagAnswerService.class);
        fetchSpecsTool = mock(FetchVehicleSpecsTool.class);
        llm = mock(LlmClient.class);
        judgeService = mock(JudgeService.class);
        agent = new ArticleSubAgent(articleRagService, ragAnswerService, fetchSpecsTool, llm, judgeService);

        // Default judge stub — returns passing score so tests don't fail on judge threshold
        when(judgeService.evaluate(any(), any(), any()))
                .thenReturn(new JudgeResult(true, true, true, 0.9, "ok"));
    }

    // ─── Test data builders ───────────────────────────────────────────────

    private ArticleRagService.ArticleSearchHit articleHit(String articleId, String chunkId,
                                                          int rank, double score, String excerpt) {
        return new ArticleRagService.ArticleSearchHit(chunkId, articleId, 1, rank, score, excerpt);
    }

    private SearchHit searchHit(String id, String text, double score) {
        return new SearchHit(new VectorRecord(1L, 1, id, text, List.of()), score);
    }

    private FetchVehicleSpecsTool.SpecChunk specChunk(String chunkId, String text) {
        return new FetchVehicleSpecsTool.SpecChunk(chunkId, text, 0.9);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Routing tests — verify the task/args-based router picks the right path
    // ═══════════════════════════════════════════════════════════════════════

    @Nested
    class Routing {

        @Test
        void routesToSingleArticleAskWhenArticleIdPresent() {
            var ragResult = new RagAnswerService.RagResult(
                    "M3 performance is excellent", List.of(), List.of("chunk:1"),
                    List.of("chunk:1"), 1, 0.85);

            when(articleRagService.askArticle(eq("motortrend-bmw-m3-2025-review"),
                    eq("How does it perform?"), eq(5), eq(RetrievalMode.HYBRID)))
                    .thenReturn(ragResult);

            Map<String, Object> args = new LinkedHashMap<>();
            args.put("articleId", "motortrend-bmw-m3-2025-review");
            args.put("question", "How does it perform?");

            SubAgentResult result = agent.execute("ask about performance", args);

            assertTrue(result.success());
            assertEquals("M3 performance is excellent", result.summary());
            assertEquals("ask_article", result.metadata().get("operation"));
            verify(articleRagService).askArticle(any(), any(), anyInt(), any());
            verifyNoInteractions(llm);  // askArticle path uses RagAnswerService internally
        }

        @Test
        void routesToVehicleEnrichedSearchForTopRankedTask() {
            when(articleRagService.searchAllArticles(any(), anyInt()))
                    .thenReturn(List.of(articleHit("motortrend-bmw-m3-2025-review",
                            "chunk:1", 1, 0.85, "M3 review")));
            when(fetchSpecsTool.execute(any())).thenReturn(List.of());
            when(llm.answer(any(), any())).thenReturn("Top ranked article summary");

            SubAgentResult result = agent.execute("top ranked vehicle articles with specs", Map.of());

            assertTrue(result.success());
            assertEquals("vehicle_enriched_search", result.metadata().get("operation"));
        }

        @Test
        void routesToVehicleEnrichedSearchForBestRatedTask() {
            when(articleRagService.searchAllArticles(any(), anyInt()))
                    .thenReturn(List.of(articleHit("art-1", "c:1", 1, 0.8, "Great car")));
            when(fetchSpecsTool.execute(any())).thenReturn(List.of());
            when(llm.answer(any(), any())).thenReturn("Answer");

            SubAgentResult result = agent.execute("best rated sports sedans", Map.of());

            assertEquals("vehicle_enriched_search", result.metadata().get("operation"));
        }

        @Test
        void routesToVehicleScopedSearchWhenVehicleQueryArgPresent() {
            var hits = Map.of("art-1", List.of(searchHit("art-1:1", "M3 review content", 0.8)));

            when(articleRagService.searchArticlesByVehicle(eq("BMW M3 2025"), anyInt()))
                    .thenReturn(hits);
            when(llm.answer(any(), any())).thenReturn("Articles about the M3");

            Map<String, Object> args = new LinkedHashMap<>();
            args.put("vehicleQuery", "BMW M3 2025");

            SubAgentResult result = agent.execute("find articles", args);

            assertEquals("vehicle_scoped_search", result.metadata().get("operation"));
        }

        @Test
        void routesToVehicleScopedSearchForArticlesAboutTask() {
            var hits = Map.of("art-1", List.of(searchHit("art-1:1", "M3 content", 0.7)));
            when(articleRagService.searchArticlesByVehicle(any(), anyInt())).thenReturn(hits);
            when(llm.answer(any(), any())).thenReturn("Summary");

            SubAgentResult result = agent.execute("articles about BMW M3", Map.of());

            assertEquals("vehicle_scoped_search", result.metadata().get("operation"));
        }

        @Test
        void routesToCrossArticleSearchByDefault() {
            when(articleRagService.searchAllArticles(any(), anyInt()))
                    .thenReturn(List.of(articleHit("art-1", "c:1", 1, 0.7, "excerpt")));
            when(llm.answer(any(), any())).thenReturn("Cross-article answer");

            SubAgentResult result = agent.execute("which car has best value?", Map.of());

            assertTrue(result.success());
            assertEquals("cross_article_search", result.metadata().get("operation"));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Single-article ask
    // ═══════════════════════════════════════════════════════════════════════

    @Nested
    class SingleArticleAsk {

        @Test
        void returnsLlmAnswerFromRagAnswerService() {
            var ragResult = new RagAnswerService.RagResult(
                    "The M3 has excellent track performance with the S58 engine.",
                    List.of(), List.of("art:1", "art:10"),
                    List.of("art:1"), 2, 0.92);

            when(articleRagService.askArticle("art-1", "performance?", 5, RetrievalMode.HYBRID))
                    .thenReturn(ragResult);

            Map<String, Object> args = Map.of("articleId", "art-1", "question", "performance?");
            SubAgentResult result = agent.execute("ask", args);

            assertTrue(result.success());
            assertEquals("article", result.agentName());
            assertFalse(result.summary().isEmpty());
            assertEquals(List.of("art:1"), result.citations());
            assertEquals(0.92, result.confidence());
            assertEquals("art-1", result.metadata().get("articleId"));
        }

        @Test
        void fallsThroughToCrossArticleSearchWhenArticleIdIsBlank() {
            // Blank articleId doesn't pass hasArg check, so routing falls through
            // to cross-article search (which fails with no results)
            when(articleRagService.searchAllArticles(any(), anyInt())).thenReturn(List.of());

            Map<String, Object> args = Map.of("articleId", "  ");
            SubAgentResult result = agent.execute("ask something", args);

            assertFalse(result.success());
            assertTrue(result.summary().contains("No articles found"));
        }

        @Test
        void failsGracefullyWhenRagServiceThrows() {
            when(articleRagService.askArticle(any(), any(), anyInt(), any()))
                    .thenThrow(new RuntimeException("DB connection lost"));

            Map<String, Object> args = Map.of("articleId", "art-1", "question", "q");
            SubAgentResult result = agent.execute("ask", args);

            assertFalse(result.success());
            assertTrue(result.summary().contains("DB connection lost"));
        }

        @Test
        void respectsModeArgument() {
            var ragResult = new RagAnswerService.RagResult(
                    "answer", List.of(), List.of(), List.of(), 1, 0.8);
            when(articleRagService.askArticle(any(), any(), anyInt(), eq(RetrievalMode.VECTOR)))
                    .thenReturn(ragResult);

            Map<String, Object> args = Map.of(
                    "articleId", "art-1", "question", "q", "mode", "VECTOR");
            agent.execute("ask", args);

            verify(articleRagService).askArticle("art-1", "q", 5, RetrievalMode.VECTOR);
        }

        @Test
        void defaultsToHybridWhenModeInvalid() {
            var ragResult = new RagAnswerService.RagResult(
                    "answer", List.of(), List.of(), List.of(), 1, 0.8);
            when(articleRagService.askArticle(any(), any(), anyInt(), eq(RetrievalMode.HYBRID)))
                    .thenReturn(ragResult);

            Map<String, Object> args = Map.of(
                    "articleId", "art-1", "question", "q", "mode", "NONSENSE");
            agent.execute("ask", args);

            verify(articleRagService).askArticle("art-1", "q", 5, RetrievalMode.HYBRID);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Cross-article search
    // ═══════════════════════════════════════════════════════════════════════

    @Nested
    class CrossArticleSearch {

        @Test
        void synthesizesAnswerFromRetrievedChunksViaLlm() {
            var hits = List.of(
                    articleHit("art-m3", "art-m3:2", 1, 0.85, "M3 rated 9.2 overall"),
                    articleHit("art-911", "art-911:2", 2, 0.78, "911 rated 9.0 overall"));

            when(articleRagService.searchAllArticles("best value?", 5)).thenReturn(hits);
            when(llm.answer(eq("best value?"), any())).thenReturn("The M3 is rated slightly higher.");

            SubAgentResult result = agent.execute("best value?", Map.of());

            assertTrue(result.success());
            assertEquals("The M3 is rated slightly higher.", result.summary());
            assertEquals(List.of("art-m3:2", "art-911:2"), result.citations());
            assertEquals(0.85, result.confidence());
            assertEquals(2, result.metadata().get("resultCount"));
            assertEquals("art-m3", result.metadata().get("topArticle"));

            // Verify LLM was called with context built from hits
            verify(llm).answer(eq("best value?"), argThat(chunks ->
                    chunks.size() == 2
                            && chunks.get(0).contains("[art-m3:2]")
                            && chunks.get(1).contains("[art-911:2]")));
        }

        @Test
        void failsWhenNoArticlesFound() {
            when(articleRagService.searchAllArticles(any(), anyInt())).thenReturn(List.of());

            SubAgentResult result = agent.execute("nonexistent topic", Map.of());

            assertFalse(result.success());
            assertTrue(result.summary().contains("No articles found"));
            verifyNoInteractions(llm);
        }

        @Test
        void usesQuestionArgOverTaskWhenProvided() {
            var hits = List.of(articleHit("art-1", "c:1", 1, 0.7, "excerpt"));
            when(articleRagService.searchAllArticles("specific question", 5)).thenReturn(hits);
            when(llm.answer(any(), any())).thenReturn("answer");

            Map<String, Object> args = Map.of("question", "specific question");
            agent.execute("generic task description", args);

            verify(articleRagService).searchAllArticles("specific question", 5);
        }

        @Test
        void respectsTopKArg() {
            var hits = List.of(articleHit("art-1", "c:1", 1, 0.7, "excerpt"));
            when(articleRagService.searchAllArticles("q", 10)).thenReturn(hits);
            when(llm.answer(any(), any())).thenReturn("answer");

            Map<String, Object> args = Map.of("question", "q", "topK", 10);
            agent.execute("search", args);

            verify(articleRagService).searchAllArticles("q", 10);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Vehicle-scoped search
    // ═══════════════════════════════════════════════════════════════════════

    @Nested
    class VehicleScopedSearch {

        @Test
        void synthesizesAnswerFromVehicleArticleChunks() {
            var hits = Map.of(
                    "art-m3-review", List.of(
                            searchHit("art-m3-review:1", "M3 verdict: excellent", 0.9),
                            searchHit("art-m3-review:2", "M3 ratings: 9.2", 0.85)));

            when(articleRagService.searchArticlesByVehicle("BMW M3 2025", 5)).thenReturn(hits);
            when(llm.answer(any(), any())).thenReturn("The BMW M3 has been well-reviewed.");

            Map<String, Object> args = Map.of("vehicleQuery", "BMW M3 2025");
            SubAgentResult result = agent.execute("find reviews", args);

            assertTrue(result.success());
            assertEquals("The BMW M3 has been well-reviewed.", result.summary());
            assertEquals("BMW M3 2025", result.metadata().get("vehicleQuery"));
            assertEquals(1, result.metadata().get("articleCount"));

            // LLM called with article chunks as context
            verify(llm).answer(contains("BMW M3 2025"), argThat(chunks ->
                    chunks.size() == 2 && chunks.get(0).contains("M3 verdict")));
        }

        @Test
        void failsWhenNoArticlesFoundForVehicle() {
            when(articleRagService.searchArticlesByVehicle(any(), anyInt()))
                    .thenReturn(Map.of());

            Map<String, Object> args = Map.of("vehicleQuery", "Nonexistent Car");
            SubAgentResult result = agent.execute("find reviews", args);

            assertFalse(result.success());
            assertTrue(result.summary().contains("No articles found"));
            verifyNoInteractions(llm);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Vehicle-enriched search (article → vehicle inter-agent communication)
    // ═══════════════════════════════════════════════════════════════════════

    @Nested
    class VehicleEnrichedSearch {

        @Test
        void mergesArticleAndSpecChunksForLlmSynthesis() {
            // Phase 1: article search returns hits
            var articleHits = List.of(
                    articleHit("motortrend-bmw-m3-2025-review", "mt-m3:1", 1, 0.9,
                            "2025 BMW M3 Competition — MotorTrend's top pick"),
                    articleHit("motortrend-porsche-911-2025-review", "mt-911:1", 2, 0.85,
                            "2025 Porsche 911 Carrera — still the benchmark"));

            when(articleRagService.searchAllArticles(any(), anyInt())).thenReturn(articleHits);

            // Phase 2: spec fetch — use answer() to route by vehicleId without NPE
            when(fetchSpecsTool.execute(any())).thenAnswer(invocation -> {
                FetchVehicleSpecsTool.Input input = invocation.getArgument(0);
                if (input.vehicleId().contains("bmw"))
                    return List.of(specChunk("bmw-m3:specs", "S58 engine, 503hp twin-turbo I6"));
                if (input.vehicleId().contains("porsche"))
                    return List.of(specChunk("911:specs", "3.0L flat-six, 379hp"));
                return List.of();
            });

            // Phase 3: LLM synthesis
            when(llm.answer(any(), any())).thenReturn("The M3 leads with 503hp vs the 911's 379hp.");

            SubAgentResult result = agent.execute("top ranked vehicle articles with specs", Map.of());

            assertTrue(result.success());
            assertEquals("The M3 leads with 503hp vs the 911's 379hp.", result.summary());

            // Citations include both article and spec chunk IDs
            assertTrue(result.citations().contains("mt-m3:1"));
            assertTrue(result.citations().contains("mt-911:1"));
            assertTrue(result.citations().contains("bmw-m3:specs"));
            assertTrue(result.citations().contains("911:specs"));
        }

        @Test
        void succeedsEvenWhenSpecFetchFailsForSomeVehicles() {
            var articleHits = List.of(
                    articleHit("motortrend-bmw-m3-2025-review", "mt-m3:1", 1, 0.9, "M3 review"),
                    articleHit("motortrend-porsche-911-2025-review", "mt-911:1", 2, 0.85, "911 review"));

            when(articleRagService.searchAllArticles(any(), anyInt())).thenReturn(articleHits);

            // BMW specs succeed, Porsche specs fail
            when(fetchSpecsTool.execute(any())).thenAnswer(invocation -> {
                FetchVehicleSpecsTool.Input input = invocation.getArgument(0);
                if (input.vehicleId().contains("bmw"))
                    return List.of(specChunk("bmw:specs", "S58 engine"));
                if (input.vehicleId().contains("porsche"))
                    throw new RuntimeException("Connection timeout");
                return List.of();
            });

            when(llm.answer(any(), any())).thenReturn("Partial answer with BMW specs only.");

            SubAgentResult result = agent.execute("top ranked with specs", Map.of());

            assertTrue(result.success());
            assertEquals(1, result.metadata().get("specs_resolved"));
        }

        @Test
        void succeedsWithNoSpecsAtAllIfFetchReturnsEmpty() {
            var articleHits = List.of(
                    articleHit("motortrend-bmw-m3-2025-review", "mt-m3:1", 1, 0.9, "M3 review"));
            when(articleRagService.searchAllArticles(any(), anyInt())).thenReturn(articleHits);

            // Spec fetch returns empty (vehicle not ingested in vehicle domain)
            when(fetchSpecsTool.execute(any())).thenReturn(List.of());
            when(llm.answer(any(), any())).thenReturn("Article-only answer.");

            SubAgentResult result = agent.execute("top ranked with specs", Map.of());

            assertTrue(result.success());
            assertEquals(0, result.metadata().get("specs_resolved"));
            // LLM gets article chunks only
            verify(llm).answer(any(), argThat(chunks -> chunks.size() == 1));
        }

        @Test
        void failsWhenNoArticlesFound() {
            when(articleRagService.searchAllArticles(any(), anyInt())).thenReturn(List.of());

            SubAgentResult result = agent.execute("top ranked articles", Map.of());

            assertFalse(result.success());
            assertTrue(result.summary().contains("No articles found"));
            verifyNoInteractions(fetchSpecsTool);
            verifyNoInteractions(llm);
        }

        @Test
        void usesExplicitVehicleIdsFromArgsInsteadOfExtractingFromArticleIds() {
            var articleHits = List.of(
                    articleHit("some-article", "art:1", 1, 0.8, "article content"));
            when(articleRagService.searchAllArticles(any(), anyInt())).thenReturn(articleHits);
            when(fetchSpecsTool.execute(any()))
                    .thenReturn(List.of(specChunk("tesla:specs", "Model 3 specs")));
            when(llm.answer(any(), any())).thenReturn("answer");

            // Supervisor passes explicit vehicleIds
            Map<String, Object> args = Map.of("vehicleIds", "tesla-model3-2025,bmw-m3-2025");

            agent.execute("top ranked with specs", args);

            // Should use the explicit IDs, not try to extract from articleId
            verify(fetchSpecsTool).execute(argThat(input ->
                    input.vehicleId().equals("tesla-model3-2025")));
            verify(fetchSpecsTool).execute(argThat(input ->
                    input.vehicleId().equals("bmw-m3-2025")));
        }

        @Test
        void extractsVehicleIdFromMotortrendArticleIdConvention() {
            // articleId = "motortrend-bmw-m3-2025-review" → vehicleId = "bmw-m3-2025"
            var articleHits = List.of(
                    articleHit("motortrend-bmw-m3-2025-review", "mt:1", 1, 0.9, "content"));
            when(articleRagService.searchAllArticles(any(), anyInt())).thenReturn(articleHits);
            when(fetchSpecsTool.execute(any())).thenReturn(List.of());
            when(llm.answer(any(), any())).thenReturn("answer");

            agent.execute("top ranked with specs", Map.of());

            verify(fetchSpecsTool).execute(argThat(input ->
                    input.vehicleId().equals("bmw-m3-2025")));
        }

        @Test
        void deduplicatesVehicleIds() {
            // Two articles about the same vehicle should only trigger one spec fetch
            var articleHits = List.of(
                    articleHit("motortrend-bmw-m3-2025-review", "mt-r:1", 1, 0.9, "review"),
                    articleHit("motortrend-bmw-m3-2025-comparison", "mt-c:1", 2, 0.85, "comparison"));
            when(articleRagService.searchAllArticles(any(), anyInt())).thenReturn(articleHits);
            when(fetchSpecsTool.execute(any())).thenReturn(List.of());
            when(llm.answer(any(), any())).thenReturn("answer");

            agent.execute("top ranked with specs", Map.of());

            // "bmw-m3-2025" extracted from both articles, but fetch called only once
            verify(fetchSpecsTool, times(1)).execute(any());
        }

        @Test
        void limitsVehicleIdsToFive() {
            // 6 articles about different vehicles, but we cap at 5 spec fetches
            var articleHits = List.of(
                    articleHit("motortrend-car-a-2025-review", "a:1", 1, 0.9, "a"),
                    articleHit("motortrend-car-b-2025-review", "b:1", 2, 0.88, "b"),
                    articleHit("motortrend-car-c-2025-review", "c:1", 3, 0.86, "c"),
                    articleHit("motortrend-car-d-2025-review", "d:1", 4, 0.84, "d"),
                    articleHit("motortrend-car-e-2025-review", "e:1", 5, 0.82, "e"),
                    articleHit("motortrend-car-f-2025-review", "f:1", 6, 0.80, "f"));
            when(articleRagService.searchAllArticles(any(), anyInt())).thenReturn(articleHits);
            when(fetchSpecsTool.execute(any())).thenReturn(List.of());
            when(llm.answer(any(), any())).thenReturn("answer");

            agent.execute("top ranked with specs", Map.of());

            // Max 5 parallel fetches, not 6
            verify(fetchSpecsTool, times(5)).execute(any());
        }

        @Test
        void parallelFetchActuallyRunsConcurrently() throws Exception {
            // Simulate slow spec fetches to verify they don't run serially
            var articleHits = List.of(
                    articleHit("motortrend-car-a-2025-review", "a:1", 1, 0.9, "a"),
                    articleHit("motortrend-car-b-2025-review", "b:1", 2, 0.88, "b"),
                    articleHit("motortrend-car-c-2025-review", "c:1", 3, 0.86, "c"));
            when(articleRagService.searchAllArticles(any(), anyInt())).thenReturn(articleHits);

            // Each fetch sleeps 200ms. If serial: ~600ms. If parallel: ~200ms.
            when(fetchSpecsTool.execute(any())).thenAnswer(invocation -> {
                Thread.sleep(200);
                String vid = invocation.getArgument(0, FetchVehicleSpecsTool.Input.class).vehicleId();
                return List.of(specChunk(vid + ":specs", vid + " specs"));
            });
            when(llm.answer(any(), any())).thenReturn("answer");

            long start = System.currentTimeMillis();
            SubAgentResult result = agent.execute("top ranked with specs", Map.of());
            long elapsed = System.currentTimeMillis() - start;

            assertTrue(result.success());
            assertEquals(3, result.metadata().get("specs_resolved"));

            // If truly parallel, should complete well under 600ms.
            // Allow generous margin for CI/slow machines but still catch serial execution.
            assertTrue(elapsed < 500,
                    "Expected parallel execution (<500ms) but took " + elapsed + "ms");
        }

        @Test
        void truncatesLongSpecChunksTo500Chars() {
            var articleHits = List.of(
                    articleHit("motortrend-bmw-m3-2025-review", "mt:1", 1, 0.9, "review"));
            when(articleRagService.searchAllArticles(any(), anyInt())).thenReturn(articleHits);

            // Spec with 800 chars of text
            String longText = "x".repeat(800);
            when(fetchSpecsTool.execute(any()))
                    .thenReturn(List.of(specChunk("bmw:specs", longText)));
            when(llm.answer(any(), any())).thenReturn("answer");

            agent.execute("top ranked with specs", Map.of());

            // LLM context should have truncated spec text (spec prefix + max ~500 chars + "...")
            verify(llm).answer(any(), argThat(chunks -> {
                String spec = chunks.stream()
                        .filter(c -> c.contains("bmw:specs"))
                        .findFirst().orElse("");
                // Should be truncated — well under the original 800 chars
                return spec.length() < 600;
            }));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // LLM synthesis verification (applies to cross-article and enriched)
    // ═══════════════════════════════════════════════════════════════════════

    @Nested
    class LlmSynthesis {

        @Test
        void crossArticleSearchPassesChunkIdsInContext() {
            var hits = List.of(
                    articleHit("art-1", "art-1:1", 1, 0.9, "first article excerpt"),
                    articleHit("art-2", "art-2:3", 2, 0.8, "second article excerpt"));
            when(articleRagService.searchAllArticles(any(), anyInt())).thenReturn(hits);
            when(llm.answer(any(), any())).thenReturn("synthesized");

            agent.execute("which is best?", Map.of());

            verify(llm).answer(eq("which is best?"), argThat(chunks ->
                    chunks.get(0).startsWith("[art-1:1]")
                            && chunks.get(1).startsWith("[art-2:3]")));
        }

        @Test
        void vehicleEnrichedSearchAppendsSpecInstructionToQuestion() {
            var hits = List.of(articleHit("motortrend-bmw-m3-2025-review", "mt:1", 1, 0.9, "review"));
            when(articleRagService.searchAllArticles(any(), anyInt())).thenReturn(hits);
            when(fetchSpecsTool.execute(any())).thenReturn(List.of());
            when(llm.answer(any(), any())).thenReturn("answer");

            agent.execute("top ranked cars with specs", Map.of());

            // The enriched question should contain the original question and citation rules
            verify(llm).answer(argThat(q ->
                    q.contains("top ranked cars with specs")
                            && q.contains("INSTRUCTIONS")), any());
        }

        @Test
        void failsGracefullyWhenLlmThrows() {
            var hits = List.of(articleHit("art-1", "c:1", 1, 0.8, "excerpt"));
            when(articleRagService.searchAllArticles(any(), anyInt())).thenReturn(hits);
            when(llm.answer(any(), any())).thenThrow(new RuntimeException("Ollama is down"));

            SubAgentResult result = agent.execute("search something", Map.of());

            assertFalse(result.success());
            assertTrue(result.summary().contains("Ollama is down"));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Edge cases
    // ═══════════════════════════════════════════════════════════════════════

    @Nested
    class EdgeCases {

        @Test
        void handlesNullTask() {
            when(articleRagService.searchAllArticles(any(), anyInt()))
                    .thenReturn(List.of(articleHit("a", "c:1", 1, 0.7, "ex")));
            when(llm.answer(any(), any())).thenReturn("answer");

            // null task should fall through to cross-article search
            SubAgentResult result = agent.execute(null, Map.of());

            assertTrue(result.success());
            assertEquals("cross_article_search", result.metadata().get("operation"));
        }

        @Test
        void handlesNullArgs() {
            when(articleRagService.searchAllArticles(any(), anyInt()))
                    .thenReturn(List.of(articleHit("a", "c:1", 1, 0.7, "ex")));
            when(llm.answer(any(), any())).thenReturn("answer");

            // null args should not throw NPE
            SubAgentResult result = agent.execute("search", null);

            assertTrue(result.success());
        }

        @Test
        void handlesNonMotortrendArticleIdsForVehicleExtraction() {
            // articleId doesn't follow motortrend convention — falls through to using
            // the full articleId as vehicleId (which will probably return empty specs)
            var hits = List.of(articleHit("custom-article-123", "ca:1", 1, 0.9, "content"));
            when(articleRagService.searchAllArticles(any(), anyInt())).thenReturn(hits);
            when(fetchSpecsTool.execute(any())).thenReturn(List.of());
            when(llm.answer(any(), any())).thenReturn("answer");

            agent.execute("top ranked with specs", Map.of());

            // Uses full articleId as vehicleId since no "motortrend-" prefix
            verify(fetchSpecsTool).execute(argThat(input ->
                    input.vehicleId().equals("custom-article-123")));
        }

        @Test
        void topKDefaultsToFiveWhenNotProvided() {
            var hits = List.of(articleHit("a", "c:1", 1, 0.7, "ex"));
            when(articleRagService.searchAllArticles("q", 5)).thenReturn(hits);
            when(llm.answer(any(), any())).thenReturn("answer");

            agent.execute("q", Map.of());

            verify(articleRagService).searchAllArticles("q", 5);
        }

        @Test
        void vehicleIdsListArgAcceptedAsListType() {
            var hits = List.of(articleHit("art", "c:1", 1, 0.9, "content"));
            when(articleRagService.searchAllArticles(any(), anyInt())).thenReturn(hits);
            when(fetchSpecsTool.execute(any())).thenReturn(List.of());
            when(llm.answer(any(), any())).thenReturn("answer");

            // vehicleIds as a List (Jackson would parse JSON arrays this way)
            Map<String, Object> args = Map.of("vehicleIds", List.of("tesla-3", "bmw-m3"));
            agent.execute("top ranked with specs", args);

            verify(fetchSpecsTool).execute(argThat(i -> i.vehicleId().equals("tesla-3")));
            verify(fetchSpecsTool).execute(argThat(i -> i.vehicleId().equals("bmw-m3")));
        }
    }
}
