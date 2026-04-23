package com.example.airagassistant.agentic.multi;

import com.example.airagassistant.LlmClient;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The supervisor's planner. Much simpler than PlannerService because it
 * only decides WHICH AGENT to call, not which individual steps to run.
 *
 * Compare the prompt sizes:
 *   - PlannerService.plan():       ~80 lines, 13 step types, complex rules
 *   - SupervisorPlanner.plan():    ~40 lines, 4 agent types, simple rules
 *
 * This is the core benefit of multi-agent: each planner has a focused,
 * short prompt that the LLM can follow reliably.
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class SupervisorPlanner {

    private static final Set<String> VALID_AGENTS = Set.of(
            "research", "vehicle", "communication", "article"
    );

    private final LlmClient llm;

    private final ObjectMapper objectMapper = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    /**
     * Decompose a user request into agent delegations.
     *
     * The LLM sees only 4 agent types — not 13 individual steps.
     * Each sub-agent's internal planner handles the step-level decomposition.
     */
    public List<Delegation> plan(String userRequest, String docId) {

        String prompt = """
                You are a supervisor that delegates tasks to specialized agents.
                
                Available agents:
                - research: retrieves information from ingested documents (PDFs, Q&A)
                - vehicle: handles vehicle specs, performance, summaries, comparisons, enrichment.
                  Use vehicle when the user asks about a car's specs, engine, horsepower, 0-60, price,
                  features, safety ratings, or comparisons between vehicles.
                - article: handles MotorTrend articles, editorial reviews, and article ratings.
                  Use article ONLY when the user explicitly mentions "article", "review", "MotorTrend",
                  "rated", "Article ID", or asks about editorial opinions/ratings from publications.
                  The article agent can fetch vehicle specs internally if needed.
                - communication: drafts and sends emails or SMS messages
                
                CRITICAL ROUTING RULES (follow in this exact order):
                1. Does the user mention "article", "review", "MotorTrend", "rated", "Article ID"?
                   → YES: delegate to article
                   → NO: continue to rule 2
                2. Does the user ask about vehicle specs, performance, engine, horsepower, comparison, price?
                   → YES: delegate to vehicle
                   → NO: continue to rule 3
                3. Does the user ask about documents, PDFs, or general knowledge?
                   → YES: delegate to research
                   → NO: default to research
                4. Does the user ask to email/text/send something?
                   → Add communication AFTER the content agent. NEVER add it unless explicitly requested.
                
                JSON key naming — MUST use camelCase:
                  articleId (NOT article_id), vehicleId (NOT vehicle_id)
                
                Return ONLY valid JSON. Response MUST start with { and end with }.
                
                Examples:
                  User: "Tell me about the BMW M3 specs and performance"
                  → {"delegations": [{"agent": "vehicle", "task": "get M3 specs and performance", "args": {"vehicleId": "bmw-m3-2025-competition"}}]}
                
                  User: "What is the horsepower of the Tesla Model 3?"
                  → {"delegations": [{"agent": "vehicle", "task": "get horsepower", "args": {"vehicleId": "tesla-model3-2025-long-range"}}]}
                
                  User: "Compare BMW M3 vs Porsche 911"
                  → {"delegations": [{"agent": "vehicle", "task": "compare vehicles", "args": {"vehicleIds": "bmw-m3-2025-competition,porsche-911-2025"}}]}
                
                  User: "What did MotorTrend say about the M3? Article ID is motortrend-bmw-m3-2025-review"
                  → {"delegations": [{"agent": "article", "task": "ask about M3", "args": {"articleId": "motortrend-bmw-m3-2025-review", "question": "what did MotorTrend say"}}]}
                
                  User: "Show me top rated articles with specs"
                  → {"delegations": [{"agent": "article", "task": "top rated articles with specs", "args": {}}]}
                
                  User: "Which sports sedan was rated best?"
                  → {"delegations": [{"agent": "article", "task": "best rated sports sedan", "args": {"question": "which sports sedan was rated best"}}]}
                
                  User: "What is Spring Boot auto-configuration?"
                  → {"delegations": [{"agent": "research", "task": "explain auto-configuration", "args": {}}]}
                
                Do NOT return markdown. Do NOT return explanations.
                Response MUST start with { and end with }.
                
                User request:
                %s
                """.formatted(safe(userRequest));

        String response = llm.answer(prompt, List.of());
        log.info("Supervisor planner response: {}", response);

        return parseAndValidate(response, docId);
    }

    // ─── Parsing ──────────────────────────────────────────────────────────

    private List<Delegation> parseAndValidate(String response, String docId) {
        if (response == null || response.isBlank()) {
            log.warn("Supervisor planner returned empty response. Defaulting to research.");
            return List.of(new Delegation("research", "answer the user's question", Map.of()));
        }

        try {
            String json = extractJson(response);
            SupervisorPayload payload = objectMapper.readValue(json, SupervisorPayload.class);

            if (payload == null || payload.delegations() == null || payload.delegations().isEmpty()) {
                log.warn("Supervisor planner returned no delegations. Defaulting to research.");
                return List.of(new Delegation("research", "answer the user's question", Map.of()));
            }

            List<Delegation> validated = new ArrayList<>();
            for (RawDelegation raw : payload.delegations()) {
                if (raw == null || raw.agent() == null || raw.agent().isBlank()) continue;

                String agent = raw.agent().trim().toLowerCase();
                if (!VALID_AGENTS.contains(agent)) {
                    log.warn("Supervisor planner returned unknown agent '{}' — skipping.", raw.agent());
                    continue;
                }

                String task = raw.task() != null ? raw.task().trim() : "";
                Map<String, Object> args = raw.args() != null ? new LinkedHashMap<>(raw.args()) : new LinkedHashMap<>();

                validated.add(new Delegation(agent, task, args));
            }

            if (validated.isEmpty()) {
                log.warn("No valid delegations after filtering. Defaulting to research.");
                return List.of(new Delegation("research", "answer the user's question", Map.of()));
            }

            return validated;

        } catch (Exception e) {
            log.warn("Failed to parse supervisor planner response: {}. Defaulting to research.", e.getMessage());
            return List.of(new Delegation("research", "answer the user's question", Map.of()));
        }
    }

    private String extractJson(String text) {
        if (text == null || text.isBlank()) return "{}";
        String trimmed = text.trim();

        // Strip code fences
        if (trimmed.startsWith("```")) {
            trimmed = trimmed
                    .replaceFirst("^```json\\\\s*", "")
                    .replaceFirst("^```\\\\s*", "")
                    .replaceFirst("\\\\s*```$", "")
                    .trim();
        }

        int first = trimmed.indexOf('{');
        int last = trimmed.lastIndexOf('}');
        if (first >= 0 && last > first) {
            return trimmed.substring(first, last + 1);
        }
        return trimmed;
    }

    private String safe(String value) {
        return value == null ? "" : value;
    }

    // ─── Internal DTOs for JSON parsing ───────────────────────────────────

    private record SupervisorPayload(List<RawDelegation> delegations) {}
    private record RawDelegation(String agent, String task, Map<String, Object> args) {}
}
