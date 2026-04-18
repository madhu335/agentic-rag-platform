package com.example.airagassistant.agentic.multi;

import com.example.airagassistant.agentic.dto.AgentResponse;
import com.example.airagassistant.agentic.multi.agents.ArticleSubAgent;
import com.example.airagassistant.agentic.multi.agents.CommunicationSubAgent;
import com.example.airagassistant.agentic.multi.agents.ResearchSubAgent;
import com.example.airagassistant.agentic.multi.agents.VehicleSubAgent;
import com.example.airagassistant.agentic.state.AgentSessionState;
import com.example.airagassistant.agentic.state.AgentStateStore;
import com.example.airagassistant.agentic.state.StepHistoryEntry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Multi-agent supervisor.
 *
 * This is the top-level coordinator that replaces AgentExecutorService for
 * the multi-agent flow. It:
 *
 *   1. Asks SupervisorPlanner to decompose the request into agent delegations
 *   2. Dispatches each delegation to the appropriate sub-agent
 *   3. Passes outputs between agents (research result → communication content)
 *   4. Persists session state for observability
 *   5. Synthesizes the final response
 *
 * Comparison with single-agent AgentExecutorService:
 *
 *   Single-agent:
 *     PlannerService generates [research, email, stop] → executor runs each step
 *     One planner prompt knows all 13 step types
 *     One flat loop, one state object
 *
 *   Multi-agent (this):
 *     SupervisorPlanner generates [research, communication, article, vehicle] → supervisor delegates
 *     Supervisor prompt knows 4 agent types (not 13 step types)
 *     Each sub-agent has its own internal logic and can evolve independently
 *
 * The supervisor pattern is the simplest multi-agent architecture:
 * one coordinator, N specialized workers, no peer-to-peer communication.
 * This keeps the system debuggable — you can trace every decision back to
 * "supervisor chose agent X because of delegation Y."
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class SupervisorAgent {

    private final SupervisorPlanner planner;
    private final ResearchSubAgent researchAgent;
    private final VehicleSubAgent vehicleAgent;
    private final CommunicationSubAgent communicationAgent;
    private final ArticleSubAgent articleAgent;
    private final AgentStateStore stateStore;

    public AgentResponse handle(String prompt, String docId, Integer topK) {
        String sessionId = UUID.randomUUID().toString();
        log.info("SupervisorAgent — sessionId={} prompt='{}'", sessionId, prompt);

        // Initialize session state (reuses existing AgentSessionState for
        // dashboard compatibility — multi-agent sessions show up in the
        // same dashboard as single-agent sessions)
        AgentSessionState state = AgentSessionState.empty(sessionId, prompt, docId);
        state = addHistory(state, "supervisor", Map.of(), "STARTED",
                "Multi-agent supervisor started");
        stateStore.save(state);

        // Step 1: Supervisor planner decomposes the request
        List<Delegation> delegations;
        try {
            delegations = planner.plan(prompt, docId);
            log.info("SupervisorAgent — {} delegations: {}",
                    delegations.size(),
                    delegations.stream().map(Delegation::agent).toList());

            state = addHistory(state, "supervisor_plan", Map.of(),
                    "SUCCESS", "Planned " + delegations.size() + " delegations: " +
                    delegations.stream().map(Delegation::agent).toList());
            stateStore.save(state);

        } catch (Exception e) {
            log.error("SupervisorAgent — planning failed: {}", e.getMessage());
            state = addHistory(state, "supervisor_plan", Map.of(),
                    "FAILED", "Planning failed: " + e.getMessage());
            stateStore.save(state);

            return new AgentResponse(sessionId, "Planning failed: " + e.getMessage(),
                    List.of(), 0.0, null, "FAILED");
        }

        // Step 2: Execute delegations in order, passing outputs forward
        SubAgentResult lastContentResult = null;  // tracks the last result that produced content
        SubAgentResult lastResult = null;

        for (Delegation delegation : delegations) {
            log.info("SupervisorAgent — dispatching to '{}': task='{}'",
                    delegation.agent(), delegation.task());

            state = addHistory(state, "delegate_" + delegation.agent(),
                    delegation.args(), "STARTED",
                    "Delegating to " + delegation.agent() + ": " + delegation.task());
            stateStore.save(state);

            SubAgentResult result;

            try {
                result = switch (delegation.agent()) {
                    case "research" -> {
                        SubAgentResult r = researchAgent.execute(
                                delegation.task(), docId, topK);
                        // Update research snapshot for dashboard visibility
                        if (r.success()) {
                            state = state.withResearch(new AgentSessionState.ResearchSnapshot(
                                    r.summary(), r.citations(), r.confidence(),
                                    r.judge(), List.of()));
                        }
                        yield r;
                    }

                    case "vehicle" -> {
                        SubAgentResult r = vehicleAgent.execute(
                                delegation.task(), delegation.args());
                        if (r.success()) {
                            state = state.withResearch(new AgentSessionState.ResearchSnapshot(
                                    r.summary(), r.citations(), r.confidence(),
                                    r.judge(), List.of()));
                        }
                        yield r;
                    }

                    case "article" -> {
                        SubAgentResult r = articleAgent.execute(
                                delegation.task(), delegation.args());
                        if (r.success()) {
                            // Update research snapshot (for backward compat with dashboard)
                            state = state.withResearch(new AgentSessionState.ResearchSnapshot(
                                    r.summary(), r.citations(), r.confidence(),
                                    r.judge(), List.of()));

                            // Update article snapshot with extraction visibility
                            Map<String, Object> meta = r.metadata() != null ? r.metadata() : Map.of();
                            @SuppressWarnings("unchecked")
                            List<String> articleIds = meta.containsKey("retrievedArticles")
                                    ? ((List<Map<String, Object>>) meta.get("retrievedArticles")).stream()
                                        .map(a -> String.valueOf(a.get("articleId")))
                                        .distinct()
                                        .toList()
                                    : List.of();
                            @SuppressWarnings("unchecked")
                            List<String> extractedVehicleIds = meta.containsKey("vehicleIds")
                                    ? (List<String>) meta.get("vehicleIds")
                                    : List.of();
                            @SuppressWarnings("unchecked")
                            List<String> resolvedVehicleIds = meta.containsKey("vehicleSpecs")
                                    ? new ArrayList<>(((Map<String, Object>) meta.get("vehicleSpecs")).keySet())
                                    : List.of();

                            state = state.withArticle(new AgentSessionState.ArticleSnapshot(
                                    articleIds,
                                    extractedVehicleIds,
                                    resolvedVehicleIds,
                                    String.valueOf(meta.getOrDefault("operation", "unknown")),
                                    r.judge() != null ? r.judge().score() : null,
                                    meta.containsKey("contextChunks")
                                            ? ((List<?>) meta.get("contextChunks")).size() : 0
                            ));
                        }
                        yield r;
                    }

                    case "communication" -> {
                        // Pass the content from the previous agent's output
                        String contentToSend = lastContentResult != null
                                ? lastContentResult.summary()
                                : prompt;  // fallback to user prompt if no prior content
                        SubAgentResult r = communicationAgent.execute(
                                delegation.task(), contentToSend, delegation.args());
                        // Update email/sms snapshot for dashboard visibility
                        if (r.success()) {
                            updateCommunicationState(state, r);
                        }
                        yield r;
                    }

                    default -> SubAgentResult.failure("unknown",
                            "Unknown agent: " + delegation.agent());
                };
            } catch (Exception e) {
                log.error("SupervisorAgent — agent '{}' threw exception: {}",
                        delegation.agent(), e.getMessage());
                result = SubAgentResult.failure(delegation.agent(), e.getMessage());
            }

            // Track results
            lastResult = result;
            if (result.success() && result.summary() != null
                    && !result.summary().isBlank()
                    && !"communication".equals(delegation.agent())) {
                // Communication results are status messages ("Email drafted to ..."),
                // not content. Don't use them as input to subsequent agents.
                lastContentResult = result;
            }

            String status = result.success() ? "SUCCESS" : "FAILED";

            // Merge delegation args with sub-agent metadata for dashboard visibility.
            // delegation.args() has the input (question, articleId, etc.)
            // result.metadata() has the output (contextChunks, judgeScore, vehicleSpecs, etc.)
            // Both show up in the session history under "args" so the dashboard
            // can display what went in AND what came out.
            Map<String, Object> historyArgs = new LinkedHashMap<>(delegation.args());
            if (result.metadata() != null && !result.metadata().isEmpty()) {
                historyArgs.put("_result", result.metadata());
            }
            if (result.judge() != null) {
                historyArgs.put("_judge", Map.of(
                        "grounded", result.judge().grounded(),
                        "correct", result.judge().correct(),
                        "complete", result.judge().complete(),
                        "score", result.judge().score(),
                        "reason", result.judge().reason() != null ? result.judge().reason() : ""
                ));
            }

            state = addHistory(state, "delegate_" + delegation.agent(),
                    historyArgs, status, result.summary());
            stateStore.save(state);

            if (!result.success()) {
                log.warn("SupervisorAgent — agent '{}' failed: {}. Continuing with remaining delegations.",
                        delegation.agent(), result.summary());
                // Don't abort — try remaining delegations. A failed email
                // shouldn't prevent the research result from being returned.
            }
        }

        // Step 3: Synthesize final response
        state = addHistory(state, "supervisor", Map.of(), "COMPLETED",
                "Multi-agent flow completed");
        stateStore.save(state);

        return buildResponse(sessionId, lastContentResult, lastResult, state);
    }

    // ─── Response building ────────────────────────────────────────────────

    private AgentResponse buildResponse(String sessionId,
                                         SubAgentResult contentResult,
                                         SubAgentResult lastResult,
                                         AgentSessionState state) {
        // Prefer the content-producing result (research/vehicle) over
        // the status-producing result (communication)
        SubAgentResult primary = contentResult != null ? contentResult : lastResult;

        if (primary == null) {
            return new AgentResponse(sessionId, "No agents produced a result.",
                    List.of(), 0.0, null, "FAILED");
        }

        String emailStatus = "NOT_STARTED";
        if (state.email() != null) {
            emailStatus = state.email().status().name();
        } else if (state.sms() != null) {
            emailStatus = state.sms().status().name();
        }

        return new AgentResponse(
                sessionId,
                primary.summary(),
                primary.citations() != null ? primary.citations() : List.of(),
                primary.confidence() != null ? primary.confidence() : 0.0,
                primary.judge(),
                emailStatus
        );
    }

    private void updateCommunicationState(AgentSessionState state,
                                           SubAgentResult result) {
        Map<String, Object> meta = result.metadata();
        String type = meta.getOrDefault("type", "email").toString();

        if ("sms".equals(type)) {
            state.withSms(new AgentSessionState.SmsSnapshot(
                    meta.getOrDefault("phoneNumber", "+10000000000").toString(),
                    meta.getOrDefault("message", "").toString(),
                    "SENT".equals(meta.get("status"))
                            ? AgentSessionState.SmsStatus.SENT
                            : AgentSessionState.SmsStatus.COMPOSED
            ));
        } else {
            state.withEmail(new AgentSessionState.EmailSnapshot(
                    meta.getOrDefault("recipient", "hr@company.com").toString(),
                    meta.getOrDefault("subject", "Requested Summary").toString(),
                    meta.getOrDefault("bodyPreview", "").toString(),
                    "SENT".equals(meta.get("status"))
                            ? AgentSessionState.EmailStatus.SENT
                            : AgentSessionState.EmailStatus.DRAFTED
            ));
        }
    }

    // ─── History helper ───────────────────────────────────────────────────

    private AgentSessionState addHistory(AgentSessionState state,
                                          String step, Map<String, Object> args,
                                          String status, String summary) {
        List<StepHistoryEntry> history = new ArrayList<>();
        if (state.history() != null) {
            history.addAll(state.history());
        }
        history.add(new StepHistoryEntry(step, args, status, summary, null, Instant.now()));
        return state.withHistory(history);
    }
}
