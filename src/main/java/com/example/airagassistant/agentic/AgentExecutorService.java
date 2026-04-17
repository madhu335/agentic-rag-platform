package com.example.airagassistant.agentic;

import com.example.airagassistant.agentic.divergence.DivergenceCheckService;
import com.example.airagassistant.agentic.dto.*;
import com.example.airagassistant.agentic.state.AgentSessionState;
import com.example.airagassistant.agentic.state.AgentStateStore;
import com.example.airagassistant.agentic.state.StepHistoryEntry;
import com.example.airagassistant.agentic.tools.ResearchTool;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.*;
import java.util.stream.Collectors;

@Service
@Slf4j
public class AgentExecutorService {

    private static final int MAX_RETRIES = 2;
    private static final int MAX_STEPS = 5;
    private static final int DEFAULT_TOP_K = 5;
    private static final String DEFAULT_SUBJECT = "Requested Summary";
    private static final String DEFAULT_RECIPIENT = "hr@company.com";

    private final PlannerService plannerService;
    private final ResearchTool researchTool;
    private final AgentStateStore stateStore;
    private final DivergenceCheckService divergenceCheckService;
    private final Map<String, AgentTool> tools;

    public AgentExecutorService(
            PlannerService plannerService,
            ResearchTool researchTool,
            AgentStateStore stateStore,
            DivergenceCheckService divergenceCheckService,
            List<AgentTool> toolList
    ) {
        this.plannerService = plannerService;
        this.researchTool = researchTool;
        this.stateStore = stateStore;
        this.divergenceCheckService = divergenceCheckService;
        this.tools = toolList.stream()
                .collect(Collectors.toMap(AgentTool::name, t -> t));
    }

    public AgentResponse execute(AgentRequest request) {

        String sessionId = UUID.randomUUID().toString();

        AgentSessionState state = initState(sessionId, request);
        AgentPlan plan = plannerService.plan(request.prompt());

        int stepCount = 0;

        while (stepCount < MAX_STEPS) {

            if (plan == null || plan.steps() == null || plan.steps().isEmpty()) {
                log.warn("Planner returned no steps. Stopping. sessionId={}", sessionId);
                return buildResponse(state);
            }

            boolean planAdvanced = false;
            boolean restartPlanning = false;

            for (PlanStep step : plan.steps()) {

                if (step == null || step.step() == null || step.step().isBlank()) {
                    continue;
                }

                switch (step.step().toLowerCase()) {

                    case "research" -> {
                        try {
                            state = addHistoryEntry(state, "research", step.args(), "STARTED", null, null);
                            stateStore.save(state);

                            String query = getStringArg(step.args(), "query", request.prompt());

                            ResearchResult result = researchTool.research(
                                    request.docId(),
                                    query,
                                    request.topK() != null && request.topK() > 0 ? request.topK() : DEFAULT_TOP_K
                            );

                            log.info(
                                    "RESEARCH RESULT → retrievalScore={}, judgeScore={}, grounded={}, correct={}, complete={}",
                                    result.retrievalScore(),
                                    result.judge() != null ? result.judge().score() : null,
                                    result.judge() != null ? result.judge().grounded() : null,
                                    result.judge() != null ? result.judge().correct() : null,
                                    result.judge() != null ? result.judge().complete() : null
                            );

                            state = updateResearch(state, result);

                            state = addHistoryEntry(
                                    state,
                                    "research",
                                    step.args(),
                                    "SUCCESS",
                                    result.answer(),
                                    null
                            );
                            stateStore.save(state);

                            boolean diverged = divergenceCheckService.isDiverged(result);
                            boolean lowQuality = shouldRetryResearch(result);

                            if ((diverged || lowQuality) && canRetry(state)) {
                                String refinedQuery = refineQuery(query);

                                state = addHistoryEntry(
                                        state,
                                        "research",
                                        Map.of("query", refinedQuery),
                                        "RETRYING",
                                        "Research result was weak; retrying with refined query",
                                        null
                                );
                                stateStore.save(state);

                                plan = singleResearchPlan(refinedQuery);
                                planAdvanced = true;
                                restartPlanning = true;
                                break;
                            }

                            if ((diverged || lowQuality) && !canRetry(state)) {
                                state = addHistoryEntry(
                                        state,
                                        "research",
                                        step.args(),
                                        "MAX_RETRIES_REACHED",
                                        "Research remained weak after retries",
                                        null
                                );
                                stateStore.save(state);

                                return buildResponse(state);
                            }

                            AgentObservation observation = new AgentObservation(
                                    "research",
                                    result.answer(),
                                    result.retrievalScore(),
                                    result.judge()
                            );

                            AgentPlan replanned = plannerService.replan(state, observation);

                            if (replanned == null || replanned.steps() == null || replanned.steps().isEmpty()) {
                                plan = stopPlan();
                            } else {
                                plan = replanned;
                            }

                        } catch (Exception e) {
                            state = addHistoryEntry(
                                    state,
                                    "research",
                                    step.args(),
                                    "FAILED",
                                    null,
                                    e.getMessage()
                            );
                            stateStore.save(state);

                            if (canRetry(state)) {
                                String query = getStringArg(step.args(), "query", request.prompt());
                                String refinedQuery = refineQuery(query);

                                state = addHistoryEntry(
                                        state,
                                        "research",
                                        Map.of("query", refinedQuery),
                                        "RETRYING",
                                        "Research tool failed; retrying with refined query",
                                        e.getMessage()
                                );
                                stateStore.save(state);

                                plan = singleResearchPlan(refinedQuery);
                                planAdvanced = true;
                                restartPlanning = true;
                                break;
                            }

                            return buildResponse(state);
                        }

                        planAdvanced = true;
                        break;
                    }

                    case "stop" -> {
                        return buildResponse(state);
                    }

                    default -> {
                        try {
                            state = executeToolStep(state, step);

                            if ("email".equalsIgnoreCase(step.step())
                                    || "compose_sms".equalsIgnoreCase(step.step())
                                    || "send".equalsIgnoreCase(step.step())
                                    || "generate_vehicle_summary".equalsIgnoreCase(step.step())
                                    || "compare_vehicles".equalsIgnoreCase(step.step())
                                    || "enrich_vehicle_data".equalsIgnoreCase(step.step())) {
                                return buildResponse(state);
                            }

                        } catch (Exception e) {
                            state = addHistoryEntry(
                                    state,
                                    step.step(),
                                    step.args(),
                                    "FAILED",
                                    null,
                                    e.getMessage()
                            );
                            stateStore.save(state);
                            throw e;
                        }

                        planAdvanced = true;
                        break;
                    }
                }

                if (restartPlanning) {
                    break;
                }
            }

            if (!planAdvanced) {
                log.warn("Plan did not advance. Stopping. sessionId={}", state.sessionId());
                return buildResponse(state);
            }

            stepCount++;
        }

        log.warn("Max agent steps reached. Returning best available response. sessionId={}", state.sessionId());
        return buildResponse(state);
    }

    public AgentResponse continueExecution(ContinueRequest request) {
        AgentSessionState state = stateStore.get(request.sessionId());

        if (state == null) {
            throw new IllegalArgumentException("Session not found: " + request.sessionId());
        }

        if ((state.email() != null && state.email().status() == AgentSessionState.EmailStatus.SENT)
                || (state.sms() != null && state.sms().status() == AgentSessionState.SmsStatus.SENT)) {

            state = addHistoryEntry(
                    state,
                    "continue",
                    Map.of("instruction", request.instruction()),
                    "SKIPPED",
                    "Session already completed",
                    null
            );
            stateStore.save(state);

            return buildResponse(state);
        }

        AgentPlan plan = plannerService.planContinuation(state, request.instruction());

        for (PlanStep step : plan.steps()) {
            state = executeToolStep(state, step);
        }

        return buildResponse(state);
    }

    private boolean isUnresolvedSms(AgentSessionState state) {
        return state.sms() == null
                || state.sms().message() == null
                || state.sms().message().isBlank();
    }

    public AgentSessionState getSessionState(String sessionId) {
        AgentSessionState state = stateStore.get(sessionId);

        if (state == null) {
            throw new IllegalArgumentException("Session not found: " + sessionId);
        }

        return state;
    }

    private AgentSessionState executeToolStep(AgentSessionState state, PlanStep step) {
        String stepName = step.step().toLowerCase();

        if (isEmailSideEffectStep(stepName) && isUnresolvedResearch(state)) {
            state = addHistoryEntry(
                    state,
                    stepName,
                    step.args(),
                    "BLOCKED",
                    "Research summary unresolved",
                    null
            );
            stateStore.save(state);
            return state;
        }

        if (isSmsSideEffectStep(stepName) && isUnresolvedSms(state)) {
            state = addHistoryEntry(
                    state,
                    stepName,
                    step.args(),
                    "BLOCKED",
                    "SMS message unresolved",
                    null
            );
            stateStore.save(state);
            return state;
        }

        if (("send".equals(stepName) || "send_email".equals(stepName))
                && state.email() != null
                && state.email().status() == AgentSessionState.EmailStatus.SENT) {
            log.warn("Skipping {} - already sent. sessionId={}", stepName, state.sessionId());
            state = addHistoryEntry(
                    state,
                    stepName,
                    step.args(),
                    "SKIPPED",
                    "Already sent",
                    null
            );
            stateStore.save(state);
            return state;
        }

        if ("send_sms".equals(stepName)
                && state.sms() != null
                && state.sms().status() == AgentSessionState.SmsStatus.SENT) {
            log.warn("Skipping send_sms - already sent. sessionId={}", state.sessionId());
            state = addHistoryEntry(
                    state,
                    stepName,
                    step.args(),
                    "SKIPPED",
                    "SMS already sent",
                    null
            );
            stateStore.save(state);
            return state;
        }

        if ("draft_email".equals(stepName)) {
            String recipient = getStringArg(step.args(), "recipient", getRecipient(state));
            String subject = getStringArg(step.args(), "subject", getSubject(state));

            if (sameDraftAlreadyExists(state, recipient, subject)) {
                log.warn("Skipping draft_email - identical draft already exists. sessionId={}", state.sessionId());
                state = addHistoryEntry(
                        state,
                        "draft_email",
                        step.args(),
                        "SKIPPED",
                        "Identical draft already exists",
                        null
                );
                stateStore.save(state);
                return state;
            }
        }

        AgentTool tool = tools.get(stepName);
        if (tool == null) {
            throw new IllegalStateException("Tool not registered: " + stepName);
        }

        state = addHistoryEntry(state, stepName, step.args(), "STARTED", null, null);
        stateStore.save(state);

        state = tool.execute(state, step);

        state = addHistoryEntry(
                state,
                stepName,
                step.args(),
                "SUCCESS",
                resultMessageForStep(stepName),
                null
        );
        stateStore.save(state);

        return state;
    }

    private boolean isEmailSideEffectStep(String stepName) {
        return "draft_email".equals(stepName)
                || "send_email".equals(stepName)
                || "send".equals(stepName);
    }

    private boolean isSmsSideEffectStep(String stepName) {
        return "send_sms".equals(stepName);
    }

    private String resultMessageForStep(String stepName) {
        return switch (stepName) {
            case "email" -> "Email content generated";
            case "draft_email" -> "Email drafted";
            case "send_email", "send" -> "Email sent";
            case "shorten_email" -> "Email body shortened";
            case "update_recipient" -> "Recipient updated";
            case "compose_sms" -> "SMS content generated";
            case "send_sms" -> "SMS sent";
            default -> stepName + " completed";
        };
    }

    private AgentSessionState initState(String sessionId, AgentRequest request) {
        return stateStore.save(AgentSessionState.empty(sessionId, request.prompt(), request.docId()));
    }

    private AgentSessionState updateResearch(AgentSessionState state, ResearchResult result) {
        return state
                .withResearch(new AgentSessionState.ResearchSnapshot(
                        result.answer(),
                        result.citations(),
                        result.retrievalScore(),
                        result.judge(),
                        result.chunks()
                ))
                .withResearchAttempts(state.researchAttempts() + 1);
    }

    private AgentResponse buildResponse(AgentSessionState state) {
        AgentSessionState.ResearchSnapshot research = state.research();

        double retrievalScore = research != null && research.retrievalScore() != null
                ? research.retrievalScore()
                : 0.0;

        String summary = research != null && research.summary() != null
                ? research.summary()
                : "";

        List<String> citations = research != null && research.citations() != null
                ? research.citations()
                : List.of();

        String status = state.email() != null
                ? state.email().status().name()
                : state.sms() != null
                ? state.sms().status().name()
                : "NOT_STARTED";

        return new AgentResponse(
                state.sessionId(),
                summary,
                citations,
                retrievalScore,
                research != null ? research.judge() : null,
                status
        );
    }

    private AgentPlan singleResearchPlan(String query) {
        return new AgentPlan(List.of(
                new PlanStep("research", Map.of("query", query))
        ));
    }

    private AgentPlan stopPlan() {
        return new AgentPlan(List.of(
                new PlanStep("stop", Map.of())
        ));
    }

    private String refineQuery(String query) {
        return query + " explained with implementation details, example, and controller advice pattern";
    }

    private String getRecipient(AgentSessionState state) {
        return state.email() != null && state.email().recipient() != null
                ? state.email().recipient()
                : DEFAULT_RECIPIENT;
    }

    private String getSubject(AgentSessionState state) {
        return state.email() != null && state.email().subject() != null
                ? state.email().subject()
                : DEFAULT_SUBJECT;
    }

    private String getPhoneNumber(AgentSessionState state) {
        return state.sms() != null && state.sms().phoneNumber() != null
                ? state.sms().phoneNumber()
                : "+10000000000";
    }

    private String getStringArg(Map<String, Object> args, String key, String defaultValue) {
        if (args == null) {
            return defaultValue;
        }

        Object value = args.get(key);
        if (value == null) {
            return defaultValue;
        }

        String s = String.valueOf(value).trim();
        return s.isEmpty() ? defaultValue : s;
    }

    private boolean sameDraftAlreadyExists(AgentSessionState state, String recipient, String subject) {
        return state.email() != null
                && state.email().status() == AgentSessionState.EmailStatus.DRAFTED
                && recipient != null
                && subject != null
                && recipient.equalsIgnoreCase(state.email().recipient())
                && subject.equals(state.email().subject());
    }

    private boolean canRetry(AgentSessionState state) {
        return state.researchAttempts() < MAX_RETRIES;
    }

    private boolean shouldRetryResearch(ResearchResult result) {
        if (result == null) {
            return true;
        }

        String answer = result.answer() != null ? result.answer().toLowerCase() : "";
        boolean saysIDontKnow = answer.contains("i don't know based on the ingested documents");
        boolean hasChunks = result.chunks() != null && !result.chunks().isEmpty();

        return saysIDontKnow && hasChunks;
    }

    private boolean isUnresolvedResearch(AgentSessionState state) {
        return state.research() != null
                && state.research().summary() != null
                && state.research().summary().toLowerCase().contains("i don't know based on the ingested documents");
    }

    private AgentSessionState addHistoryEntry(
            AgentSessionState state,
            String step,
            Map<String, Object> args,
            String status,
            String result,
            String error
    ) {
        List<StepHistoryEntry> newHistory = new ArrayList<>();

        if (state.history() != null) {
            newHistory.addAll(state.history());
        }

        newHistory.add(new StepHistoryEntry(
                step,
                args,
                status,
                result,
                error,
                Instant.now()
        ));

        return state.withHistory(newHistory);
    }
}