package com.example.airagassistant.agentic;

import com.example.airagassistant.agentic.dto.AgentObservation;
import com.example.airagassistant.agentic.dto.AgentPlan;
import com.example.airagassistant.agentic.dto.PlanStep;
import com.example.airagassistant.agentic.state.AgentSessionState;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

@Service
@RequiredArgsConstructor
@Slf4j
public class PlannerService {

    private static final String DEFAULT_RECIPIENT = "hr@company.com";
    private static final String DEFAULT_SUBJECT = "Requested Summary";
    private static final String DEFAULT_PHONE = "+10000000000";

    private static final Set<String> INITIAL_ALLOWED_STEPS =
            Set.of("research", "email", "send", "compose_sms", "send_sms",
                    "fetch_vehicle_specs", "generate_vehicle_summary",
                    "compare_vehicles", "enrich_vehicle_data", "stop");

    private static final Set<String> CONTINUATION_ALLOWED_STEPS =
            Set.of("update_recipient", "shorten_email", "draft_email", "send_email",
                    "compose_sms", "send_sms",
                    "fetch_vehicle_specs", "generate_vehicle_summary",
                    "compare_vehicles", "enrich_vehicle_data", "stop");

    private final ChatClient chatClient;

    private final ObjectMapper objectMapper = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    // =========================
    // INITIAL PLAN
    // =========================
    public AgentPlan plan(String userPrompt) {

        String prompt = """
                You are a workflow planner.

                Convert the user request into workflow steps.

                Allowed steps:
                - research -> args: query
                - email -> args: recipient, subject
                - send -> args: recipient, subject
                - compose_sms -> args: phoneNumber
                - send_sms -> args: phoneNumber
                - fetch_vehicle_specs -> args: vehicleId, question (optional), topK (optional)
                - generate_vehicle_summary -> args: vehicleId
                - compare_vehicles -> args: vehicleIds (comma-separated), question
                - enrich_vehicle_data -> args: vehicleId
                - stop

                Step meanings:
                - research = retrieve information from ingested PDF or text documents only
                - email = generate email content in workflow state only (NO external side effect)
                - send = send email externally now
                - compose_sms = generate SMS content in workflow state only (NO external side effect)
                - send_sms = send SMS externally now
                - fetch_vehicle_specs = retrieve raw spec data for a specific vehicle from the vector store
                - generate_vehicle_summary = write a consumer narrative summary for a vehicle (use after fetch_vehicle_specs)
                - compare_vehicles = compare two or more vehicles side-by-side on a given dimension
                - enrich_vehicle_data = auto-generate or improve a vehicle summary and re-ingest it

                Vehicle routing rules (IMPORTANT):
                - If the user asks about a specific vehicle's specs, features, performance, or wants a summary -> use fetch_vehicle_specs then generate_vehicle_summary
                - If the user asks to compare vehicles -> use compare_vehicles with vehicleIds as comma-separated string
                - If the user asks to enrich or improve a vehicle summary -> use enrich_vehicle_data
                - Vehicle steps are SEPARATE from research — do NOT use research for vehicle queries
                - Extract vehicleId from context: "Tesla Model 3 Long Range" -> "tesla-model3-2025-long-range", "BMW M3" -> "bmw-m3-2025-competition"
                - If docId is provided in context and looks like a vehicleId (contains hyphens, a year), use it directly

                Rules:
                - If the user only asks a question about a document or general knowledge, return ONLY a research step.
                - If the user asks for an answer to be emailed, separate research from email.
                - If the user asks for an answer to be texted or sent by SMS, use compose_sms and/or send_sms.
                - The research query must contain ONLY the information request.
                - If the user asks to email someone, include an email step.
                - If the user asks to text or send SMS, include a compose_sms or send_sms step.
                - If the user ONLY wants to draft or prepare an email and does NOT require external knowledge, return ONLY an email step.
                - If the user ONLY wants to send an email now and does NOT require external knowledge, return ONLY a send step.
                - If the user ONLY wants to compose or prepare an SMS and does NOT require external knowledge, return ONLY a compose_sms step.
                - If the user ONLY wants to send an SMS now and does NOT require external knowledge, return ONLY a send_sms step.
                - Do NOT include a research step unless information retrieval is needed.
                - If the recipient is not explicitly provided, you may use a placeholder recipient like "hr@company.com".
                - If the phone number is not explicitly provided, you may use a placeholder like "+10000000000".
                - Do NOT generate full email or SMS body text in the plan.
                - Do NOT include email or SMS wording inside the research query.
                - If the user explicitly says "send now", use send or send_sms instead of email or compose_sms.
                - Return ONLY strict JSON using this schema:
                  {
                    "steps": [
                      { "step": "research", "args": { "query": "spring boot answers" } },
                      { "step": "email", "args": { "recipient": "hr@company.com", "subject": "Spring Boot Answers" } }
                    ]
                  }

                User request:
                %s
                """.formatted(safe(userPrompt));

        String response = chatClient.prompt(prompt).call().content();
        log.info("Planner response: {}", response);

        AgentPlan parsed = parseAndValidateInitialPlan(response);
        return enforceExecutionRules(parsed);
    }

    // =========================
    // REPLAN AFTER OBSERVATION
    // =========================
    public AgentPlan replan(AgentSessionState state, AgentObservation observation) {

        String prompt = """
                You are replanning after a completed or failed workflow step.

                Original user request:
                %s

                Current workflow state:
                - research exists: %s
                - email exists: %s
                - email status: %s
                - sms exists: %s
                - sms status: %s
                - current recipient: %s
                - current subject: %s
                - current phone number: %s
                - execution history: %s

                Last step result:
                - step: %s
                - confidence: %s
                - judge grounded: %s
                - judge score: %s

                Decide the next step.

                Allowed steps:
                - research -> args: query
                - email -> args: recipient, subject
                - send -> args: recipient, subject
                - compose_sms -> args: phoneNumber
                - send_sms -> args: phoneNumber
                - stop

                Step meanings:
                - research = retrieve information
                - email = generate email content in workflow state only
                - send = send email externally now
                - compose_sms = generate SMS content in workflow state only
                - send_sms = send SMS externally now

                Rules:
                - If the last research result is good enough, continue only with the communication steps the original request asked for.
                - If the last research result is weak, you may request one more research step with a refined query.
                - If research has already failed multiple times, avoid infinite retries.
                - If the original request was only Q&A, return stop after good research.
                - If email already exists:
                  - DO NOT change recipient unless explicitly requested.
                  - DO NOT change subject unless explicitly requested.
                - If sms already exists:
                  - DO NOT change phone number unless explicitly requested.
                - Avoid repeating already successful steps unless necessary.
                - Return ONLY strict JSON.
                """.formatted(
                safe(state.originalUserRequest()),
                state.research() != null,
                state.email() != null,
                state.email() != null ? state.email().status().name() : "NOT_STARTED",
                state.sms() != null,
                state.sms() != null ? state.sms().status().name() : "NOT_STARTED",
                getRecipient(state),
                getSubject(state),
                getPhoneNumber(state),
                summarizeHistory(state),
                observation != null ? safe(observation.lastStep()) : "",
                observation != null ? observation.confidence() : null,
                observation != null && observation.judge() != null ? observation.judge().grounded() : null,
                observation != null && observation.judge() != null ? observation.judge().score() : null
        );

        String response = chatClient.prompt(prompt).call().content();
        log.info("Replan response: {}", response);

        AgentPlan parsed = parseAndValidateInitialPlan(response);
        return enforceExecutionRules(parsed);
    }

    // =========================
    // CONTINUATION PLAN
    // =========================
    public AgentPlan planContinuation(AgentSessionState state, String instruction) {

        String lower = instruction == null ? "" : instruction.toLowerCase(Locale.ROOT);

        if (lower.contains("send sms") || lower.contains("send text") || lower.contains("text it") || lower.contains("sms it")) {
            return new AgentPlan(List.of(
                    new PlanStep("send_sms", Map.of(
                            "phoneNumber", getPhoneNumber(state)
                    ))
            ));
        }

        if (lower.contains("text") || lower.contains("sms")) {
            return new AgentPlan(List.of(
                    new PlanStep("compose_sms", Map.of(
                            "phoneNumber", getPhoneNumber(state)
                    )),
                    new PlanStep("send_sms", Map.of(
                            "phoneNumber", getPhoneNumber(state)
                    ))
            ));
        }

        if (lower.contains("send")) {
            return new AgentPlan(List.of(
                    new PlanStep("send_email", Map.of(
                            "recipient", getRecipient(state),
                            "subject", getSubject(state)
                    ))
            ));
        }

        if (lower.contains("shorten")) {
            return new AgentPlan(List.of(
                    new PlanStep("shorten_email", Map.of()),
                    new PlanStep("draft_email", Map.of(
                            "recipient", getRecipient(state),
                            "subject", getSubject(state)
                    ))
            ));
        }

        if (lower.contains("draft")) {
            return new AgentPlan(List.of(
                    new PlanStep("draft_email", Map.of(
                            "recipient", getRecipient(state),
                            "subject", getSubject(state)
                    ))
            ));
        }

        String prompt = """
                You are continuing an existing workflow.

                Current workflow state:
                - original request: %s
                - current request: %s
                - research exists: %s
                - current recipient: %s
                - current subject: %s
                - current email status: %s
                - current phone number: %s
                - current sms status: %s
                - execution history: %s

                New user instruction:
                %s

                Allowed steps:
                - update_recipient -> args: recipient
                - shorten_email
                - draft_email -> args: recipient, subject
                - send_email -> args: recipient, subject
                - compose_sms -> args: phoneNumber
                - send_sms -> args: phoneNumber
                - stop

                Step meanings:
                - update_recipient = change recipient in workflow state
                - shorten_email = modify existing email body in workflow state
                - draft_email = persist or refresh draft externally
                - send_email = send externally
                - compose_sms = generate SMS content in workflow state
                - send_sms = send SMS externally

                Rules:
                - Reuse existing research; do NOT request research again if a summary already exists.
                - If the user asks to change recipient, update recipient first and then draft again.
                - If the user asks to shorten, shorten email and then draft again.
                - If the user asks to send, use send_email.
                - If the user asks to text or send SMS, use compose_sms and/or send_sms.
                - If email already exists:
                  - DO NOT change recipient unless the user asks.
                  - DO NOT change subject unless the user explicitly asks.
                  - Only modify body-related behavior when requested.
                - If sms already exists:
                  - DO NOT change phone number unless the user explicitly asks.
                - Avoid repeating already successful steps unless necessary.
                - Return ONLY strict JSON.
                """.formatted(
                safe(state.originalUserRequest()),
                safe(state.currentUserRequest()),
                state.research() != null,
                getRecipient(state),
                getSubject(state),
                state.email() != null ? state.email().status().name() : "NOT_STARTED",
                getPhoneNumber(state),
                state.sms() != null ? state.sms().status().name() : "NOT_STARTED",
                summarizeHistory(state),
                safe(instruction)
        );

        String response = chatClient.prompt(prompt).call().content();
        log.info("Continuation planner response: {}", response);

        AgentPlan parsed = parseAndValidateContinuationPlan(response, state);
        return enforceContinuationRules(parsed, state);
    }

    // =========================
    // PARSE + VALIDATE
    // =========================

    private AgentPlan parseAndValidateInitialPlan(String response) {
        PlannerPayload payload = parsePayload(response);

        if (payload == null || payload.steps() == null || payload.steps().isEmpty()) {
            log.warn("Planner returned invalid or empty initial/replan payload. Returning stop.");
            return stopPlan();
        }

        List<PlanStep> validated = new ArrayList<>();

        for (PlannerStep rawStep : payload.steps()) {
            if (rawStep == null || rawStep.step() == null || rawStep.step().isBlank()) {
                continue;
            }

            String stepName = normalizeStepName(rawStep.step());
            if (!INITIAL_ALLOWED_STEPS.contains(stepName)) {
                log.warn("Ignoring unsupported initial step: {}", rawStep.step());
                continue;
            }

            Map<String, Object> args = normalizeArgs(rawStep.args());

            switch (stepName) {
                case "research" -> {
                    String query = getString(args, "query");
                    if (isBlank(query)) {
                        log.warn("Ignoring research step with blank query.");
                        continue;
                    }

                    validated.add(new PlanStep("research", Map.of(
                            "query", query.trim()
                    )));
                }

                case "email", "send" -> {
                    String recipient = defaultIfBlank(getString(args, "recipient"), DEFAULT_RECIPIENT);
                    String subject = defaultIfBlank(getString(args, "subject"), DEFAULT_SUBJECT);

                    validated.add(new PlanStep(stepName, Map.of(
                            "recipient", recipient,
                            "subject", subject
                    )));
                }

                case "compose_sms", "send_sms" -> {
                    String phoneNumber = defaultIfBlank(getString(args, "phoneNumber"), DEFAULT_PHONE);

                    validated.add(new PlanStep(stepName, Map.of(
                            "phoneNumber", phoneNumber
                    )));
                }

                case "stop" -> validated.add(new PlanStep("stop", Map.of()));

                // Vehicle tool steps — pass through args as-is, no special validation needed
                case "fetch_vehicle_specs" -> {
                    String vehicleId = getString(args, "vehicleId");
                    if (isBlank(vehicleId)) {
                        log.warn("Ignoring fetch_vehicle_specs step with blank vehicleId.");
                        continue;
                    }
                    validated.add(new PlanStep(stepName, new LinkedHashMap<>(args)));
                }

                case "generate_vehicle_summary" -> {
                    String vehicleId = getString(args, "vehicleId");
                    if (isBlank(vehicleId)) {
                        log.warn("Ignoring generate_vehicle_summary step with blank vehicleId.");
                        continue;
                    }
                    validated.add(new PlanStep(stepName, new LinkedHashMap<>(args)));
                }

                case "compare_vehicles" -> {
                    String vehicleIds = getString(args, "vehicleIds");
                    if (isBlank(vehicleIds)) {
                        log.warn("Ignoring compare_vehicles step with blank vehicleIds.");
                        continue;
                    }
                    validated.add(new PlanStep(stepName, new LinkedHashMap<>(args)));
                }

                case "enrich_vehicle_data" -> {
                    String vehicleId = getString(args, "vehicleId");
                    if (isBlank(vehicleId)) {
                        log.warn("Ignoring enrich_vehicle_data step with blank vehicleId.");
                        continue;
                    }
                    validated.add(new PlanStep(stepName, new LinkedHashMap<>(args)));
                }

                default -> {
                }
            }
        }

        if (validated.isEmpty()) {
            log.warn("Planner produced no usable initial steps. Returning stop.");
            return stopPlan();
        }

        return new AgentPlan(validated);
    }

    private AgentPlan parseAndValidateContinuationPlan(String response, AgentSessionState state) {
        PlannerPayload payload = parsePayload(response);

        if (payload == null || payload.steps() == null || payload.steps().isEmpty()) {
            log.warn("Planner returned invalid or empty continuation payload. Returning stop.");
            return stopPlan();
        }

        List<PlanStep> validated = new ArrayList<>();

        for (PlannerStep rawStep : payload.steps()) {
            if (rawStep == null || rawStep.step() == null || rawStep.step().isBlank()) {
                continue;
            }

            String stepName = normalizeStepName(rawStep.step());
            if (!CONTINUATION_ALLOWED_STEPS.contains(stepName)) {
                log.warn("Ignoring unsupported continuation step: {}", rawStep.step());
                continue;
            }

            Map<String, Object> args = normalizeArgs(rawStep.args());

            switch (stepName) {
                case "update_recipient" -> {
                    String recipient = getString(args, "recipient");
                    if (isBlank(recipient)) {
                        log.warn("Ignoring update_recipient with blank recipient.");
                        continue;
                    }

                    validated.add(new PlanStep("update_recipient", Map.of(
                            "recipient", recipient.trim()
                    )));
                }

                case "shorten_email" -> validated.add(new PlanStep("shorten_email", Map.of()));

                case "draft_email", "send_email" -> {
                    String recipient = defaultIfBlank(getString(args, "recipient"), getRecipient(state));
                    String subject = defaultIfBlank(getString(args, "subject"), getSubject(state));

                    validated.add(new PlanStep(stepName, Map.of(
                            "recipient", recipient,
                            "subject", subject
                    )));
                }

                case "compose_sms", "send_sms" -> {
                    String phoneNumber = defaultIfBlank(getString(args, "phoneNumber"), getPhoneNumber(state));

                    validated.add(new PlanStep(stepName, Map.of(
                            "phoneNumber", phoneNumber
                    )));
                }

                case "stop" -> validated.add(new PlanStep("stop", Map.of()));

                // Vehicle tool steps — pass through args as-is, no special validation needed
                case "fetch_vehicle_specs" -> {
                    String vehicleId = getString(args, "vehicleId");
                    if (isBlank(vehicleId)) {
                        log.warn("Ignoring fetch_vehicle_specs step with blank vehicleId.");
                        continue;
                    }
                    validated.add(new PlanStep(stepName, new LinkedHashMap<>(args)));
                }

                case "generate_vehicle_summary" -> {
                    String vehicleId = getString(args, "vehicleId");
                    if (isBlank(vehicleId)) {
                        log.warn("Ignoring generate_vehicle_summary step with blank vehicleId.");
                        continue;
                    }
                    validated.add(new PlanStep(stepName, new LinkedHashMap<>(args)));
                }

                case "compare_vehicles" -> {
                    String vehicleIds = getString(args, "vehicleIds");
                    if (isBlank(vehicleIds)) {
                        log.warn("Ignoring compare_vehicles step with blank vehicleIds.");
                        continue;
                    }
                    validated.add(new PlanStep(stepName, new LinkedHashMap<>(args)));
                }

                case "enrich_vehicle_data" -> {
                    String vehicleId = getString(args, "vehicleId");
                    if (isBlank(vehicleId)) {
                        log.warn("Ignoring enrich_vehicle_data step with blank vehicleId.");
                        continue;
                    }
                    validated.add(new PlanStep(stepName, new LinkedHashMap<>(args)));
                }

                default -> {
                }
            }
        }

        if (validated.isEmpty()) {
            log.warn("Planner produced no usable continuation steps. Returning stop.");
            return stopPlan();
        }

        return new AgentPlan(validated);
    }

    // =========================
    // ENFORCEMENT
    // =========================

    private AgentPlan enforceExecutionRules(AgentPlan plan) {
        if (plan == null || plan.steps() == null || plan.steps().isEmpty()) {
            return stopPlan();
        }

        List<PlanStep> validated = new ArrayList<>();
        boolean hasResearch = false;

        for (PlanStep step : plan.steps()) {
            if (step == null || step.step() == null || step.step().isBlank()) {
                continue;
            }

            String name = step.step().trim().toLowerCase(Locale.ROOT);

            switch (name) {
                case "research" -> {
                    String query = getString(step.args(), "query");
                    if (isBlank(query)) {
                        log.warn("Invalid research query -> skipping");
                        continue;
                    }

                    hasResearch = true;
                    validated.add(new PlanStep("research", Map.of(
                            "query", query.trim()
                    )));
                }

                case "email" -> {
                    if (!hasResearch && requiresResearch(plan)) {
                        log.warn("Email before research -> skipping until research happens");
                        continue;
                    }

                    validated.add(normalizeEmailLikeStep("email", step.args()));
                }

                case "send" -> {
                    boolean hasEmail = validated.stream()
                            .anyMatch(s -> "email".equalsIgnoreCase(s.step()));

                    if (!hasEmail && requiresResearch(plan) && !hasResearch) {
                        log.warn("Send before research/email -> skipping send");
                        continue;
                    }

                    if (!hasEmail && !requiresResearch(plan)) {
                        validated.add(normalizeEmailLikeStep("send", step.args()));
                        continue;
                    }

                    if (!hasEmail) {
                        validated.add(normalizeEmailLikeStep("email", step.args()));
                        continue;
                    }

                    validated.add(normalizeEmailLikeStep("send", step.args()));
                }

                case "compose_sms" -> validated.add(new PlanStep(
                        "compose_sms",
                        Map.of("phoneNumber", defaultIfBlank(getString(step.args(), "phoneNumber"), DEFAULT_PHONE))
                ));

                case "send_sms" -> validated.add(new PlanStep(
                        "send_sms",
                        Map.of("phoneNumber", defaultIfBlank(getString(step.args(), "phoneNumber"), DEFAULT_PHONE))
                ));

                case "stop" -> validated.add(new PlanStep("stop", Map.of()));

                // Vehicle steps pass through as-is — no ordering constraints needed
                case "fetch_vehicle_specs",
                     "generate_vehicle_summary",
                     "compare_vehicles",
                     "enrich_vehicle_data" -> validated.add(new PlanStep(name, new LinkedHashMap<>(step.args() != null ? step.args() : Map.of())));

                default -> log.warn("Unsupported step during enforcement: {}", name);
            }
        }

        if (validated.isEmpty()) {
            return stopPlan();
        }

        return new AgentPlan(validated);
    }

    private AgentPlan enforceContinuationRules(AgentPlan plan, AgentSessionState state) {
        if (plan == null || plan.steps() == null || plan.steps().isEmpty()) {
            return stopPlan();
        }

        List<PlanStep> validated = new ArrayList<>();
        boolean recipientUpdated = false;
        boolean shortened = false;

        for (PlanStep step : plan.steps()) {
            if (step == null || step.step() == null || step.step().isBlank()) {
                continue;
            }

            String name = step.step().trim().toLowerCase(Locale.ROOT);

            switch (name) {
                case "update_recipient" -> {
                    String recipient = getString(step.args(), "recipient");
                    if (isBlank(recipient)) {
                        log.warn("Invalid continuation recipient update -> skipping");
                        continue;
                    }

                    recipientUpdated = true;
                    validated.add(new PlanStep("update_recipient", Map.of(
                            "recipient", recipient.trim()
                    )));
                }

                case "shorten_email" -> {
                    shortened = true;
                    validated.add(new PlanStep("shorten_email", Map.of()));
                }

                case "draft_email" -> validated.add(
                        normalizeContinuationEmailStep("draft_email", step.args(), state)
                );

                case "send_email" -> {
                    boolean hasDraft = validated.stream()
                            .anyMatch(s -> "draft_email".equalsIgnoreCase(s.step()));

                    if ((recipientUpdated || shortened) && !hasDraft) {
                        validated.add(new PlanStep("draft_email", Map.of(
                                "recipient", getRecipient(state),
                                "subject", getSubject(state)
                        )));
                    }

                    validated.add(normalizeContinuationEmailStep("send_email", step.args(), state));
                }

                case "compose_sms" -> validated.add(new PlanStep(
                        "compose_sms",
                        Map.of("phoneNumber", defaultIfBlank(getString(step.args(), "phoneNumber"), getPhoneNumber(state)))
                ));

                case "send_sms" -> {
                    boolean hasCompose = validated.stream()
                            .anyMatch(s -> "compose_sms".equalsIgnoreCase(s.step()));

                    String phoneNumber = defaultIfBlank(getString(step.args(), "phoneNumber"), getPhoneNumber(state));

                    if (!hasCompose) {
                        validated.add(new PlanStep("compose_sms", Map.of(
                                "phoneNumber", phoneNumber
                        )));
                    }

                    validated.add(new PlanStep("send_sms", Map.of(
                            "phoneNumber", phoneNumber
                    )));
                }

                case "stop" -> validated.add(new PlanStep("stop", Map.of()));

                case "fetch_vehicle_specs",
                     "generate_vehicle_summary",
                     "compare_vehicles",
                     "enrich_vehicle_data" -> validated.add(new PlanStep(name, new LinkedHashMap<>(step.args() != null ? step.args() : Map.of())));

                default -> log.warn("Unsupported continuation step during enforcement: {}", name);
            }
        }

        if (validated.isEmpty()) {
            return stopPlan();
        }

        return new AgentPlan(validated);
    }

    private boolean requiresResearch(AgentPlan plan) {
        return plan != null
                && plan.steps() != null
                && plan.steps().stream()
                .filter(step -> step != null && step.step() != null)
                .anyMatch(step -> "research".equalsIgnoreCase(step.step()));
    }

    private PlanStep normalizeEmailLikeStep(String stepName, Map<String, Object> args) {
        String recipient = defaultIfBlank(getString(args, "recipient"), DEFAULT_RECIPIENT);
        String subject = defaultIfBlank(getString(args, "subject"), DEFAULT_SUBJECT);

        return new PlanStep(stepName, Map.of(
                "recipient", recipient,
                "subject", subject
        ));
    }

    private PlanStep normalizeContinuationEmailStep(String stepName,
                                                    Map<String, Object> args,
                                                    AgentSessionState state) {
        String recipient = defaultIfBlank(getString(args, "recipient"), getRecipient(state));
        String subject = defaultIfBlank(getString(args, "subject"), getSubject(state));

        return new PlanStep(stepName, Map.of(
                "recipient", recipient,
                "subject", subject
        ));
    }

    private String summarizeHistory(AgentSessionState state) {
        if (state == null || state.history() == null || state.history().isEmpty()) {
            return "No prior execution history.";
        }

        return state.history().stream()
                .map(h -> "%s(%s)".formatted(h.step(), h.status()))
                .reduce((a, b) -> a + ", " + b)
                .orElse("No prior execution history.");
    }

    private PlannerPayload parsePayload(String response) {
        if (response == null || response.isBlank()) {
            return null;
        }

        String json = null;
        String cleaned = null;

        try {
            json = extractJsonObject(response);
            cleaned = cleanJson(json);
            return objectMapper.readValue(cleaned, PlannerPayload.class);
        } catch (Exception e) {
            log.warn("Failed to parse planner JSON. extracted={}, cleaned={}", json, cleaned, e);
            return null;
        }
    }
    private String cleanJson(String text) {
        if (text == null || text.isBlank()) {
            return "{}";
        }

        String trimmed = text.trim();

        if (trimmed.startsWith("```")) {
            trimmed = trimmed
                    .replaceFirst("^```json\\s*", "")
                    .replaceFirst("^```\\s*", "")
                    .replaceFirst("\\s*```$", "")
                    .trim();
        }

        trimmed = trimmed.replaceAll("(?m)//.*$", "");
        trimmed = trimmed.replaceAll("(?s)/\\*.*?\\*/", "");

        return trimmed.trim();
    }
    private String extractJsonObject(String text) {
        String trimmed = text.trim();

        if (trimmed.startsWith("```")) {
            trimmed = trimmed
                    .replaceFirst("^```json\\s*", "")
                    .replaceFirst("^```\\s*", "")
                    .replaceFirst("\\s*```$", "")
                    .trim();
        }

        int firstBrace = trimmed.indexOf('{');
        int lastBrace = trimmed.lastIndexOf('}');
        if (firstBrace >= 0 && lastBrace > firstBrace) {
            return trimmed.substring(firstBrace, lastBrace + 1);
        }

        return trimmed;
    }

    private String normalizeStepName(String step) {
        return step == null ? "" : step.trim().toLowerCase(Locale.ROOT);
    }

    private Map<String, Object> normalizeArgs(Map<String, Object> args) {
        return args == null ? Map.of() : new LinkedHashMap<>(args);
    }

    private String getString(Map<String, Object> args, String key) {
        if (args == null) {
            return null;
        }

        Object value = args.get(key);
        return value == null ? null : String.valueOf(value).trim();
    }

    private String defaultIfBlank(String value, String defaultValue) {
        return isBlank(value) ? defaultValue : value.trim();
    }

    private boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    private String safe(String value) {
        return value == null ? "" : value;
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
                : DEFAULT_PHONE;
    }

    private AgentPlan stopPlan() {
        return new AgentPlan(List.of(
                new PlanStep("stop", Map.of())
        ));
    }

    private record PlannerPayload(List<PlannerStep> steps) {}

    private record PlannerStep(String step, Map<String, Object> args) {}
}