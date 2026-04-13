package com.example.airagassistant.agentic;

import com.example.airagassistant.agentic.dto.AgentObservation;
import com.example.airagassistant.agentic.dto.AgentPlan;
import com.example.airagassistant.agentic.dto.PlanStep;
import com.example.airagassistant.agentic.state.AgentSessionState;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class bkp_PlannerService {

    private final ChatClient chatClient;
    private final ObjectMapper objectMapper;

    public AgentPlan plan(String userPrompt) {
        String prompt = """
                You are an AI workflow planner.
                
                Convert the user request into execution steps.
                
                Allowed steps:
                - research -> requires "query"
                - email -> requires "recipient", "subject"
                - stop -> no args
                
                Rules:
                - Return ONLY valid JSON.
                - Do NOT wrap JSON in markdown.
                - Do NOT include explanations before or after JSON.
                - If the user asks to email someone, include an email step.
                - If the user ONLY wants to send an email and does NOT require external knowledge, return ONLY an email step.
                - Do NOT include a research step unless information retrieval is needed.
                - If the recipient is not explicitly provided, you may use a placeholder recipient like "hr@company.com".
                - Do NOT generate email body content.
                
                Example:
                {
                  "steps": [
                    { "step": "research", "args": { "query": "spring boot answers" } },
                    { "step": "
                    ", "args": { "recipient": "hr@company.com", "subject": "Spring Boot Answers" } }
                  ]
                }
                
                User request:
                %s
                """.formatted(userPrompt);

        String response = chatClient.prompt(prompt).call().content();
        log.info("Planner initial response: {}", response);

        String cleaned = extractJson(response);
        return parsePlan(cleaned, userPrompt);
    }

    public AgentPlan replan(String userPrompt, AgentObservation observation) {
        String prompt = """
                You are an AI agent replanner.
                
                You previously executed a step and got this result:
                
                Step: %s
                Summary: %s
                Confidence: %s
                Judge Score: %s
                Grounded: %s
                Correct: %s
                Complete: %s
                
                Decide the next step.
                
                Allowed steps:
                - refine_query -> requires "query"
                - research -> requires "query"
                - email -> requires "recipient", "subject"
                - stop -> no args
                
                Rules:
                - Return ONLY valid JSON.
                - Do NOT wrap JSON in markdown.
                - Do NOT include explanations before or after JSON.
                - Do NOT generate email body content.
                - Use "refine_query" when the previous research result is weak, incomplete, or needs a more precise query.
                - Use "email" only when the result is good enough to act on.
                - Use "stop" when no further useful action is needed.
                
                Example:
                {
                  "steps": [
                    { "step": "refine_query", "args": { "query": "summarize Spring Boot exception handling with examples" } }
                  ]
                }
                """.formatted(
                observation.lastStep(),
                safe(observation.summary()),
                observation.confidence(),
                observation.judge() != null ? observation.judge().score() : 0.0,
                observation.judge() != null && observation.judge().grounded(),
                observation.judge() != null && observation.judge().correct(),
                observation.judge() != null && observation.judge().complete()
        );

        log.info("Observer starting replan...");
        String response = chatClient.prompt(prompt).call().content();
        log.info("Observer replan raw response: {}", response);

        String cleaned = extractJson(response);
        return parsePlan(cleaned, userPrompt);
    }
    public AgentPlan planContinuation(AgentSessionState state, String instruction) {

        String prompt = """
            You are continuing an existing workflow.

            Current workflow state:
            - Original request: %s
            - Current request: %s
            - Research summary exists: %s
            - Current recipient: %s
            - Current email status: %s

            New user instruction:
            %s

            Allowed steps:
            - update_recipient → args: recipient
            - shorten_email → no args
            - draft_email → args: recipient, subject
            - send_email → args: recipient, subject
            - stop → no args

            Rules:
            - Do not request research again if a research summary already exists.
            - Reuse existing state whenever possible.
            - If the user asks to change recipient, return update_recipient.
            - If the user asks to send, return send_email.
            - If the user asks to shorten, return shorten_email.
            - Return ONLY JSON.

            Example:
            {
              "steps": [
                { "step": "update_recipient", "args": { "recipient": "finance@company.com" } },
                { "step": "draft_email", "args": { "recipient": "finance@company.com", "subject": "Requested Summary" } }
              ]
            }
            """.formatted(
                state.originalUserRequest(),
                state.currentUserRequest(),
                state.research() != null,
                state.email() != null ? state.email().recipient() : "",
                state.email() != null ? state.email().status().name() : "NOT_STARTED",
                instruction
        );

        String response = chatClient.prompt(prompt).call().content();

        // For now, safe deterministic fallback
        return fallbackContinuationPlan(state, instruction);
    }
    private AgentPlan parsePlan(String json, String fallbackPrompt) {
        try {
            @SuppressWarnings("unchecked")
            Map<String, Object> root = objectMapper.readValue(json, Map.class);

            @SuppressWarnings("unchecked")
            List<Map<String, Object>> stepsRaw = (List<Map<String, Object>>) root.get("steps");

            if (stepsRaw == null || stepsRaw.isEmpty()) {
                log.warn("Planner returned empty steps. Falling back.");
                return simpleFallback(fallbackPrompt);
            }

            List<PlanStep> steps = stepsRaw.stream()
                    .map(step -> new PlanStep(
                            String.valueOf(step.get("step")),
                            castArgs(step.get("args"))
                    ))
                    .toList();

            return new AgentPlan(steps);

        } catch (Exception e) {
            log.error("Failed to parse planner JSON. Falling back. json={}", json, e);
            return simpleFallback(fallbackPrompt);
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> castArgs(Object argsObj) {
        if (argsObj instanceof Map<?, ?> rawMap) {
            return (Map<String, Object>) rawMap;
        }
        return Map.of();
    }

    private String extractJson(String response) {
        if (response == null || response.isBlank()) {
            return "{}";
        }

        int start = response.indexOf('{');
        int end = response.lastIndexOf('}');

        if (start >= 0 && end > start) {
            return response.substring(start, end + 1).trim();
        }

        return response.trim();
    }

    private String safe(String value) {
        return value == null ? "" : value;
    }
    private AgentPlan fallbackContinuationPlan(AgentSessionState state, String instruction) {
        String lower = instruction.toLowerCase();

        if (lower.contains("finance")) {
            return new AgentPlan(List.of(
                    new PlanStep("update_recipient", Map.of("recipient", "finance@company.com")),
                    new PlanStep("draft_email", Map.of(
                            "recipient", "finance@company.com",
                            "subject", state.email() != null ? state.email().subject() : "Requested Summary"
                    ))
            ));
        }

        if (lower.contains("shorten")) {
            return new AgentPlan(List.of(
                    new PlanStep("shorten_email", Map.of()),
                    new PlanStep("draft_email", Map.of(
                            "recipient", state.email() != null ? state.email().recipient() : "hr@company.com",
                            "subject", state.email() != null ? state.email().subject() : "Requested Summary"
                    ))
            ));
        }

        if (lower.contains("send")) {
            return new AgentPlan(List.of(
                    new PlanStep("send_email", Map.of(
                            "recipient", state.email() != null ? state.email().recipient() : "hr@company.com",
                            "subject", state.email() != null ? state.email().subject() : "Requested Summary"
                    ))
            ));
        }

        return new AgentPlan(List.of(
                new PlanStep("stop", Map.of())
        ));
    }
    private AgentPlan simpleFallback(String userPrompt) {
        return new AgentPlan(List.of(
                new PlanStep("research", Map.of("query", userPrompt)),
                new PlanStep("email", Map.of(
                        "recipient", "hr@company.com",
                        "subject", "Requested Summary"
                ))
        ));
    }
}