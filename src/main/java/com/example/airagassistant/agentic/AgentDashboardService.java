package com.example.airagassistant.agentic;

import com.example.airagassistant.agentic.dto.AgentSessionSummaryDto;
import com.example.airagassistant.agentic.state.AgentSessionEntity;
import com.example.airagassistant.agentic.state.AgentSessionRepository;
import com.example.airagassistant.agentic.state.AgentSessionState;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

@Service
@RequiredArgsConstructor
@Slf4j
public class AgentDashboardService {

    private final AgentSessionRepository repository;
    private final ObjectMapper objectMapper;

    public List<AgentSessionSummaryDto> listRecentSessions(int limit) {
        int safeLimit = Math.max(1, Math.min(limit, 100));

        return repository.findAllByOrderByUpdatedAtDesc(PageRequest.of(0, safeLimit))
                .stream()
                .map(this::toSummary)
                .filter(s -> s != null)
                .sorted(Comparator.comparing(AgentSessionSummaryDto::updatedAt).reversed())
                .toList();
    }

    public AgentSessionState getSession(String sessionId) {
        return repository.findById(sessionId)
                .map(this::deserialize)
                .orElse(null);
    }

    private AgentSessionSummaryDto toSummary(AgentSessionEntity entity) {
        try {
            AgentSessionState state = deserialize(entity);

            List<String> stepNames = state.history() == null
                    ? List.of()
                    : state.history().stream()
                    .map(h -> h.step())
                    .filter(s -> s != null && !s.isBlank())
                    .distinct()
                    .toList();

            Double judgeScore = state.research() != null && state.research().judge() != null
                    ? state.research().judge().score()
                    : null;

            Boolean judgeGrounded = state.research() != null && state.research().judge() != null
                    ? state.research().judge().grounded()
                    : null;

            return new AgentSessionSummaryDto(
                    state.sessionId(),
                    detectFlowType(state),
                    detectStatus(state),
                    state.originalUserRequest(),
                    state.currentUserRequest(),
                    state.docId(),
                    state.researchAttempts(),
                    state.research() != null ? state.research().retrievalScore() : null,
                    judgeScore,
                    judgeGrounded,
                    state.email() != null && state.email().status() != null ? state.email().status().name() : "NOT_PRESENT",
                    state.sms() != null && state.sms().status() != null ? state.sms().status().name() : "NOT_PRESENT",
                    state.history() != null ? state.history().size() : 0,
                    stepNames,
                    state.createdAt(),
                    state.updatedAt()
            );
        } catch (Exception e) {
            log.warn("Failed to build dashboard summary for session {}", entity.getSessionId(), e);
            return null;
        }
    }

    private AgentSessionState deserialize(AgentSessionEntity entity) {
        try {
            return objectMapper.readValue(entity.getStateJson(), AgentSessionState.class);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to deserialize session " + entity.getSessionId(), e);
        }
    }

    private String detectFlowType(AgentSessionState state) {
        String request = ((state.originalUserRequest() == null ? "" : state.originalUserRequest()) + " "
                + (state.currentUserRequest() == null ? "" : state.currentUserRequest())).toLowerCase(Locale.ROOT);

        List<String> steps = state.history() == null
                ? List.of()
                : state.history().stream().map(h -> h.step().toLowerCase(Locale.ROOT)).toList();

        boolean hasVehicle = steps.stream().anyMatch(s ->
                s.contains("vehicle") || s.contains("compare_vehicles") || s.contains("fetch_vehicle_specs") || s.contains("enrich_vehicle_data"));

        boolean hasCms = request.contains("cms")
                || request.contains("ingestion")
                || request.contains("rag")
                || request.contains("retrieval")
                || request.contains("document flow");

        if (hasVehicle) {
            return "VEHICLE";
        }
        if (hasCms) {
            return "CMS";
        }
        return "DOC";
    }

    private String detectStatus(AgentSessionState state) {
        if (state.history() == null || state.history().isEmpty()) {
            return "NEW";
        }

        boolean hasFailure = state.history().stream()
                .anyMatch(h -> h.status() != null && "FAILED".equalsIgnoreCase(h.status()));

        if (hasFailure) {
            return "FAILED";
        }

        boolean completed = state.history().stream()
                .anyMatch(h -> "continue".equalsIgnoreCase(h.step())
                        && h.resultSummary() != null
                        && h.resultSummary().toLowerCase(Locale.ROOT).contains("already completed"));

        if (completed) {
            return "COMPLETED";
        }

        boolean hasStop = state.history().stream()
                .anyMatch(h -> "stop".equalsIgnoreCase(h.step()));

        if (hasStop) {
            return "STOPPED";
        }

        boolean hasStarted = state.history().stream()
                .anyMatch(h -> h.status() != null && "STARTED".equalsIgnoreCase(h.status()));

        return hasStarted ? "RUNNING" : "COMPLETED";
    }
}