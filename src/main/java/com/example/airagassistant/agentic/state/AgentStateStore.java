package com.example.airagassistant.agentic.state;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

@Component
@RequiredArgsConstructor
@Slf4j
public class AgentStateStore {

    private final AgentSessionRepository repository;
    private final ObjectMapper objectMapper;

    @Transactional
    public AgentSessionState save(AgentSessionState state) {
        try {
            String json = objectMapper.writeValueAsString(state);

            AgentSessionEntity entity = repository.findById(state.sessionId())
                    .orElseGet(() -> AgentSessionEntity.builder()
                            .sessionId(state.sessionId())
                            .build());

            entity.setOriginalUserRequest(state.originalUserRequest());
            entity.setCurrentUserRequest(state.currentUserRequest());
            entity.setDocId(state.docId());
            entity.setStateJson(json);
            entity.setCreatedAt(state.createdAt());
            entity.setUpdatedAt(state.updatedAt());

            repository.save(entity);
            return state;

        } catch (JsonProcessingException e) {
            throw new IllegalStateException("Failed to serialize agent session state for sessionId=" + state.sessionId(), e);
        }
    }

    @Transactional(readOnly = true)
    public AgentSessionState get(String sessionId) {
        return repository.findById(sessionId)
                .map(this::deserialize)
                .orElse(null);
    }

    @Transactional
    public void delete(String sessionId) {
        repository.deleteById(sessionId);
    }

    private AgentSessionState deserialize(AgentSessionEntity entity) {
        try {
            return objectMapper.readValue(entity.getStateJson(), AgentSessionState.class);
        } catch (Exception e) {
            log.error("Failed to deserialize session state. sessionId={}", entity.getSessionId(), e);
            throw new IllegalStateException("Failed to deserialize agent session state for sessionId=" + entity.getSessionId(), e);
        }
    }
}