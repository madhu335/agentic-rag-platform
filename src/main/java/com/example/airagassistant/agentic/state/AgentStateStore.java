package com.example.airagassistant.agentic.state;

import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class AgentStateStore {

    private final Map<String, AgentSessionState> sessions = new ConcurrentHashMap<>();

    public AgentSessionState save(AgentSessionState state) {
        sessions.put(state.sessionId(), state);
        return state;
    }

    public AgentSessionState get(String sessionId) {
        return sessions.get(sessionId);
    }

    public void delete(String sessionId) {
        sessions.remove(sessionId);
    }
}