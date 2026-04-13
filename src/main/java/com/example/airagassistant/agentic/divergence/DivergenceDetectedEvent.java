package com.example.airagassistant.agentic.divergence;

import lombok.Getter;
import org.springframework.context.ApplicationEvent;

@Getter
public class DivergenceDetectedEvent extends ApplicationEvent {

    private final String sessionId;
    private final String originalQuery;

    public DivergenceDetectedEvent(Object source, String sessionId, String originalQuery) {
        super(source);
        this.sessionId = sessionId;
        this.originalQuery = originalQuery;
    }
}