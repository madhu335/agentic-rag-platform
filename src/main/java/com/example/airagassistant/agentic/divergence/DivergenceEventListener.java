package com.example.airagassistant.agentic.divergence;

import lombok.RequiredArgsConstructor;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class DivergenceEventListener {

    private final FallbackService fallbackService;

    @EventListener
    public void handleDivergence(DivergenceDetectedEvent event) {

        fallbackService.retryWithBetterQuery(
                event.getSessionId(),
                event.getOriginalQuery()
        );
    }
}