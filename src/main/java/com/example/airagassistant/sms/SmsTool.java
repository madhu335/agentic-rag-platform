package com.example.airagassistant.sms;

import com.example.airagassistant.agentic.state.AgentSessionState;
import com.example.airagassistant.agentic.state.AgentStateStore;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

@Slf4j
@Component
@RequiredArgsConstructor
public class SmsTool {

    private final AgentStateStore stateStore;
    private final SmsService smsService;

    public AgentSessionState sendSms(String sessionId, String phoneNumber, String message) {
        if (sessionId == null || sessionId.isBlank()) throw new IllegalArgumentException("sessionId is required");
        if (phoneNumber == null || phoneNumber.isBlank()) throw new IllegalArgumentException("phoneNumber is required");
        if (message == null || message.isBlank()) throw new IllegalArgumentException("message is required");

        AgentSessionState state = stateStore.get(sessionId);
        if (state == null) throw new IllegalArgumentException("No session state found for sessionId=" + sessionId);

        smsService.sendSms(phoneNumber, message);

        AgentSessionState updated = state.withSms(new AgentSessionState.SmsSnapshot(
                phoneNumber, message, AgentSessionState.SmsStatus.SENT));

        stateStore.save(updated);
        return updated;
    }
}