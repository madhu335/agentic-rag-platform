package com.example.airagassistant.agentic.tools;

import com.example.airagassistant.agentic.dto.EmailToolRequest;
import com.example.airagassistant.agentic.dto.EmailToolResponse;
import com.example.airagassistant.agentic.state.AgentSessionState;
import com.example.airagassistant.agentic.state.AgentStateStore;
import com.example.airagassistant.email.EmailService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

@Slf4j
@Component
@RequiredArgsConstructor
public class EmailTool {

    private final AgentStateStore stateStore;
    private final EmailService emailService;

    public EmailToolResponse draftEmail(EmailToolRequest request) {
        log.info("EmailTool draft: sessionId={} to={} subject={}", request.sessionId(), request.to(), request.subject());
        validateRequest(request);
        String subject = normalizeSubject(request.subject());
        AgentSessionState state = requireState(request.sessionId());
        String body = buildEmailBody(state);
        emailService.createDraft(request.to(), subject, body);
        stateStore.save(state.withEmail(new AgentSessionState.EmailSnapshot(
                request.to(), subject, body, AgentSessionState.EmailStatus.DRAFTED)));
        return new EmailToolResponse(true, "DRAFTED");
    }

    public EmailToolResponse sendEmail(EmailToolRequest request) {
        log.info("EmailTool send: sessionId={} to={} subject={}", request.sessionId(), request.to(), request.subject());
        validateRequest(request);
        String subject = normalizeSubject(request.subject());
        AgentSessionState state = requireState(request.sessionId());
        String body = buildEmailBody(state);
        emailService.sendEmail(request.to(), subject, body);
        stateStore.save(state.withEmail(new AgentSessionState.EmailSnapshot(
                request.to(), subject, body, AgentSessionState.EmailStatus.SENT)));
        return new EmailToolResponse(true, "SENT");
    }

    private AgentSessionState requireState(String sessionId) {
        AgentSessionState state = stateStore.get(sessionId);
        if (state == null) throw new IllegalArgumentException("No session state found for sessionId=" + sessionId);
        return state;
    }

    private void validateRequest(EmailToolRequest request) {
        if (request == null) throw new IllegalArgumentException("request is required");
        if (request.sessionId() == null || request.sessionId().isBlank()) throw new IllegalArgumentException("sessionId is required");
        if (request.to() == null || request.to().isBlank()) throw new IllegalArgumentException("recipient email is required");
    }

    private String normalizeSubject(String subject) {
        return (subject == null || subject.isBlank()) ? "Requested Summary" : subject.trim();
    }

    private String buildEmailBody(AgentSessionState state) {
        if (state.email() != null && state.email().body() != null && !state.email().body().isBlank())
            return state.email().body();
        if (state.research() != null && state.research().summary() != null && !state.research().summary().isBlank())
            return """
                    Hello,

                    Please find the requested summary below:

                    %s

                    Citations:
                    %s

                    Regards
                    """.formatted(state.research().summary(),
                    state.research().citations() != null ? String.join(", ", state.research().citations()) : "");
        return """
                Hello,

                %s

                Regards
                """.formatted(state.currentUserRequest() != null ? state.currentUserRequest() : "");
    }
}