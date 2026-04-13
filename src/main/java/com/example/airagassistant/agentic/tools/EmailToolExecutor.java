package com.example.airagassistant.agentic.tools;

import com.example.airagassistant.agentic.dto.AgentTool;
import com.example.airagassistant.agentic.dto.PlanStep;
import com.example.airagassistant.agentic.state.AgentSessionState;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
public class EmailToolExecutor implements AgentTool {

    private static final String DEFAULT_RECIPIENT = "hr@company.com";
    private static final String DEFAULT_SUBJECT   = "Requested Summary";

    @Override public String name() { return "email"; }

    @Override
    public AgentSessionState execute(AgentSessionState state, PlanStep step) {
        Map<String, Object> args = step != null && step.args() != null ? step.args() : Map.of();

        String recipient = getStringArg(args, "recipient",
                state.email() != null && state.email().recipient() != null
                        ? state.email().recipient() : DEFAULT_RECIPIENT);

        String subject = getStringArg(args, "subject",
                state.email() != null && state.email().subject() != null
                        ? state.email().subject() : DEFAULT_SUBJECT);

        String body = buildEmailBody(state);

        return state.withEmail(new AgentSessionState.EmailSnapshot(
                recipient, subject, body, AgentSessionState.EmailStatus.NOT_STARTED));
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

    private String getStringArg(Map<String, Object> args, String key, String def) {
        Object v = args.get(key);
        if (v == null) return def;
        String s = String.valueOf(v).trim();
        return s.isEmpty() ? def : s;
    }
}