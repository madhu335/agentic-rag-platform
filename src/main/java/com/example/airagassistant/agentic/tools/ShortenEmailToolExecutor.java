package com.example.airagassistant.agentic.tools;

import com.example.airagassistant.agentic.dto.AgentTool;
import com.example.airagassistant.agentic.dto.PlanStep;
import com.example.airagassistant.agentic.state.AgentSessionState;
import org.springframework.stereotype.Component;

@Component
public class ShortenEmailToolExecutor implements AgentTool {

    @Override public String name() { return "shorten_email"; }

    @Override
    public AgentSessionState execute(AgentSessionState state, PlanStep step) {
        if (state.email() == null) return state;

        String shorterBody;
        if (state.research() != null && state.research().summary() != null && !state.research().summary().isBlank()) {
            String summary = state.research().summary();
            String shortened = summary.length() > 300 ? summary.substring(0, 300).trim() + "..." : summary;
            shorterBody = """
                    Hello,

                    Please find the requested short summary below:

                    %s

                    Citations:
                    %s

                    Regards
                    """.formatted(shortened,
                    state.research().citations() != null ? String.join(", ", state.research().citations()) : "");
        } else {
            String current = state.email().body() != null ? state.email().body() : "";
            shorterBody = current.length() > 300 ? current.substring(0, 300).trim() + "..." : current;
        }

        return state.withEmail(new AgentSessionState.EmailSnapshot(
                state.email().recipient(), state.email().subject(),
                shorterBody, AgentSessionState.EmailStatus.NOT_STARTED));
    }
}