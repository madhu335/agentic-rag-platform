package com.example.airagassistant.agentic.tools;

import com.example.airagassistant.agentic.dto.AgentTool;
import com.example.airagassistant.agentic.dto.PlanStep;
import com.example.airagassistant.agentic.state.AgentSessionState;
import org.springframework.stereotype.Component;

@Component
public class UpdateRecipientToolExecutor implements AgentTool {

    @Override public String name() { return "update_recipient"; }

    @Override
    public AgentSessionState execute(AgentSessionState state, PlanStep step) {
        if (state.email() == null) return state;
        String recipient = getArg(step, "recipient", state.email().recipient());
        return state.withEmail(new AgentSessionState.EmailSnapshot(
                recipient, state.email().subject(),
                state.email().body(), AgentSessionState.EmailStatus.NOT_STARTED));
    }

    private String getArg(PlanStep step, String key, String def) {
        if (step.args() == null || step.args().get(key) == null) return def;
        String v = String.valueOf(step.args().get(key)).trim();
        return v.isEmpty() ? def : v;
    }
}