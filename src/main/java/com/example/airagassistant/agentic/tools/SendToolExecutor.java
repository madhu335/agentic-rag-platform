package com.example.airagassistant.agentic.tools;

import com.example.airagassistant.agentic.dto.AgentTool;
import com.example.airagassistant.agentic.dto.EmailToolRequest;
import com.example.airagassistant.agentic.dto.PlanStep;
import com.example.airagassistant.agentic.state.AgentSessionState;
import com.example.airagassistant.agentic.state.AgentStateStore;
import org.springframework.stereotype.Component;

@Component
public class SendToolExecutor implements AgentTool {

    private final EmailTool emailTool;
    private final AgentStateStore stateStore;

    public SendToolExecutor(EmailTool emailTool, AgentStateStore stateStore) {
        this.emailTool = emailTool;
        this.stateStore = stateStore;
    }

    @Override
    public String name() {
        return "send";
    }

    @Override
    public AgentSessionState execute(AgentSessionState state, PlanStep step) {
        String recipient = getArg(
                step,
                "recipient",
                state.email() != null && state.email().recipient() != null ? state.email().recipient() : "hr@company.com"
        );
        String subject = getArg(
                step,
                "subject",
                state.email() != null && state.email().subject() != null ? state.email().subject() : "Requested Summary"
        );

        emailTool.sendEmail(new EmailToolRequest(
                state.sessionId(),
                recipient,
                subject
        ));

        return stateStore.get(state.sessionId());
    }

    private String getArg(PlanStep step, String key, String defaultValue) {
        if (step.args() == null || step.args().get(key) == null) {
            return defaultValue;
        }
        String value = String.valueOf(step.args().get(key)).trim();
        return value.isEmpty() ? defaultValue : value;
    }
}