package com.example.airagassistant.agentic.tools;

import com.example.airagassistant.agentic.dto.AgentTool;
import com.example.airagassistant.agentic.dto.PlanStep;
import com.example.airagassistant.agentic.state.AgentSessionState;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
public class SmsToolExecutor implements AgentTool {

    private static final String DEFAULT_PHONE = "+10000000000";

    @Override public String name() { return "compose_sms"; }

    @Override
    public AgentSessionState execute(AgentSessionState state, PlanStep step) {
        Map<String, Object> args = step != null && step.args() != null ? step.args() : Map.of();
        String phone = getStringArg(args, "phoneNumber",
                state.sms() != null && state.sms().phoneNumber() != null
                        ? state.sms().phoneNumber() : DEFAULT_PHONE);

        return state.withSms(new AgentSessionState.SmsSnapshot(
                phone, buildSmsMessage(state), AgentSessionState.SmsStatus.COMPOSED));
    }

    private String buildSmsMessage(AgentSessionState state) {
        if (state.sms() != null && state.sms().message() != null && !state.sms().message().isBlank())
            return state.sms().message();
        if (state.research() != null && state.research().summary() != null && !state.research().summary().isBlank()) {
            String s = state.research().summary();
            return s.length() > 160 ? s.substring(0, 160).trim() + "..." : s;
        }
        String f = state.currentUserRequest() != null ? state.currentUserRequest() : "";
        return f.length() > 160 ? f.substring(0, 160).trim() + "..." : f;
    }

    private String getStringArg(Map<String, Object> args, String key, String def) {
        Object v = args.get(key);
        if (v == null) return def;
        String s = String.valueOf(v).trim();
        return s.isEmpty() ? def : s;
    }
}