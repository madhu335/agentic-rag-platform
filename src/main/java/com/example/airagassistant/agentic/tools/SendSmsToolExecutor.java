package com.example.airagassistant.agentic.tools;

import com.example.airagassistant.agentic.dto.AgentTool;
import com.example.airagassistant.agentic.dto.PlanStep;
import com.example.airagassistant.agentic.state.AgentSessionState;
import com.example.airagassistant.sms.SmsService;
import org.springframework.stereotype.Component;

@Component
public class SendSmsToolExecutor implements AgentTool {

    private static final String DEFAULT_PHONE = "+10000000000";
    private final SmsService smsService;

    public SendSmsToolExecutor(SmsService smsService) { this.smsService = smsService; }

    @Override public String name() { return "send_sms"; }

    @Override
    public AgentSessionState execute(AgentSessionState state, PlanStep step) {
        String phone = getArg(step, "phoneNumber",
                state.sms() != null && state.sms().phoneNumber() != null
                        ? state.sms().phoneNumber() : DEFAULT_PHONE);
        String message = state.sms() != null && state.sms().message() != null
                ? state.sms().message() : "";

        smsService.sendSms(phone, message);

        return state.withSms(new AgentSessionState.SmsSnapshot(
                phone, message, AgentSessionState.SmsStatus.SENT));
    }

    private String getArg(PlanStep step, String key, String def) {
        if (step.args() == null || step.args().get(key) == null) return def;
        String v = String.valueOf(step.args().get(key)).trim();
        return v.isEmpty() ? def : v;
    }
}