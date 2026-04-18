package com.example.airagassistant.agentic.multi.agents;

import com.example.airagassistant.agentic.multi.SubAgentResult;
import com.example.airagassistant.email.EmailService;
import com.example.airagassistant.sms.SmsService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Sub-agent: Communication
 *
 * Owns: email drafting/sending, SMS composing/sending.
 *
 * Unlike the single-agent approach where email body generation reads from
 * AgentSessionState.research(), this sub-agent receives the content to send
 * as an explicit parameter from the supervisor. This makes the data flow
 * visible: supervisor passes research result → communication agent formats
 * and sends it.
 *
 * Design choice: this agent does NOT use the LLM for email composition.
 * It wraps the content in a simple template. If you wanted LLM-generated
 * email prose, you'd add a prompt call here — but for now, the template
 * approach matches what the existing EmailToolExecutor does.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class CommunicationSubAgent {

    private static final String DEFAULT_RECIPIENT = "hr@company.com";
    private static final String DEFAULT_SUBJECT = "Requested Summary";
    private static final String DEFAULT_PHONE = "+10000000000";

    private final EmailService emailService;
    private final SmsService smsService;

    /**
     * @param task    natural-language task from the supervisor
     * @param content the content to send (usually from ResearchSubAgent or VehicleSubAgent)
     * @param args    domain-specific args: type (email/sms), recipient, subject, phoneNumber
     */
    public SubAgentResult execute(String task, String content, Map<String, Object> args) {
        String type = getStringArg(args, "type", inferType(task));

        if ("sms".equals(type)) {
            return executeSms(content, args);
        }

        return executeEmail(content, args);
    }

    // ─── Email ────────────────────────────────────────────────────────────

    private SubAgentResult executeEmail(String content, Map<String, Object> args) {
        String recipient = getStringArg(args, "recipient", DEFAULT_RECIPIENT);
        String subject = getStringArg(args, "subject", DEFAULT_SUBJECT);
        boolean sendNow = getBooleanArg(args, "send", false);

        String body = buildEmailBody(content);

        log.info("CommunicationSubAgent — email to='{}' subject='{}' send={}",
                recipient, subject, sendNow);

        try {
            if (sendNow) {
                emailService.sendEmail(recipient, subject, body);
            } else {
                emailService.createDraft(recipient, subject, body);
            }

            String status = sendNow ? "SENT" : "DRAFTED";

            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("type", "email");
            metadata.put("recipient", recipient);
            metadata.put("subject", subject);
            metadata.put("status", status);
            metadata.put("bodyPreview", body.length() > 100
                    ? body.substring(0, 100) + "..." : body);

            return SubAgentResult.success(
                    "communication",
                    "Email " + status.toLowerCase() + " to " + recipient,
                    List.of(), 1.0, null, metadata
            );

        } catch (Exception e) {
            log.error("CommunicationSubAgent — email failed: {}", e.getMessage());
            return SubAgentResult.failure("communication",
                    "Email failed: " + e.getMessage());
        }
    }

    // ─── SMS ──────────────────────────────────────────────────────────────

    private SubAgentResult executeSms(String content, Map<String, Object> args) {
        String phone = getStringArg(args, "phoneNumber", DEFAULT_PHONE);
        boolean sendNow = getBooleanArg(args, "send", true);

        String message = buildSmsMessage(content);

        log.info("CommunicationSubAgent — SMS to='{}' send={}", phone, sendNow);

        try {
            if (sendNow) {
                smsService.sendSms(phone, message);
            }

            String status = sendNow ? "SENT" : "COMPOSED";

            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("type", "sms");
            metadata.put("phoneNumber", phone);
            metadata.put("status", status);
            metadata.put("message", message);

            return SubAgentResult.success(
                    "communication",
                    "SMS " + status.toLowerCase() + " to " + phone,
                    List.of(), 1.0, null, metadata
            );

        } catch (Exception e) {
            log.error("CommunicationSubAgent — SMS failed: {}", e.getMessage());
            return SubAgentResult.failure("communication",
                    "SMS failed: " + e.getMessage());
        }
    }

    // ─── Templates ────────────────────────────────────────────────────────

    private String buildEmailBody(String content) {
        if (content == null || content.isBlank()) {
            return "Hello,\n\nNo content available.\n\nRegards";
        }
        return """
                Hello,
                
                Please find the requested summary below:
                
                %s
                
                Regards
                """.formatted(content);
    }

    private String buildSmsMessage(String content) {
        if (content == null || content.isBlank()) return "No content available.";
        return content.length() > 160
                ? content.substring(0, 160).trim() + "..."
                : content;
    }

    // ─── Helpers ──────────────────────────────────────────────────────────

    private String inferType(String task) {
        if (task == null) return "email";
        String lower = task.toLowerCase();
        if (lower.contains("sms") || lower.contains("text")) return "sms";
        return "email";
    }

    private String getStringArg(Map<String, Object> args, String key, String fallback) {
        if (args == null) return fallback;
        Object v = args.get(key);
        if (v == null) return fallback;
        String s = String.valueOf(v).trim();
        return s.isEmpty() ? fallback : s;
    }

    private boolean getBooleanArg(Map<String, Object> args, String key, boolean fallback) {
        if (args == null) return fallback;
        Object v = args.get(key);
        if (v == null) return fallback;
        if (v instanceof Boolean b) return b;
        return Boolean.parseBoolean(String.valueOf(v).trim());
    }
}
