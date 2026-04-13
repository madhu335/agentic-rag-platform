package com.example.airagassistant.email;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Slf4j
@Service
public class SmtpEmailService implements EmailService {

    @Override
    public void createDraft(String to, String subject, String body) {
        log.info("Drafting email to={} subject={}", to, subject);
        // TODO draft integration
    }

    @Override
    public void sendEmail(String to, String subject, String body) {
        log.info("Sending email to={} subject={}", to, subject);
        // TODO actual SMTP send integration
    }
}