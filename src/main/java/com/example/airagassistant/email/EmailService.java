package com.example.airagassistant.email;

public interface EmailService {
    void createDraft(String to, String subject, String body);
    void sendEmail(String to, String subject, String body);
}