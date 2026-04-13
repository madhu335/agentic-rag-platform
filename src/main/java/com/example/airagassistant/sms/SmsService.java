package com.example.airagassistant.sms;

public interface SmsService {
    void sendSms(String phoneNumber, String message);
}