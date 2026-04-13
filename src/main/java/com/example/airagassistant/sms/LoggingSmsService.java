package com.example.airagassistant.sms;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Slf4j
@Service
public class LoggingSmsService implements SmsService {

    @Override
    public void sendSms(String phoneNumber, String message) {
        log.info("Sending SMS to={} message={}", phoneNumber, message);
    }
}