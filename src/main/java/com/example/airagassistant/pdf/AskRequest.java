package com.example.airagassistant.pdf;

import lombok.Data;

@Data
public class AskRequest {
    private String docId;
    private String question;
    private int topK;

    // getters + setters
}