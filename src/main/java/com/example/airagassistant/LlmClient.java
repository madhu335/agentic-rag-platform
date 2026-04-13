package com.example.airagassistant;

import java.util.List;

public interface LlmClient {
    String answer(String question, List<String> contextChunks);
}
