package com.example.airagassistant.judge;

import java.util.List;

public interface JudgeClient {
    String evaluate(String prompt, List<String> contextChunks);
}