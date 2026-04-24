package com.example.airagassistant.judge;

import com.example.airagassistant.LlmClient;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class DefaultJudgeClient implements JudgeClient {

    private final LlmClient llmClient;

    public DefaultJudgeClient(LlmClient llmClient) {
        this.llmClient = llmClient;
    }

    @Override
    public String evaluate(String prompt, List<String> contextChunks) {
        return llmClient.answer(prompt, contextChunks);
    }
}