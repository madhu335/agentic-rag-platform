package com.example.airagassistant.judge;

import com.example.airagassistant.LlmClient;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@ConditionalOnProperty(name = "judge.provider", havingValue = "ollama")
public class OllamaJudgeClient implements JudgeClient {

    private final LlmClient llmClient;

    public OllamaJudgeClient(LlmClient llmClient) {
        this.llmClient = llmClient;
    }

    @Override
    public String evaluate(String prompt, List<String> contextChunks) {
        return llmClient.answer(prompt, contextChunks);
    }
}