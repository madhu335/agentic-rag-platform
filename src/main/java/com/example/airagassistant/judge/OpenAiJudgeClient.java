package com.example.airagassistant.judge;

import com.example.airagassistant.LlmClient;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@ConditionalOnProperty(name = "judge.provider", havingValue = "openai")
public class OpenAiJudgeClient implements JudgeClient {

    private final LlmClient openAiClient;

    public OpenAiJudgeClient(@Qualifier("openAiLlmClient") LlmClient openAiClient) {
        this.openAiClient = openAiClient;
    }

    @Override
    public String evaluate(String prompt, List<String> contextChunks) {
        return openAiClient.answer(prompt, contextChunks);
    }
}