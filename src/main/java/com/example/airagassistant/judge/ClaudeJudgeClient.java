package com.example.airagassistant.judge;

import com.example.airagassistant.LlmClient;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@ConditionalOnProperty(name = "judge.provider", havingValue = "claude")
public class ClaudeJudgeClient implements JudgeClient {

    private final LlmClient claudeClient;

    public ClaudeJudgeClient(@Qualifier("claudeClient") LlmClient claudeClient) {
        this.claudeClient = claudeClient;
    }

    @Override
    public String evaluate(String prompt, List<String> contextChunks) {
        return claudeClient.answer(prompt, contextChunks);
    }
}