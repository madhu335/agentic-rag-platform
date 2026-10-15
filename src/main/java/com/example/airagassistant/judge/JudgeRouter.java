package com.example.airagassistant.judge;

import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Component;

import java.util.List;

@Primary
@Component
public class JudgeRouter implements JudgeClient {

    private final String provider;
    private final JudgeClient defaultJudge;
    private final JudgeClient claudeJudge;
    private final JudgeClient openAiJudge;
    private final JudgeClient haloJudge;

    public JudgeRouter(
            @Value("${judge.provider:default}") String provider,
            ObjectProvider<DefaultJudgeClient> defaultProvider,
            ObjectProvider<ClaudeJudgeClient> claudeProvider,
            ObjectProvider<OpenAiJudgeClient> openAiProvider,
            ObjectProvider<HaloJudgeClient> haloProvider
    ) {
        this.provider = provider;
        this.defaultJudge = defaultProvider.getIfAvailable();
        this.claudeJudge = claudeProvider.getIfAvailable();
        this.openAiJudge = openAiProvider.getIfAvailable();
        this.haloJudge = haloProvider.getIfAvailable();
    }

    @Override
    public String evaluate(String prompt, List<String> contextChunks) {

        JudgeClient client = switch (provider) {
            case "claude" -> require(claudeJudge, "claude");
            case "openai" -> require(openAiJudge, "openai");
            case "halo" -> require(haloJudge, "halo");
            case "default", "vllm", "ollama" -> require(defaultJudge, "default");
            default -> throw new IllegalStateException("Unsupported judge.provider: " + provider);
        };

        return client.evaluate(prompt, contextChunks);
    }

    private JudgeClient require(JudgeClient client, String name) {
        if (client == null) {
            throw new IllegalStateException("Judge provider not available: " + name);
        }
        return client;
    }
}
