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
    private final JudgeClient tritonJudge;

    public JudgeRouter(
            @Value("${judge.provider:default}") String provider,
            ObjectProvider<DefaultJudgeClient> defaultProvider,
            ObjectProvider<ClaudeJudgeClient> claudeProvider,
            ObjectProvider<OpenAiJudgeClient> openAiProvider,
            ObjectProvider<HaloJudgeClient> haloProvider,
            ObjectProvider<TritonJudgeClient> tritonProvider
    ) {
        this.provider = provider;
        this.defaultJudge = defaultProvider.getIfAvailable();
        this.claudeJudge = claudeProvider.getIfAvailable();
        this.openAiJudge = openAiProvider.getIfAvailable();
        this.haloJudge = haloProvider.getIfAvailable();
        this.tritonJudge = tritonProvider.getIfAvailable();
    }

    @Override
    public String evaluate(String prompt, List<String> contextChunks) {
        return active().evaluate(prompt, contextChunks);
    }

    @Override
    public boolean supportsFastMode() {
        return active().supportsFastMode();
    }

    @Override
    public String evaluateFast(String prompt, List<String> contextChunks) {
        return active().evaluateFast(prompt, contextChunks);
    }

    private JudgeClient active() {
        return switch (provider) {
            case "claude" -> require(claudeJudge, "claude");
            case "openai" -> require(openAiJudge, "openai");
            case "halo" -> require(haloJudge, "halo");
            case "triton" -> require(tritonJudge, "triton");
            case "default", "vllm", "ollama" -> require(defaultJudge, "default");
            default -> throw new IllegalStateException("Unsupported judge.provider: " + provider);
        };
    }

    private JudgeClient require(JudgeClient client, String name) {
        if (client == null) {
            throw new IllegalStateException("Judge provider not available: " + name);
        }
        return client;
    }
}
