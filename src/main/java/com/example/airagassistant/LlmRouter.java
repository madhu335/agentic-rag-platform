package com.example.airagassistant;

import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.function.Consumer;

@Primary
@Component
public class LlmRouter implements LlmClient {

    private final String provider;
    private final OpenAiClient openAiClient;
    private final OllamaClient ollamaClient;
    private final TritonVllmClient tritonVllmClient;
    private final VllmClient vllmClient;

    public LlmRouter(
            @Value("${llm.provider:ollama}") String provider,
            ObjectProvider<OpenAiClient> openAiClientProvider,
            ObjectProvider<OllamaClient> ollamaClientProvider,
            ObjectProvider<TritonVllmClient> tritonVllmClientProvider,
            ObjectProvider<VllmClient> vllmClientProvider
    ) {
        this.provider = provider;
        this.openAiClient = openAiClientProvider.getIfAvailable();
        this.ollamaClient = ollamaClientProvider.getIfAvailable();
        this.tritonVllmClient = tritonVllmClientProvider.getIfAvailable();
        this.vllmClient = vllmClientProvider.getIfAvailable();
    }

    @Override
    public String answer(String question, List<String> contextChunks) {
        return activeClient().answer(question, contextChunks);
    }

    @Override
    public void streamAnswer(String question, List<String> contextChunks, Consumer<String> onToken) {
        activeClient().streamAnswer(question, contextChunks, onToken);
    }

    private LlmClient activeClient() {
        return switch (provider) {
            case "openai" -> require(openAiClient, "openai");
            case "triton-vllm" -> require(tritonVllmClient, "triton-vllm");
            case "ollama" -> require(ollamaClient, "ollama");
            case "vllm" -> require(vllmClient, "vllm");
            default -> throw new IllegalStateException("Unsupported llm.provider: " + provider);
        };
    }

    private LlmClient require(LlmClient client, String name) {
        if (client == null) {
            throw new IllegalStateException("LLM provider bean not available: " + name);
        }
        return client;
    }
}