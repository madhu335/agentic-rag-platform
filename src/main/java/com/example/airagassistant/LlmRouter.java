package com.example.airagassistant;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.function.Consumer;

@Primary
@Component
public class LlmRouter implements LlmClient {

    private final OpenAiClient openAi;
    private final OllamaClient ollama;

    @Value("${llm.provider:openai}")
    private String provider;

    public LlmRouter(OpenAiClient openAi, OllamaClient ollama) {
        this.openAi = openAi;
        this.ollama = ollama;
    }

    @Override
    public String answer(String question, List<String> contextChunks) {
        System.out.println("LlmRouter provider=" + provider);
        return switch (provider.toLowerCase()) {
            case "ollama" -> ollama.answer(question, contextChunks);
            case "openai" -> openAi.answer(question, contextChunks);
            default -> throw new IllegalArgumentException("Unknown llm.provider: " + provider);
        };
    }
    public void streamAnswer(String question, List<String> contextChunks, Consumer<String> onToken) {
        if ("ollama".equalsIgnoreCase(provider)) {
            ollama.streamAnswer(question, contextChunks, onToken);
        } else {
            ollama.streamAnswer(question, contextChunks, onToken);
        }
    }
}
