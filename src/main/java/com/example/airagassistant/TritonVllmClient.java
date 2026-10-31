package com.example.airagassistant;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Generator served by Triton Inference Server (vLLM backend), e.g. Nemotron-3-Nano-30B-A3B NVFP4.
 *
 * Uses Triton's native generate extension:
 *   POST /v2/models/{model}/generate          -> {"text_output": "..."}
 *   POST /v2/models/{model}/generate_stream   -> SSE, data: {"text_output": "..."}
 *
 * Triton's raw generate endpoint does not apply a chat template, so this client renders the
 * model's template itself. The format below was copied from the model's own tokenizer
 * (apply_chat_template, enable_thinking=False):
 *
 *   <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think></think>
 *
 * The empty think block turns Nemotron's reasoning off (same intent as the judge's low reasoning).
 */
@Slf4j
@Component
@ConditionalOnProperty(name = "llm.provider", havingValue = "triton-vllm")
public class TritonVllmClient implements LlmClient {

    private static final String IM_START = "<|im_start|>";
    private static final String IM_END = "<|im_end|>";
    /**
     * Default system prompt. A "focused" prompt (answer only the named vehicle, 1-3 sentences) was
     * A/B tested on 2026-10-30 and did not reduce multi-variant answers (8/20 vs 5/20 bare), so the
     * bare prompt stays the default. Override with llm.triton.system-prompt.
     */
    private static final String DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant.";

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final WebClient webClient;
    private final String modelName;
    private final int maxTokens;
    private final double temperature;
    private final boolean enableThinking;
    private final Duration timeout;
    private final String systemPrompt;

    public TritonVllmClient(
            WebClient.Builder builder,
            @Value("${llm.triton.base-url}") String baseUrl,
            @Value("${llm.triton.model-name}") String modelName,
            @Value("${llm.triton.max-tokens:2048}") int maxTokens,
            @Value("${llm.triton.temperature:0.1}") double temperature,
            @Value("${llm.triton.enable-thinking:false}") boolean enableThinking,
            @Value("${llm.triton.timeout-seconds:120}") long timeoutSeconds,
            @Value("${llm.triton.system-prompt:}") String systemPrompt
    ) {
        this.webClient = builder.baseUrl(baseUrl).build();
        this.modelName = modelName;
        this.maxTokens = maxTokens;
        this.temperature = temperature;
        this.enableThinking = enableThinking;
        this.timeout = Duration.ofSeconds(timeoutSeconds);
        this.systemPrompt = (systemPrompt == null || systemPrompt.isBlank()) ? DEFAULT_SYSTEM_PROMPT : systemPrompt;
        log.info("TritonVllmClient -> {} model={} maxTokens={} temperature={} thinking={} systemPrompt={} chars",
                baseUrl, modelName, maxTokens, temperature, enableThinking, this.systemPrompt.length());
    }

    @Override
    public String answer(String question, List<String> contextChunks) {
        Map<String, Object> request = buildRequest(question, contextChunks, false);
        String promptText = (String) request.get("text_input");

        String raw = webClient.post()
                .uri("/v2/models/{model}/generate", modelName)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(request)
                .retrieve()
                .bodyToMono(String.class)
                .timeout(timeout)
                .block();

        try {
            JsonNode root = objectMapper.readTree(raw == null ? "{}" : raw);
            if (root.hasNonNull("error")) {
                throw new RuntimeException("Triton error: " + root.get("error").asText());
            }
            String text = root.path("text_output").asText("");
            String cleaned = cleanOutput(text);
            // #15 diagnostics: identical prompt hash + different output => engine non-determinism;
            // different prompt hash => the app sent different context/instructions.
            log.debug("Triton generate: promptChars={} promptHash={} chunks={} outChars={} outHash={}",
                    promptText.length(), Integer.toHexString(promptText.hashCode()),
                    contextChunks == null ? 0 : contextChunks.size(),
                    cleaned.length(), Integer.toHexString(cleaned.hashCode()));
            return cleaned.isBlank() ? "I don't know." : cleaned;
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException("Error parsing Triton generate response: " + raw, e);
        }
    }

    /**
     * Blocking stream: returns only after the last token, because RagAnswerService reads the
     * accumulated answer right after this call returns.
     */
    @Override
    public void streamAnswer(String question, List<String> contextChunks, Consumer<String> onToken) {
        Map<String, Object> request = buildRequest(question, contextChunks, true);
        StringBuilder accumulated = new StringBuilder();

        webClient.post()
                .uri("/v2/models/{model}/generate_stream", modelName)
                .contentType(MediaType.APPLICATION_JSON)
                .accept(MediaType.TEXT_EVENT_STREAM)
                .bodyValue(request)
                .retrieve()
                .bodyToFlux(new ParameterizedTypeReference<ServerSentEvent<String>>() {})
                .timeout(timeout)
                .doOnNext(event -> {
                    String data = event.data();
                    if (data == null || data.isBlank()) {
                        return;
                    }
                    String json = data.startsWith("data:") ? data.substring(5).trim() : data.trim();
                    try {
                        JsonNode node = objectMapper.readTree(json);
                        if (node.hasNonNull("error")) {
                            throw new RuntimeException("Triton stream error: " + node.get("error").asText());
                        }
                        String text = node.path("text_output").asText("");
                        if (text.isEmpty()) {
                            return;
                        }
                        // Handle both delta and cumulative chunks.
                        String delta;
                        if (accumulated.length() > 0 && text.startsWith(accumulated.toString())) {
                            delta = text.substring(accumulated.length());
                        } else {
                            delta = text;
                        }
                        if (accumulated.length() == 0) {
                            delta = stripLeading(delta);
                        }
                        if (delta.isEmpty()) {
                            return;
                        }
                        accumulated.append(delta);
                        onToken.accept(delta);
                    } catch (RuntimeException e) {
                        throw e;
                    } catch (Exception e) {
                        log.debug("Skipping unparseable Triton SSE chunk: {}", json);
                    }
                })
                .blockLast();
    }

    private Map<String, Object> buildRequest(String question, List<String> contextChunks, boolean stream) {
        Map<String, Object> parameters = new LinkedHashMap<>();
        parameters.put("stream", stream);
        parameters.put("exclude_input_in_output", true);
        parameters.put("max_tokens", maxTokens);
        parameters.put("temperature", temperature);

        Map<String, Object> request = new LinkedHashMap<>();
        request.put("text_input", renderChatTemplate(systemPrompt, buildUserMessage(question, contextChunks)));
        request.put("parameters", parameters);
        return request;
    }

    /** Nemotron-3 chat template (ChatML style), copied from the model's tokenizer. */
    private String renderChatTemplate(String system, String user) {
        return IM_START + "system\n" + system + IM_END + "\n"
                + IM_START + "user\n" + user + IM_END + "\n"
                + IM_START + "assistant\n"
                + (enableThinking ? "<think>\n" : "<think></think>");
    }

    /** Same RAG prompt body as VllmClient (context, question, "Answer:"). */
    private String buildUserMessage(String question, List<String> contextChunks) {
        StringBuilder sb = new StringBuilder();

        if (contextChunks != null && !contextChunks.isEmpty()) {
            sb.append("Context:\n");
            for (int i = 0; i < contextChunks.size(); i++) {
                sb.append(i + 1).append(". ").append(contextChunks.get(i)).append("\n");
            }
            sb.append("\n");
        }

        sb.append("Question:\n").append(question).append("\n\n");
        sb.append("Answer:");

        return sb.toString();
    }

    /** Drops any reasoning block and stray end-of-turn markers. */
    private String cleanOutput(String text) {
        String out = text;
        int thinkEnd = out.indexOf("</think>");
        if (thinkEnd >= 0) {
            out = out.substring(thinkEnd + "</think>".length());
        }
        out = out.replace(IM_END, "");
        return out.trim();
    }

    private String stripLeading(String s) {
        int i = 0;
        while (i < s.length() && Character.isWhitespace(s.charAt(i))) {
            i++;
        }
        return s.substring(i);
    }
}
