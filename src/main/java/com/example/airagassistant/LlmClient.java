package com.example.airagassistant;

import java.util.List;
import java.util.function.Consumer;

public interface LlmClient {

    String answer(String question, List<String> contextChunks);

    void streamAnswer(String question, List<String> contextChunks, Consumer<String> onToken);
}